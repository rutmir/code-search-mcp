//! AIMD adaptive batcher for embedding requests.
//!
//! Replaces the static `batch_size` / `max_input_chars` / `timeout_secs`
//! config knobs with runtime self-tuning. Two things are adapted:
//!
//! 1. **Budget** (in bytes — llama.cpp tokenization roughly scales with bytes
//!    for code). Floats based on observed server behavior:
//!      - After [`INCREASE_THRESHOLD`] consecutive successes, +25%
//!        (multiplicative — fast probe upward).
//!      - On any failure, ÷2 (multiplicative decrease).
//!      - Bounded below by [`MIN_BUDGET`], above by whichever is smaller:
//!        [`MAX_BUDGET`] (a memory-pressure bound) or the work that fits
//!        in [`TARGET_BATCH_SECS`] at the observed throughput. The second
//!        is the one that matters on a slow host: without it the budget
//!        climbs to a size that host can never finish in time, times out,
//!        halves, and climbs again — a full timeout burnt per cycle.
//!
//!    A failure also arms a one-shot **retry ceiling** at half the size of
//!    the batch that actually failed. Halving the budget alone does not
//!    guarantee a smaller retry: a batch under the *new* budget packs
//!    identically, so the same request goes out again and burns another
//!    full timeout before anything changes. The ceiling makes every retry
//!    strictly smaller; it is cleared by the next success so the budget
//!    remains the thing that carries learned capacity between files.
//!
//! 2. **Per-request timeout**, derived from observed throughput. After each
//!    success we record `chars_per_sec` (EWMA). Future timeouts are
//!    `base + safety × (batch_chars / chars_per_sec)` — never need a
//!    static `timeout_secs`. Bootstrap uses a conservative chars/sec
//!    estimate (~500 chars/s, lower bound on CPU jina-code) until we have
//!    a real measurement.
//!
//!    That timeout is kept strictly shorter than the transport's own hard
//!    cap (`[embedding].timeout_secs`). When the two can both fire — as
//!    they could when both were 300 s — a slow batch and a dead connection
//!    arrive as the same error, and the batcher shrinks the budget over
//!    what was really a wrong estimate of how long the server takes.
//!
//!    A timeout also feeds *back* into the estimate: a batch that did not
//!    finish in `elapsed` proves the server is slower than
//!    `chars / elapsed`, so the EWMA is corrected downward. Halving alone
//!    is forgotten after five successes; a corrected estimate lowers the
//!    derived budget ceiling for good.
//!
//! Modeled after TCP congestion control: small failure cost in exchange for
//! never needing static configuration of server capacity *or* throughput.

use std::time::Duration;
use tracing::{debug, warn};

/// Lower bound on the budget. If a single chunk exceeds even this and still
/// fails, it's permanently unembeddable on this server — caller skips it
/// with a WARN.
pub const MIN_BUDGET: usize = 1024;

/// Absolute upper bound on the budget, regardless of how fast the server
/// turns out to be. This one is about memory pressure on both ends, not
/// about capacity — capacity is derived (see [`AdaptiveBatcher::budget_ceiling`]).
pub const MAX_BUDGET: usize = 256_000;

/// How long a single embedding request should aim to take, whatever the
/// host. This is a policy choice, not a claim about any server: it fixes
/// how much work is lost when a batch fails and how coarse progress
/// reporting is, and those should not differ between a laptop and a GPU
/// box. The budget ceiling is derived from it and the observed
/// throughput, which is what stops the budget climbing to a size the host
/// could never finish in time.
const TARGET_BATCH_SECS: f64 = 30.0;

/// Fraction of the transport's hard timeout that the batcher's own
/// per-request timeout may use. Keeping our timeout strictly the shorter
/// one means a slow batch fails as "this batch was too big for the time
/// we gave it" rather than as an indistinguishable transport error.
const TIMEOUT_HEADROOM: f64 = 0.9;

/// Successful batches in a row before we try growing the budget.
const INCREASE_THRESHOLD: u32 = 5;
/// Multiplicative growth factor on increase. 1.25 = +25%.
const INCREASE_FACTOR: f64 = 1.25;

/// Conservative throughput estimate used for the very first batch (before
/// we have observations). 500 chars/sec ≈ 167 tok/s × 3 chars/tok is a safe
/// lower bound for CPU-bound jina-code-embeddings. Overshooting on bootstrap
/// just means a longer-than-needed timeout; undershooting means false-failure.
const BOOTSTRAP_CHARS_PER_SEC: f64 = 500.0;

/// EWMA smoothing factor for throughput updates. 0.3 means each new
/// observation pulls the estimate ~30% toward the new value, converging
/// within ~5-10 batches.
const THROUGHPUT_EWMA_ALPHA: f64 = 0.3;

/// Multiplier on the estimated processing time, to absorb jitter, GC
/// pauses, and other server-side variance.
const TIMEOUT_SAFETY_FACTOR: f64 = 3.0;
/// Fixed overhead: network RTT, server slot scheduling, JSON parsing.
const TIMEOUT_BASE_SECS: f64 = 10.0;
/// Per-request timeout never goes below this — even a tiny request needs
/// enough headroom for the server to actually pick it up.
const TIMEOUT_FLOOR_SECS: f64 = 15.0;

pub struct AdaptiveBatcher {
    budget: usize,
    consecutive_ok: u32,
    /// EWMA of observed chars-per-second across successful batches.
    /// None until the first measurement; bootstrap uses
    /// [`BOOTSTRAP_CHARS_PER_SEC`] in that case.
    chars_per_sec: Option<f64>,
    /// Armed by [`note_failure`] at half the failing batch's size, cleared
    /// by the next success. Caps [`pack`] in addition to the budget, which
    /// is what makes a retry strictly smaller than what just failed.
    retry_ceiling: Option<usize>,
    /// The transport's hard timeout (`[embedding].timeout_secs`). The
    /// batcher needs to know it so its own derived timeout can stay
    /// strictly shorter; when the two are equal — as they were, both 300 —
    /// whichever fires is a coin toss, and a transport error and an
    /// oversized batch become indistinguishable at the point where the
    /// batcher decides what to shrink.
    request_timeout: Duration,
}

impl AdaptiveBatcher {
    pub fn new(initial: usize, request_timeout: Duration) -> Self {
        Self {
            budget: initial.clamp(MIN_BUDGET, MAX_BUDGET),
            consecutive_ok: 0,
            chars_per_sec: None,
            retry_ceiling: None,
            request_timeout,
        }
    }

    pub fn budget(&self) -> usize {
        self.budget
    }

    /// Longest per-request timeout the batcher will ask for: a fraction of
    /// the transport's hard cap, so ours always fires first.
    fn timeout_ceiling_secs(&self) -> f64 {
        self.request_timeout.as_secs_f64() * TIMEOUT_HEADROOM
    }

    /// Largest budget worth having on *this* host, derived from observed
    /// throughput rather than assumed.
    ///
    /// Without it the budget grows +25% per five successes toward a fixed
    /// 256 KB. On a CPU-bound server that size cannot finish inside any
    /// sane timeout, so the loop climbed to the cap, timed out, halved,
    /// and climbed again — paying a full timeout per cycle, indefinitely.
    /// Observed doing exactly that through a multi-hour reindex.
    ///
    /// The fix is not a smaller constant, which would merely move the
    /// problem to a faster machine. Once throughput is known, the ceiling
    /// is the work that fits in [`TARGET_BATCH_SECS`]; until then the
    /// absolute cap stands and AIMD probes as before.
    fn budget_ceiling(&self) -> usize {
        match self.chars_per_sec {
            Some(cps) => ((cps * TARGET_BATCH_SECS) as usize).clamp(MIN_BUDGET, MAX_BUDGET),
            None => MAX_BUDGET,
        }
    }

    /// Ceiling in force for the next batch, when a failure has armed one.
    /// Exposed so the caller can log why a retry is smaller than the budget
    /// would suggest.
    pub fn retry_ceiling(&self) -> Option<usize> {
        self.retry_ceiling
    }

    /// Chars the next batch may occupy: the budget, further capped by any
    /// armed retry ceiling.
    fn pack_limit(&self) -> usize {
        match self.retry_ceiling {
            Some(ceiling) => self.budget.min(ceiling),
            None => self.budget,
        }
    }

    /// Observed throughput (chars/sec), if at least one batch has succeeded.
    /// `None` until the first observation; bootstrap path uses
    /// [`BOOTSTRAP_CHARS_PER_SEC`] internally for timeout estimation.
    pub fn chars_per_sec(&self) -> Option<f64> {
        self.chars_per_sec
    }

    /// Estimate a per-request timeout based on observed (or bootstrap)
    /// throughput. Floors at [`TIMEOUT_FLOOR_SECS`], and never reaches the
    /// transport's hard cap — see [`timeout_ceiling_secs`].
    pub fn estimate_timeout(&self, batch_chars: usize) -> Duration {
        let cps = self.chars_per_sec.unwrap_or(BOOTSTRAP_CHARS_PER_SEC);
        let estimated_secs = batch_chars as f64 / cps;
        let secs = TIMEOUT_BASE_SECS + TIMEOUT_SAFETY_FACTOR * estimated_secs;
        let ceiling = self.timeout_ceiling_secs();
        // A tiny configured timeout can put the ceiling below the floor;
        // the ceiling wins, since exceeding it guarantees the transport
        // kills the request first and we learn nothing.
        Duration::from_secs_f64(secs.clamp(TIMEOUT_FLOOR_SECS.min(ceiling), ceiling))
    }

    /// Pick the next batch from `texts[start..]`, returning the exclusive
    /// end index. Always includes at least one element so progress is
    /// guaranteed even when a single chunk exceeds the current limit
    /// (the caller's failure path then skips it).
    pub fn pack(&self, texts: &[String], start: usize) -> usize {
        let limit = self.pack_limit();
        let mut end = start;
        let mut total = 0usize;
        while end < texts.len() {
            let len = texts[end].len();
            if end > start && total + len > limit {
                break;
            }
            total += len;
            end += 1;
        }
        end
    }

    /// Record a successful batch. Updates both the throughput EWMA (for
    /// future timeout estimates) and the AIMD success counter (which grows
    /// the budget after [`INCREASE_THRESHOLD`] in a row).
    pub fn note_success(&mut self, batch_chars: usize, elapsed: Duration) {
        // The retry ceiling exists only to force the *next* attempt smaller
        // after a failure; once something goes through, the budget is once
        // again the sole authority on batch size.
        self.retry_ceiling = None;

        // Throughput update — guard against div-by-zero on tiny batches.
        let secs = elapsed.as_secs_f64().max(0.001);
        let observed = batch_chars as f64 / secs;
        self.chars_per_sec = Some(match self.chars_per_sec {
            None => observed,
            Some(prev) => (1.0 - THROUGHPUT_EWMA_ALPHA) * prev + THROUGHPUT_EWMA_ALPHA * observed,
        });

        self.consecutive_ok = self.consecutive_ok.saturating_add(1);
        let ceiling = self.budget_ceiling();
        if self.consecutive_ok >= INCREASE_THRESHOLD && self.budget < ceiling {
            let new = ((self.budget as f64) * INCREASE_FACTOR) as usize;
            let new = new.clamp(self.budget + 1, ceiling);
            debug!(
                old = self.budget,
                new, ceiling, "adaptive batcher: increasing budget"
            );
            self.budget = new;
            self.consecutive_ok = 0;
        }
    }

    /// Record a failed batch of `failed_batch_chars`: halve the budget and
    /// arm a retry ceiling at half the failing size. Returns the new budget.
    ///
    /// Both are needed. The budget is the long-lived estimate of server
    /// capacity and must survive into the next file; the ceiling is what
    /// guarantees the *immediate* retry is smaller. Without it, a batch that
    /// already fit under the halved budget repacks identically and the same
    /// request is re-sent — paying another full timeout to learn nothing.
    ///
    /// The budget will not go below [`MIN_BUDGET`]; at the floor the caller
    /// treats a single oversized chunk as unembeddable.
    pub fn note_failure(&mut self, failed_batch_chars: usize, elapsed: Duration) -> usize {
        self.consecutive_ok = 0;

        // A batch that ran out of time is evidence about *throughput*, not
        // only about size: it demonstrably did not finish in `elapsed`, so
        // the server is slower than `chars / elapsed`. Folding that in is
        // what keeps the budget from climbing back to a level this host was
        // never capable of — halving alone is forgotten after five
        // successes, while a corrected estimate lowers the derived ceiling
        // for good.
        //
        // Self-guarding for the other failure modes: a batch rejected
        // outright (4xx, "input too large") comes back in milliseconds, so
        // the implied bound is enormous and the estimate is left alone.
        let secs = elapsed.as_secs_f64().max(0.001);
        let implied_max = failed_batch_chars as f64 / secs;
        if self.chars_per_sec.is_none_or(|cps| cps > implied_max) {
            debug!(
                previous = ?self.chars_per_sec,
                implied_max,
                "adaptive batcher: timeout contradicts the throughput estimate; lowering it"
            );
            self.chars_per_sec = Some(implied_max);
        }

        // At least 1: a ceiling of 0 would be meaningless, and `pack` always
        // takes one element regardless.
        let ceiling = (failed_batch_chars / 2).max(1);
        self.retry_ceiling = Some(match self.retry_ceiling {
            // Repeated failures keep tightening rather than resetting to a
            // value derived from the (already smaller) latest attempt.
            Some(prev) => prev.min(ceiling),
            None => ceiling,
        });

        let old = self.budget;
        let new = (self.budget / 2).max(MIN_BUDGET);
        if new < old {
            warn!(old, new, "adaptive batcher: halving budget on failure");
        }
        self.budget = new;
        new
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The default `[embedding].timeout_secs`, which is what most of these
    /// cases care about only indirectly.
    const TEST_REQUEST_TIMEOUT: Duration = Duration::from_secs(300);

    /// An outright rejection comes back immediately; used where a test
    /// means "this failed, but not by running out of time".
    const FAST: Duration = Duration::from_millis(20);

    impl AdaptiveBatcher {
        fn new_for_test(initial: usize) -> Self {
            Self::new(initial, TEST_REQUEST_TIMEOUT)
        }
    }

    #[test]
    fn pack_always_advances() {
        // Even when a single chunk dwarfs the budget, pack must include it
        // (otherwise the caller's loop would stall). Single chunks above
        // the budget are caught by the caller's failure path, not here.
        let b = AdaptiveBatcher::new_for_test(MIN_BUDGET);
        let huge = MIN_BUDGET * 10;
        let texts = vec!["x".repeat(huge)];
        assert_eq!(b.pack(&texts, 0), 1);
    }

    #[test]
    fn pack_respects_budget() {
        // Budgets below MIN_BUDGET get clamped, so build the test above it.
        let budget = 5_000;
        let b = AdaptiveBatcher::new_for_test(budget);
        let chunk_bytes = 2_000;
        let texts: Vec<String> = (0..10).map(|_| "x".repeat(chunk_bytes)).collect();
        // 2000-byte chunks, budget 5000 → 2 fit (2000+2000=4000; +2000 would be 6000).
        assert_eq!(b.pack(&texts, 0), 2);
    }

    #[test]
    fn increase_after_threshold() {
        let mut b = AdaptiveBatcher::new_for_test(10_000);
        let initial = b.budget();
        for _ in 0..INCREASE_THRESHOLD {
            b.note_success(5_000, Duration::from_secs(10));
        }
        assert!(b.budget() > initial);
    }

    #[test]
    fn throughput_ewma_updates() {
        let mut b = AdaptiveBatcher::new_for_test(10_000);
        assert!(b.chars_per_sec().is_none());
        b.note_success(10_000, Duration::from_secs(10)); // 1000 chars/s
        assert_eq!(b.chars_per_sec(), Some(1000.0));
        b.note_success(5_000, Duration::from_secs(10)); // 500 chars/s
                                                        // EWMA with alpha=0.3: 0.7*1000 + 0.3*500 = 850
        let cps = b.chars_per_sec().unwrap();
        assert!((cps - 850.0).abs() < 0.01, "got {}", cps);
    }

    #[test]
    fn estimate_timeout_uses_bootstrap_when_unobserved() {
        let b = AdaptiveBatcher::new_for_test(10_000);
        // 10 KB with bootstrap 500 chars/s → 20s estimated, base 10s, ×3 safety = 70s.
        let t = b.estimate_timeout(10_000);
        assert!(t.as_secs() >= 60 && t.as_secs() <= 80, "got {:?}", t);
    }

    #[test]
    fn estimate_timeout_uses_observed_throughput() {
        let mut b = AdaptiveBatcher::new_for_test(10_000);
        // Observed: 1000 chars/s (much faster than bootstrap).
        b.note_success(10_000, Duration::from_secs(10));
        // 10 KB at 1000 chars/s = 10s estimated, ×3 + 10 base = 40s.
        let t = b.estimate_timeout(10_000);
        assert!(t.as_secs() >= 35 && t.as_secs() <= 45, "got {:?}", t);
    }

    #[test]
    fn estimate_timeout_floors() {
        let b = AdaptiveBatcher::new_for_test(10_000);
        let t = b.estimate_timeout(0);
        assert!(t.as_secs() >= 15);
    }

    #[test]
    fn estimate_timeout_ceils() {
        let mut b = AdaptiveBatcher::new_for_test(10_000);
        b.note_success(1, Duration::from_secs(10)); // extremely slow: 0.1 chars/s
        let t = b.estimate_timeout(1_000_000);
        assert!(t.as_secs() <= 300);
    }

    #[test]
    fn halve_on_failure() {
        let mut b = AdaptiveBatcher::new_for_test(10_000);
        b.note_failure(10_000, FAST);
        assert_eq!(b.budget(), 5_000);
    }

    #[test]
    fn halve_floors_at_min() {
        let mut b = AdaptiveBatcher::new_for_test(MIN_BUDGET);
        b.note_failure(MIN_BUDGET, FAST);
        assert_eq!(b.budget(), MIN_BUDGET);
    }

    /// The regression this whole mechanism exists for. A batch that fits
    /// comfortably under the *halved* budget used to repack identically,
    /// so the retry re-sent the same request and waited out another full
    /// timeout before anything changed. Observed in the field: 17 chunks /
    /// 118 075 chars failed at budget 256 000, halved to 128 000, and the
    /// identical 17-chunk batch went out again.
    #[test]
    fn retry_after_failure_is_strictly_smaller() {
        let chunk = 7_000;
        let texts: Vec<String> = (0..17).map(|_| "x".repeat(chunk)).collect();
        let mut b = AdaptiveBatcher::new_for_test(256_000);

        let end = b.pack(&texts, 0);
        assert_eq!(end, 17, "all chunks fit the initial budget");
        let failed_chars: usize = texts[..end].iter().map(|t| t.len()).sum();
        assert!(
            failed_chars < 128_000,
            "precondition: the failing batch fits under the halved budget"
        );

        b.note_failure(failed_chars, FAST);
        let retry_end = b.pack(&texts, 0);
        assert!(
            retry_end < end,
            "retry must be smaller: got {retry_end} chunks, was {end}"
        );
        let retry_chars: usize = texts[..retry_end].iter().map(|t| t.len()).sum();
        assert!(
            retry_chars <= failed_chars / 2,
            "retry {retry_chars} should be at most half of {failed_chars}"
        );
    }

    /// Repeated failures must keep converging, never widen back out.
    #[test]
    fn repeated_failures_keep_shrinking() {
        let texts: Vec<String> = (0..64).map(|_| "x".repeat(4_000)).collect();
        let mut b = AdaptiveBatcher::new_for_test(MAX_BUDGET);

        let mut prev = b.pack(&texts, 0);
        for round in 0..6 {
            let chars: usize = texts[..prev].iter().map(|t| t.len()).sum();
            b.note_failure(chars, FAST);
            let next = b.pack(&texts, 0);
            assert!(
                next <= prev,
                "round {round}: batch grew from {prev} to {next}"
            );
            if prev > 1 {
                assert!(next < prev, "round {round}: batch stalled at {prev}");
            }
            prev = next;
        }
        // Converges to the single-chunk case the caller handles by skipping.
        assert_eq!(prev, 1);
    }

    /// Progress is still guaranteed when even one chunk is over the ceiling —
    /// `pack` must hand the caller that chunk so it can be skipped, not
    /// return an empty batch and stall the loop.
    #[test]
    fn pack_advances_even_under_a_tight_ceiling() {
        let texts = vec!["x".repeat(50_000), "y".repeat(50_000)];
        let mut b = AdaptiveBatcher::new_for_test(MAX_BUDGET);
        b.note_failure(100, FAST);
        assert_eq!(b.pack(&texts, 0), 1);
    }

    /// The ceiling is a one-shot brake, not a permanent cap: once a batch
    /// succeeds the budget is again the only limit, so throughput can
    /// recover after a transient failure.
    #[test]
    fn success_clears_the_retry_ceiling() {
        let texts: Vec<String> = (0..10).map(|_| "x".repeat(5_000)).collect();
        let mut b = AdaptiveBatcher::new_for_test(MAX_BUDGET);

        b.note_failure(50_000, FAST);
        assert!(b.retry_ceiling().is_some());
        let constrained = b.pack(&texts, 0);
        assert!(constrained < 10);

        b.note_success(constrained * 5_000, Duration::from_secs(1));
        assert!(b.retry_ceiling().is_none());
        // Budget was halved twice over from MAX but still admits all ten.
        assert_eq!(b.pack(&texts, 0), 10);
    }

    /// Feed the batcher `n` successes at a given throughput, as a real run
    /// would, so the growth path and the EWMA both see it.
    fn converge(b: &mut AdaptiveBatcher, chars_per_sec: f64, rounds: usize) {
        for _ in 0..rounds {
            let chars = b.budget();
            let secs = chars as f64 / chars_per_sec;
            b.note_success(chars, Duration::from_secs_f64(secs));
        }
    }

    /// The regression this exists for: on a slow host the budget used to
    /// climb to a fixed 256 KB, which that host cannot process inside any
    /// sane timeout, so it timed out, halved, and climbed again forever.
    #[test]
    fn budget_stops_growing_at_what_the_host_can_actually_do() {
        // ~450 chars/s is what the CPU-bound reference host measured.
        let mut b = AdaptiveBatcher::new_for_test(10_000);
        converge(&mut b, 450.0, 60);

        let ceiling = (450.0 * TARGET_BATCH_SECS) as usize;
        assert!(
            b.budget() <= ceiling,
            "budget {} exceeded the derived ceiling {}",
            b.budget(),
            ceiling
        );
        assert!(
            b.budget() < MAX_BUDGET / 4,
            "budget {} is still climbing toward the absolute cap",
            b.budget()
        );
        // And a batch at that budget stays comfortably inside the timeout,
        // which is the property whose absence caused the oscillation.
        let t = b.estimate_timeout(b.budget()).as_secs_f64();
        assert!(
            t < b.timeout_ceiling_secs(),
            "a full batch needs {t}s against a {}s ceiling",
            b.timeout_ceiling_secs()
        );
    }

    /// The same rule must not punish a fast host: there the absolute cap is
    /// the binding one, exactly as before.
    #[test]
    fn fast_host_is_limited_only_by_the_absolute_cap() {
        let mut b = AdaptiveBatcher::new_for_test(10_000);
        converge(&mut b, 50_000.0, 80);
        assert_eq!(b.budget(), MAX_BUDGET);
    }

    #[test]
    fn derived_timeout_never_reaches_the_transport_cap() {
        // Even an absurd batch against a crawling server must leave the
        // transport's own timeout unused, so a failure is attributable.
        let mut b = AdaptiveBatcher::new(MAX_BUDGET, Duration::from_secs(300));
        b.note_success(1, Duration::from_secs(100)); // 0.01 chars/s
        let t = b.estimate_timeout(10_000_000);
        assert!(t < Duration::from_secs(300), "got {t:?}");
        assert!((t.as_secs_f64() - 300.0 * TIMEOUT_HEADROOM).abs() < 1.0);
    }

    #[test]
    fn tiny_configured_timeout_still_yields_a_usable_timeout() {
        // Ceiling below the floor: the ceiling has to win, or every request
        // is killed by the transport before our own timeout means anything.
        let b = AdaptiveBatcher::new(10_000, Duration::from_secs(10));
        let t = b.estimate_timeout(100_000);
        assert!(t <= Duration::from_secs(9), "got {t:?}");
        assert!(t > Duration::ZERO);
    }

    /// A timeout says something about throughput, not just about size. The
    /// budget halving is forgotten after five successes; a corrected
    /// estimate lowers the derived ceiling for good.
    #[test]
    fn a_timeout_corrects_the_throughput_estimate() {
        let mut b = AdaptiveBatcher::new_for_test(100_000);
        converge(&mut b, 5_000.0, 10);
        let optimistic = b.chars_per_sec().unwrap();

        // 100 000 chars still unfinished after 100 s ⇒ at most 1000 chars/s.
        b.note_failure(100_000, Duration::from_secs(100));
        let corrected = b.chars_per_sec().unwrap();
        assert!(
            corrected < optimistic,
            "estimate stayed at {optimistic} after contradicting evidence"
        );
        assert!((corrected - 1_000.0).abs() < 1.0, "got {corrected}");
    }

    #[test]
    fn an_outright_rejection_leaves_the_estimate_alone() {
        // A 4xx comes back in milliseconds; the implied bound is enormous
        // and says nothing about how fast the server embeds.
        let mut b = AdaptiveBatcher::new_for_test(10_000);
        converge(&mut b, 800.0, 10);
        let before = b.chars_per_sec().unwrap();
        b.note_failure(10_000, FAST);
        assert_eq!(b.chars_per_sec().unwrap(), before);
    }

    #[test]
    fn clamps_initial() {
        assert_eq!(AdaptiveBatcher::new_for_test(0).budget(), MIN_BUDGET);
        assert_eq!(
            AdaptiveBatcher::new_for_test(usize::MAX).budget(),
            MAX_BUDGET
        );
    }
}
