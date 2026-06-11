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
//!      - Bounded by [`MIN_BUDGET`, `MAX_BUDGET`].
//!
//! 2. **Per-request timeout**, derived from observed throughput. After each
//!    success we record `chars_per_sec` (EWMA). Future timeouts are
//!    `base + safety × (batch_chars / chars_per_sec)` — never need a
//!    static `timeout_secs`. Bootstrap uses a conservative chars/sec
//!    estimate (~500 chars/s, lower bound on CPU jina-code) until we have
//!    a real measurement.
//!
//! Modeled after TCP congestion control: small failure cost in exchange for
//! never needing static configuration of server capacity *or* throughput.

use std::time::Duration;
use tracing::{debug, warn};

/// Lower bound on the budget. If a single chunk exceeds even this and still
/// fails, it's permanently unembeddable on this server — caller skips it
/// with a WARN.
pub const MIN_BUDGET: usize = 1024;

/// Upper bound on the budget. Even if the server is happy with bigger
/// payloads, we cap here to keep individual requests timely and bound
/// memory pressure on both ends.
pub const MAX_BUDGET: usize = 256_000;

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
/// Per-request timeout never goes above this. Beyond this, the bottleneck
/// is almost always something we can't time-our-way-out-of (real hang,
/// crash); AIMD halving the batch is the better response.
const TIMEOUT_CEILING_SECS: f64 = 300.0;

pub struct AdaptiveBatcher {
    budget: usize,
    consecutive_ok: u32,
    /// EWMA of observed chars-per-second across successful batches.
    /// None until the first measurement; bootstrap uses
    /// [`BOOTSTRAP_CHARS_PER_SEC`] in that case.
    chars_per_sec: Option<f64>,
}

impl AdaptiveBatcher {
    pub fn new(initial: usize) -> Self {
        Self {
            budget: initial.clamp(MIN_BUDGET, MAX_BUDGET),
            consecutive_ok: 0,
            chars_per_sec: None,
        }
    }

    pub fn budget(&self) -> usize {
        self.budget
    }

    /// Observed throughput (chars/sec), if at least one batch has succeeded.
    /// `None` until the first observation; bootstrap path uses
    /// [`BOOTSTRAP_CHARS_PER_SEC`] internally for timeout estimation.
    pub fn chars_per_sec(&self) -> Option<f64> {
        self.chars_per_sec
    }

    /// Estimate a per-request timeout based on observed (or bootstrap)
    /// throughput. Floors at [`TIMEOUT_FLOOR_SECS`], ceils at
    /// [`TIMEOUT_CEILING_SECS`].
    pub fn estimate_timeout(&self, batch_chars: usize) -> Duration {
        let cps = self.chars_per_sec.unwrap_or(BOOTSTRAP_CHARS_PER_SEC);
        let estimated_secs = batch_chars as f64 / cps;
        let secs = TIMEOUT_BASE_SECS + TIMEOUT_SAFETY_FACTOR * estimated_secs;
        Duration::from_secs_f64(secs.clamp(TIMEOUT_FLOOR_SECS, TIMEOUT_CEILING_SECS))
    }

    /// Pick the next batch from `texts[start..]`, returning the exclusive
    /// end index. Always includes at least one element so progress is
    /// guaranteed even when a single chunk exceeds the current budget
    /// (the caller's failure path then skips it).
    pub fn pack(&self, texts: &[String], start: usize) -> usize {
        let mut end = start;
        let mut total = 0usize;
        while end < texts.len() {
            let len = texts[end].len();
            if end > start && total + len > self.budget {
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
        // Throughput update — guard against div-by-zero on tiny batches.
        let secs = elapsed.as_secs_f64().max(0.001);
        let observed = batch_chars as f64 / secs;
        self.chars_per_sec = Some(match self.chars_per_sec {
            None => observed,
            Some(prev) => (1.0 - THROUGHPUT_EWMA_ALPHA) * prev + THROUGHPUT_EWMA_ALPHA * observed,
        });

        self.consecutive_ok = self.consecutive_ok.saturating_add(1);
        if self.consecutive_ok >= INCREASE_THRESHOLD && self.budget < MAX_BUDGET {
            let new = ((self.budget as f64) * INCREASE_FACTOR) as usize;
            let new = new.clamp(self.budget + 1, MAX_BUDGET);
            debug!(
                old = self.budget,
                new, "adaptive batcher: increasing budget"
            );
            self.budget = new;
            self.consecutive_ok = 0;
        }
    }

    /// Halve the budget on failure. Returns the new budget. Will not go
    /// below [`MIN_BUDGET`] — at the floor, the caller treats the input as
    /// unembeddable.
    pub fn note_failure(&mut self) -> usize {
        self.consecutive_ok = 0;
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

    #[test]
    fn pack_always_advances() {
        // Even when a single chunk dwarfs the budget, pack must include it
        // (otherwise the caller's loop would stall). Single chunks above
        // the budget are caught by the caller's failure path, not here.
        let b = AdaptiveBatcher::new(MIN_BUDGET);
        let huge = MIN_BUDGET * 10;
        let texts = vec!["x".repeat(huge)];
        assert_eq!(b.pack(&texts, 0), 1);
    }

    #[test]
    fn pack_respects_budget() {
        // Budgets below MIN_BUDGET get clamped, so build the test above it.
        let budget = 5_000;
        let b = AdaptiveBatcher::new(budget);
        let chunk_bytes = 2_000;
        let texts: Vec<String> = (0..10).map(|_| "x".repeat(chunk_bytes)).collect();
        // 2000-byte chunks, budget 5000 → 2 fit (2000+2000=4000; +2000 would be 6000).
        assert_eq!(b.pack(&texts, 0), 2);
    }

    #[test]
    fn increase_after_threshold() {
        let mut b = AdaptiveBatcher::new(10_000);
        let initial = b.budget();
        for _ in 0..INCREASE_THRESHOLD {
            b.note_success(5_000, Duration::from_secs(10));
        }
        assert!(b.budget() > initial);
    }

    #[test]
    fn throughput_ewma_updates() {
        let mut b = AdaptiveBatcher::new(10_000);
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
        let b = AdaptiveBatcher::new(10_000);
        // 10 KB with bootstrap 500 chars/s → 20s estimated, base 10s, ×3 safety = 70s.
        let t = b.estimate_timeout(10_000);
        assert!(t.as_secs() >= 60 && t.as_secs() <= 80, "got {:?}", t);
    }

    #[test]
    fn estimate_timeout_uses_observed_throughput() {
        let mut b = AdaptiveBatcher::new(10_000);
        // Observed: 1000 chars/s (much faster than bootstrap).
        b.note_success(10_000, Duration::from_secs(10));
        // 10 KB at 1000 chars/s = 10s estimated, ×3 + 10 base = 40s.
        let t = b.estimate_timeout(10_000);
        assert!(t.as_secs() >= 35 && t.as_secs() <= 45, "got {:?}", t);
    }

    #[test]
    fn estimate_timeout_floors() {
        let b = AdaptiveBatcher::new(10_000);
        let t = b.estimate_timeout(0);
        assert!(t.as_secs() >= 15);
    }

    #[test]
    fn estimate_timeout_ceils() {
        let mut b = AdaptiveBatcher::new(10_000);
        b.note_success(1, Duration::from_secs(10)); // extremely slow: 0.1 chars/s
        let t = b.estimate_timeout(1_000_000);
        assert!(t.as_secs() <= 300);
    }

    #[test]
    fn halve_on_failure() {
        let mut b = AdaptiveBatcher::new(10_000);
        b.note_failure();
        assert_eq!(b.budget(), 5_000);
    }

    #[test]
    fn halve_floors_at_min() {
        let mut b = AdaptiveBatcher::new(MIN_BUDGET);
        b.note_failure();
        assert_eq!(b.budget(), MIN_BUDGET);
    }

    #[test]
    fn clamps_initial() {
        assert_eq!(AdaptiveBatcher::new(0).budget(), MIN_BUDGET);
        assert_eq!(AdaptiveBatcher::new(usize::MAX).budget(), MAX_BUDGET);
    }
}
