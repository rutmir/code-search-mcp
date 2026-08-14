//! Hybrid search: dense (Qdrant) ∪ sparse (BM25/tantivy) → RRF merge →
//! optional cross-encoder rerank fused back into the RRF → ranked list.
//!
//! Why RRF instead of normalized score weighting? Dense uses cosine
//! similarity (0..1, near-1 for relevant), BM25 uses a raw score
//! (~0..30, no upper bound, log-scale). Linear combination requires
//! per-modality normalization that depends on the score distribution
//! of the corpus — fragile. Reciprocal Rank Fusion sidesteps this by
//! only looking at *rank*, which is comparable across modalities.
//!
//! The reranker is folded in the same way: it contributes a third
//! rank-vote (weighted by `[search].rerank_weight`) on top of the
//! dense+sparse RRF, instead of replacing the final score outright.
//! A cross-encoder is the best single judge of relevance, but giving
//! it veto power lets one model error sink a candidate both retrieval
//! modalities agree on — fusing ranks keeps the retrieval consensus
//! in play.
//!
//! Everything a query needs lives in [`SearchContext`], built once per
//! process. `serve` holds one for its whole lifetime: the HTTP clients
//! keep their connection pools warm, the tantivy reader stays open, and
//! the project-identity marker is verified at startup rather than on
//! every keystroke of a Claude Code session.

use anyhow::{Context, Result};
use std::collections::HashMap;
use std::sync::{Arc, Mutex, MutexGuard};
use tracing::{debug, warn};

use crate::bm25::{Bm25Search, ChunkText};
use crate::config::Config;
use crate::embedding;
use crate::reranker;
use crate::vector_store;

/// Quality-first defaults used when `[search]` block is absent or fields
/// are unset. See `SearchConfig` for the per-project knobs.
const DEFAULT_K_DENSE: usize = 30;
const DEFAULT_K_SPARSE: usize = 30;
const DEFAULT_RERANK_TOP_N: usize = 30;
const DEFAULT_RRF_K: usize = 60;
const DEFAULT_RERANK_WEIGHT: f32 = 2.0;
const DEFAULT_SYMBOL_BOOST: f32 = 1.0;

/// How much deeper to retrieve when a `path` filter is active.
///
/// `lang` is pushed down into both stores, but `path` is a substring
/// match neither can express cheaply (Qdrant needs a full-text payload
/// index and then matches *tokens*, not substrings; tantivy's `file`
/// field is raw-tokenized). So `path` is applied after retrieval — which
/// starves the result set unless the pool is widened first: scoping to
/// `docs/` in a repo whose top-30 is all code otherwise returns nothing.
/// Reranking still only ever sees `rerank_top_n` candidates, so the extra
/// depth costs retrieval bandwidth, not cross-encoder time.
const PATH_FILTER_OVERSAMPLE: usize = 10;

/// Ceiling on the widened pool, so a pathological config can't ask either
/// store for an unbounded page.
const MAX_RETRIEVAL_K: usize = 500;

#[derive(Debug, Clone)]
pub struct SearchResult {
    pub file: String,
    pub start_line: u64,
    pub end_line: u64,
    pub lang: String,
    /// Final score used for ranking. With rerank on: dense+sparse RRF
    /// plus the reranker's weighted rank-vote (see module docs).
    /// Without rerank: plain RRF score.
    pub score: f32,
    /// Component scores, for inspection / debugging.
    pub dense_score: Option<f32>,
    pub sparse_score: Option<f32>,
    pub rerank_score: Option<f32>,
    /// Either the full chunk text (if we have it) or the Qdrant snippet.
    pub preview: String,
    /// AST-aware metadata, when available: `("fn", "Foo::bar")`,
    /// `("struct", "Baz")`, etc. None for line-window / heading chunks.
    pub kind: Option<String>,
    pub name: Option<String>,
}

/// Coarse phase of a search, reported through [`SearchParams::progress`].
/// Kept deliberately coarse: the point is telling a waiting MCP client
/// that a multi-second search is alive and where it is, not fine-grained
/// instrumentation (that's what `tracing` is for).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SearchStage {
    Embedding,
    Retrieving,
    Reranking,
    Finalizing,
}

impl SearchStage {
    pub const TOTAL: u32 = 4;

    pub fn step(self) -> u32 {
        match self {
            SearchStage::Embedding => 1,
            SearchStage::Retrieving => 2,
            SearchStage::Reranking => 3,
            SearchStage::Finalizing => 4,
        }
    }

    pub fn label(self) -> &'static str {
        match self {
            SearchStage::Embedding => "embedding query",
            SearchStage::Retrieving => "retrieving candidates",
            SearchStage::Reranking => "reranking",
            SearchStage::Finalizing => "finalizing",
        }
    }
}

/// Callback invoked as the search moves between stages.
pub type ProgressSink<'a> = &'a (dyn Fn(SearchStage) + Send + Sync);

pub struct SearchParams<'a> {
    pub query: &'a str,
    pub limit: usize,
    pub use_rerank: bool,
    pub lang: Option<&'a str>,
    pub path: Option<&'a str>,
    /// Optional progress callback. `None` for the CLI, `Some` when the
    /// MCP client supplied a `_meta.progressToken`.
    pub progress: Option<ProgressSink<'a>>,
}

impl SearchParams<'_> {
    fn report(&self, stage: SearchStage) {
        if let Some(sink) = self.progress {
            sink(stage);
        }
    }
}

/// Everything a query needs, built once and reused.
pub struct SearchContext {
    config: Config,
    embedder: embedding::Client,
    vs: vector_store::Client,
    reranker: Option<reranker::Client>,
    /// Opened lazily and dropped on failure so the next query retries.
    /// Two situations make an eagerly-opened handle wrong: `serve` starts
    /// before its background watcher has built the index for a brand-new
    /// project, and a `config_hard` change deletes and recreates the
    /// directory underneath a running process.
    bm25: Mutex<Option<Arc<Bm25Search>>>,
}

impl SearchContext {
    /// Build the context and verify the collection belongs to this project.
    ///
    /// The marker check is a startup concern, not a per-query one: it
    /// guards against a misconfigured `vector_store.collection` pointing
    /// at another project's data, and that config can't change without
    /// restarting the process.
    pub async fn new(config: &Config) -> Result<Self> {
        let vs = vector_store::Client::new(
            &config.vector_store,
            config.vector_store.resolve_collection_name(&config.project),
        )
        .await?;
        vs.verify_marker_read_only(config).await?;

        Ok(Self {
            config: config.clone(),
            embedder: embedding::Client::new(&config.embedding),
            vs,
            reranker: config
                .reranker
                .as_ref()
                .filter(|r| r.enabled)
                .map(reranker::Client::new),
            bm25: Mutex::new(None),
        })
    }

    /// A poisoned lock here means some earlier query panicked while
    /// holding it. The guarded value is a cache handle, not an invariant
    /// that a panic could have half-updated, so recovering beats taking
    /// the whole server down for the rest of its life.
    fn bm25_slot(&self) -> MutexGuard<'_, Option<Arc<Bm25Search>>> {
        self.bm25.lock().unwrap_or_else(|e| e.into_inner())
    }

    fn bm25(&self) -> Result<Arc<Bm25Search>> {
        // The lock is taken and released twice on purpose: `Bm25Search::open`
        // does file I/O, and holding a std Mutex across it would serialize
        // every concurrent query behind one open. Two threads racing here
        // both get a valid handle; one of them simply wins the cache slot.
        {
            let slot = self.bm25_slot();
            if let Some(existing) = slot.as_ref() {
                return Ok(Arc::clone(existing));
            }
        }
        let opened = Arc::new(Bm25Search::open(&self.config.bm25.index_path)?);
        *self.bm25_slot() = Some(Arc::clone(&opened));
        Ok(opened)
    }

    fn invalidate_bm25(&self) {
        *self.bm25_slot() = None;
    }

    /// Full text of every indexed chunk of `file` overlapping the
    /// inclusive line range. Backs the `code_read_chunk` MCP tool.
    pub fn read_chunks(&self, file: &str, start: u64, end: u64) -> Result<Vec<ChunkText>> {
        let bm25 = self.bm25()?;
        match bm25.chunks_in_range(file, start, end) {
            Ok(v) => Ok(v),
            Err(e) => {
                self.invalidate_bm25();
                Err(e)
            }
        }
    }

    pub async fn search(&self, params: SearchParams<'_>) -> Result<Vec<SearchResult>> {
        let config = &self.config;
        let started = std::time::Instant::now();

        let rerank_top_n = config.search.rerank_top_n.unwrap_or(DEFAULT_RERANK_TOP_N);
        let rrf_k = config.search.rrf_k.unwrap_or(DEFAULT_RRF_K);
        // Widen the pool up front when a post-retrieval filter is in play.
        let depth = |base: usize| {
            if params.path.is_some() {
                base.saturating_mul(PATH_FILTER_OVERSAMPLE)
                    .min(MAX_RETRIEVAL_K)
            } else {
                base
            }
        };
        let k_dense = depth(config.search.dense_k.unwrap_or(DEFAULT_K_DENSE));
        let k_sparse = depth(config.search.sparse_k.unwrap_or(DEFAULT_K_SPARSE));

        // 1. Embed the query. Same model as indexing — no asymmetric query/doc
        //    encoders here; jina-code-embeddings is symmetric.
        params.report(SearchStage::Embedding);
        let mut vecs = self
            .embedder
            .embed(vec![params.query.to_string()])
            .await
            .context("embedding query")?;
        let query_vec = vecs.pop().context("empty embedding response for query")?;
        debug!(
            elapsed_ms = started.elapsed().as_millis() as u64,
            "query embedded"
        );

        // 2. Dense + sparse in parallel. `lang` is a hard restriction on
        //    both sides — pushing it down keeps each store's page full of
        //    candidates the caller can actually use.
        params.report(SearchStage::Retrieving);
        let bm25 = self.bm25()?;
        let (dense_res, sparse_res) =
            tokio::join!(self.vs.search(&query_vec, k_dense, params.lang), async {
                bm25.search(params.query, k_sparse, params.lang)
            });
        let dense_hits = dense_res.context("dense search")?;
        let sparse_hits = match sparse_res {
            Ok(hits) => hits,
            Err(e) => {
                self.invalidate_bm25();
                return Err(e).context("bm25 search");
            }
        };
        debug!(
            dense = dense_hits.len(),
            sparse = sparse_hits.len(),
            k_dense,
            k_sparse,
            elapsed_ms = started.elapsed().as_millis() as u64,
            "candidates gathered"
        );

        // 3. Merge by chunk_id using Reciprocal Rank Fusion.
        let mut by_id: HashMap<String, Candidate> = HashMap::new();
        for (rank, h) in dense_hits.iter().enumerate() {
            let c = by_id
                .entry(h.chunk_id.clone())
                .or_insert_with(|| Candidate {
                    chunk_id: h.chunk_id.clone(),
                    file: h.file.clone(),
                    start_line: h.start_line,
                    end_line: h.end_line,
                    lang: h.lang.clone(),
                    preview: h.snippet.clone(),
                    content: None,
                    rrf: 0.0,
                    dense_score: None,
                    sparse_score: None,
                    kind: h.kind.clone(),
                    name: h.name.clone(),
                });
            c.dense_score = Some(h.score);
            c.rrf += 1.0 / (rrf_k as f32 + rank as f32 + 1.0);
        }
        for (rank, h) in sparse_hits.iter().enumerate() {
            let c = by_id
                .entry(h.chunk_id.clone())
                .or_insert_with(|| Candidate {
                    chunk_id: h.chunk_id.clone(),
                    file: h.file.clone(),
                    start_line: h.start_line,
                    end_line: h.end_line,
                    lang: h.lang.clone(),
                    preview: h.content.chars().take(200).collect(),
                    content: Some(h.content.clone()),
                    rrf: 0.0,
                    dense_score: None,
                    sparse_score: None,
                    kind: h.kind.clone(),
                    name: h.name.clone(),
                });
            c.sparse_score = Some(h.score);
            // BM25 brought the full content too — upgrade the preview if it
            // came in only with a snippet from the dense side.
            if c.content.is_none() {
                c.content = Some(h.content.clone());
            }
            c.rrf += 1.0 / (rrf_k as f32 + rank as f32 + 1.0);
        }

        // Optional path filter (substring on file path). Done post-merge so
        // it applies uniformly to both modalities; the pool was widened
        // above to absorb the loss.
        if let Some(prefix) = params.path {
            by_id.retain(|_, c| c.file.contains(prefix));
        }
        // Both stores were told the language, so this only ever fires on a
        // payload/schema disagreement — cheap insurance that a `lang`-scoped
        // query can never answer with another language.
        if let Some(lang) = params.lang {
            by_id.retain(|_, c| c.lang == lang);
        }

        let mut merged: Vec<Candidate> = by_id.into_values().collect();
        if merged.is_empty() {
            return Ok(Vec::new());
        }

        // Exact-symbol boost: when the query literally names a chunk's symbol
        // (`AdaptiveBatcher::note_failure`, `build_si_portfolio`, `Indexer`),
        // that chunk gets one extra #1 rank-vote. This is what lets
        // `code_search` win the "I know the symbol, find it" queries that
        // would otherwise justify falling back to grep. Applied to the RRF
        // before the head split so a symbol match also earns a rerank slot.
        let symbol_boost = config.search.symbol_boost.unwrap_or(DEFAULT_SYMBOL_BOOST);
        if symbol_boost > 0.0 {
            let bonus = symbol_boost / (rrf_k as f32 + 1.0);
            for c in merged.iter_mut() {
                if let Some(name) = &c.name {
                    if query_names_symbol(params.query, name) {
                        debug!(file = %c.file, name = %name, "exact symbol match — boosting");
                        c.rrf += bonus;
                    }
                }
            }
        }

        merged.sort_by(|a, b| {
            b.rrf
                .partial_cmp(&a.rrf)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        debug!(
            merged = merged.len(),
            elapsed_ms = started.elapsed().as_millis() as u64,
            "merged"
        );

        // 4. Two-stage rerank:
        //    a) Split merged into head (top rerank_top_n by RRF) and tail (rest).
        //    b) Upgrade head's previews to full chunk content (one batched
        //       bm25 lookup for dense-only candidates; BM25-side hits already
        //       have full text).
        //    c) Cross-encoder scores the head; its rank-vote is fused into the
        //       head's RRF scores. Tail keeps plain RRF score — consistent,
        //       since every head candidate's fused score ≥ its RRF score ≥
        //       any tail RRF score.
        //    d) Final list = fused_head + tail (in that order), trimmed to
        //       params.limit. For default config (limit=10, top_n=30), all
        //       returned results are reranked.
        //
        //    If the reranker call fails (server down, ctx overflow, timeout),
        //    we fall back to RRF-sorted results rather than failing the whole
        //    search. The user still gets something useful; warn logged.
        let want_rerank = params.use_rerank && self.reranker.is_some();

        let mut head = merged;
        let tail: Vec<Candidate> = if want_rerank && head.len() > rerank_top_n {
            head.split_off(rerank_top_n)
        } else {
            Vec::new()
        };

        let head_results: Vec<SearchResult> = if want_rerank {
            params.report(SearchStage::Reranking);
            hydrate_content(&bm25, &mut head);
            let documents: Vec<String> = head
                .iter()
                .map(|c| c.content.clone().unwrap_or_else(|| c.preview.clone()))
                .collect();
            let rer = self
                .reranker
                .as_ref()
                .expect("want_rerank implies a reranker");
            match rer.rerank(params.query, documents).await {
                Ok(scores) => {
                    debug!(
                        reranked = scores.len(),
                        elapsed_ms = started.elapsed().as_millis() as u64,
                        "reranked head"
                    );
                    let weight = config.search.rerank_weight.unwrap_or(DEFAULT_RERANK_WEIGHT);
                    let rrf_scores: Vec<f32> = head.iter().map(|c| c.rrf).collect();
                    let fused = fuse_rerank_votes(&rrf_scores, &scores, rrf_k, weight);
                    let mut with_scores: Vec<SearchResult> = head
                        .into_iter()
                        .zip(scores)
                        .zip(fused)
                        .map(|((c, rer_score), final_score)| {
                            candidate_to_result(c, final_score, Some(rer_score))
                        })
                        .collect();
                    with_scores.sort_by(|a, b| {
                        b.score
                            .partial_cmp(&a.score)
                            .unwrap_or(std::cmp::Ordering::Equal)
                    });
                    with_scores
                }
                Err(e) => {
                    // Reranker is a quality boost, not a correctness requirement.
                    // Falling back to RRF lets the user keep using the system
                    // while they fix the rerank server (timeout, ctx overflow,
                    // model crash).
                    warn!(
                        error = %e,
                        "reranker failed; falling back to RRF-only ranking"
                    );
                    head.into_iter()
                        .map(|c| {
                            let rrf = c.rrf;
                            candidate_to_result(c, rrf, None)
                        })
                        .collect()
                }
            }
        } else {
            // No rerank requested (or no reranker configured): RRF score is final.
            head.into_iter()
                .map(|c| {
                    let rrf = c.rrf;
                    candidate_to_result(c, rrf, None)
                })
                .collect()
        };

        // Tail (everything past rerank_top_n) keeps RRF score. Only relevant
        // when params.limit > rerank_top_n — otherwise the trim below drops it.
        let tail_results: Vec<SearchResult> = tail
            .into_iter()
            .map(|c| {
                let rrf = c.rrf;
                candidate_to_result(c, rrf, None)
            })
            .collect();

        params.report(SearchStage::Finalizing);
        let mut final_results = head_results;
        final_results.extend(tail_results);
        Ok(final_results.into_iter().take(params.limit).collect())
    }
}

/// One-shot search: build a context, run a single query, drop it. The CLI
/// path — `serve` holds a [`SearchContext`] instead.
pub async fn run(config: &Config, params: SearchParams<'_>) -> Result<Vec<SearchResult>> {
    let ctx = SearchContext::new(config).await?;
    ctx.search(params).await
}

/// Give every rerank candidate its full chunk text. Dense-only candidates
/// arrive with just the 200-char Qdrant snippet; one batched tantivy
/// lookup fills them in. Failure is non-fatal — the cross-encoder then
/// judges a snippet, which is worse but still ranked.
fn hydrate_content(bm25: &Bm25Search, head: &mut [Candidate]) {
    let missing: Vec<String> = head
        .iter()
        .filter(|c| c.content.is_none())
        .map(|c| c.chunk_id.clone())
        .collect();
    if missing.is_empty() {
        return;
    }
    match bm25.lookup_contents(&missing) {
        Ok(map) => {
            for c in head.iter_mut().filter(|c| c.content.is_none()) {
                match map.get(&c.chunk_id) {
                    Some(text) => c.content = Some(text.clone()),
                    None => warn!(
                        chunk_id = %c.chunk_id,
                        file = %c.file,
                        "no BM25 doc for chunk_id; reranker will see snippet only"
                    ),
                }
            }
        }
        Err(e) => warn!(
            missing = missing.len(),
            error = %e,
            "batch content lookup failed; reranker will see snippets only"
        ),
    }
}

struct Candidate {
    chunk_id: String,
    file: String,
    start_line: u64,
    end_line: u64,
    lang: String,
    preview: String,
    content: Option<String>,
    rrf: f32,
    dense_score: Option<f32>,
    sparse_score: Option<f32>,
    kind: Option<String>,
    name: Option<String>,
}

/// Fuse the cross-encoder's opinion into the retrieval RRF by *rank*,
/// not by raw score — reranker logits (≈ −10..+10) and RRF scores
/// (≈ 0.01..0.05) live on incomparable scales, the same argument that
/// picked RRF over score-weighting for dense+sparse in the first place.
///
/// `fused[i] = rrf[i] + weight / (rrf_k + rerank_rank_of_i + 1)` — the
/// reranker contributes one more rank-vote with `weight` times the pull
/// of a single retrieval modality. A candidate both retrieval sides
/// agree on survives one bad cross-encoder call; ties among
/// retrieval-equals are still broken by the reranker.
fn fuse_rerank_votes(rrf: &[f32], rerank: &[f32], rrf_k: usize, weight: f32) -> Vec<f32> {
    debug_assert_eq!(rrf.len(), rerank.len());
    let mut order: Vec<usize> = (0..rerank.len()).collect();
    order.sort_by(|&a, &b| {
        rerank[b]
            .partial_cmp(&rerank[a])
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    let mut fused = rrf.to_vec();
    for (rank, &idx) in order.iter().enumerate() {
        fused[idx] += weight / (rrf_k as f32 + rank as f32 + 1.0);
    }
    fused
}

/// Does the query literally name this symbol?
///
/// Query tokens are runs of identifier-ish chars (`[A-Za-z0-9_:.]`). A
/// token matches when it equals the full (possibly qualified) name, or
/// the name's tail segment (after the last `::` / `.`). To avoid boosting
/// every chunk named `run` when the query merely contains the English
/// word "run", a match requires the token to *look like an identifier*:
/// contain `_`, `::`, `.`, an interior uppercase letter (camelCase), or —
/// for plain names — equal the name case-sensitively while the name is
/// capitalized (PascalCase types referenced verbatim, e.g. "Indexer").
/// Consequence: single-word lowercase symbols (`run`, `new`, `main`) are
/// only boosted when qualified in the query (`Watcher::run`).
fn query_names_symbol(query: &str, name: &str) -> bool {
    fn is_identifier_like(t: &str) -> bool {
        t.contains('_')
            || t.contains("::")
            || t.contains('.')
            || t.chars().skip(1).any(|c| c.is_uppercase())
    }
    let name_is_qualified = name.contains("::") || name.contains('.');
    let name_tail = name
        .rsplit("::")
        .next()
        .and_then(|s| s.rsplit('.').next())
        .unwrap_or(name);
    let name_lc = name.to_lowercase();
    let tail_lc = name_tail.to_lowercase();

    for raw in query.split(|ch: char| !(ch.is_alphanumeric() || "_:.".contains(ch))) {
        let t = raw.trim_matches(|ch: char| ch == ':' || ch == '.');
        if t.is_empty() {
            continue;
        }
        let t_lc = t.to_lowercase();
        // Full name match: qualified names are unambiguous on their own;
        // plain names need an identifier-like token or a verbatim
        // case-sensitive match on a capitalized name.
        if t_lc == name_lc
            && (name_is_qualified
                || is_identifier_like(t)
                || (t == name && name.starts_with(char::is_uppercase)))
        {
            return true;
        }
        // Tail-segment match (`note_failure` finding
        // `AdaptiveBatcher::note_failure`) — identifier-like tokens only.
        if name_is_qualified && t_lc == tail_lc && is_identifier_like(t) {
            return true;
        }
    }
    false
}

/// Move a Candidate into a SearchResult with the given `final_score` and
/// `rerank_score`. Centralizes the field copy so kind/name don't drift
/// between the four ranking paths (reranked head, RRF fallback head,
/// no-rerank head, tail).
fn candidate_to_result(c: Candidate, final_score: f32, rerank_score: Option<f32>) -> SearchResult {
    SearchResult {
        file: c.file,
        start_line: c.start_line,
        end_line: c.end_line,
        lang: c.lang,
        score: final_score,
        dense_score: c.dense_score,
        sparse_score: c.sparse_score,
        rerank_score,
        preview: c.content.unwrap_or(c.preview),
        kind: c.kind,
        name: c.name,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Rank order of the fused scores, best first.
    fn ranking(fused: &[f32]) -> Vec<usize> {
        let mut order: Vec<usize> = (0..fused.len()).collect();
        order.sort_by(|&a, &b| fused[b].partial_cmp(&fused[a]).unwrap());
        order
    }

    #[test]
    fn fuse_empty_input() {
        assert!(fuse_rerank_votes(&[], &[], 60, 2.0).is_empty());
    }

    #[test]
    fn fuse_retrieval_consensus_survives_reranker_miss() {
        // Doc 0: top of both retrieval modalities (rrf = 1/61 + 1/61),
        // but the reranker puts it dead last of three.
        // Doc 1: weak retrieval (rank ~25 in one modality), reranker's #1.
        // Doc 2: middling everywhere.
        let rrf = [2.0 / 61.0, 1.0 / 85.0, 1.0 / 70.0];
        let rerank = [-3.2, 0.8, -1.4];
        let fused = fuse_rerank_votes(&rrf, &rerank, 60, 2.0);
        // Retrieval consensus outweighs the reranker's lone top vote;
        // among the two weak-retrieval docs the higher RRF (doc 2) edges
        // out the reranker's favorite because their rank bonuses are
        // adjacent (2/62 vs 2/61) while their RRF gap is bigger.
        assert_eq!(ranking(&fused), vec![0, 2, 1]);
    }

    #[test]
    fn fuse_reranker_breaks_retrieval_ties() {
        // Equal RRF — the reranker's vote is the only differentiator.
        let rrf = [1.0 / 61.0; 3];
        let rerank = [-5.0, 7.0, 1.0];
        let fused = fuse_rerank_votes(&rrf, &rerank, 60, 2.0);
        assert_eq!(ranking(&fused), vec![1, 2, 0]);
    }

    #[test]
    fn fuse_zero_weight_is_pure_rrf() {
        let rrf = [0.03, 0.01, 0.02];
        let rerank = [-9.0, 9.0, 0.0];
        let fused = fuse_rerank_votes(&rrf, &rerank, 60, 0.0);
        assert_eq!(fused.to_vec(), rrf.to_vec());
    }

    #[test]
    fn fuse_scores_never_below_rrf() {
        // Fused = RRF + non-negative bonus, so every head candidate keeps
        // at least its RRF score — this is what keeps head-above-tail
        // ordering consistent in run().
        let rrf = [0.033, 0.012, 0.016, 0.020];
        let rerank = [0.78, 0.30, -1.41, -3.34];
        let fused = fuse_rerank_votes(&rrf, &rerank, 60, 2.0);
        for (f, r) in fused.iter().zip(rrf.iter()) {
            assert!(f >= r);
        }
    }

    #[test]
    fn symbol_match_qualified_name_in_query() {
        assert!(query_names_symbol(
            "AdaptiveBatcher::note_failure timeout handling",
            "AdaptiveBatcher::note_failure"
        ));
        assert!(query_names_symbol(
            "where is Watcher::run started",
            "Watcher::run"
        ));
        // Dot-qualified (Python / TS / Go style).
        assert!(query_names_symbol(
            "Portfolio.buildSiPortfolio helper",
            "Portfolio.buildSiPortfolio"
        ));
    }

    #[test]
    fn symbol_match_tail_segment() {
        assert!(query_names_symbol(
            "note_failure logic",
            "AdaptiveBatcher::note_failure"
        ));
        assert!(query_names_symbol(
            "buildSiPortfolio deposit sizing",
            "Portfolio.buildSiPortfolio"
        ));
        // Plain lowercase tail ("run") is ambiguous English — no boost.
        assert!(!query_names_symbol("how does run work", "Watcher::run"));
    }

    #[test]
    fn symbol_match_plain_names() {
        // snake_case identifiers are unambiguous.
        assert!(query_names_symbol(
            "where is build_si_portfolio defined",
            "build_si_portfolio"
        ));
        // PascalCase type referenced verbatim.
        assert!(query_names_symbol("the Indexer struct fields", "Indexer"));
        // Lowercase English word vs lowercase fn name — too ambiguous.
        assert!(!query_names_symbol("running the main loop", "run"));
        assert!(!query_names_symbol("a new approach", "new"));
        // Sentence-capitalized English word vs lowercase fn name.
        assert!(!query_names_symbol("Run the loop", "run"));
    }

    #[test]
    fn symbol_match_qualified_query_does_not_hit_bare_tail() {
        // Query names `Watcher::run`; a free function `run` must NOT match.
        assert!(!query_names_symbol("Watcher::run shutdown", "run"));
    }

    #[test]
    fn fuse_handles_tied_rerank_scores() {
        // Ties in rerank scores must not panic and must keep RRF ordering
        // influence intact.
        let rrf = [0.02, 0.03];
        let rerank = [1.0, 1.0];
        let fused = fuse_rerank_votes(&rrf, &rerank, 60, 2.0);
        assert_eq!(ranking(&fused), vec![1, 0]);
    }

    #[test]
    fn stage_steps_are_ordered_and_within_total() {
        let stages = [
            SearchStage::Embedding,
            SearchStage::Retrieving,
            SearchStage::Reranking,
            SearchStage::Finalizing,
        ];
        for (i, s) in stages.iter().enumerate() {
            assert_eq!(s.step(), i as u32 + 1);
            assert!(s.step() <= SearchStage::TOTAL);
            assert!(!s.label().is_empty());
        }
    }
}
