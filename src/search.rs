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

use anyhow::{Context, Result};
use std::collections::HashMap;
use tracing::{debug, warn};

use crate::bm25::Bm25Search;
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

pub struct SearchParams<'a> {
    pub query: &'a str,
    pub limit: usize,
    pub use_rerank: bool,
    pub lang: Option<&'a str>,
    pub path: Option<&'a str>,
}

pub async fn run(config: &Config, params: SearchParams<'_>) -> Result<Vec<SearchResult>> {
    let started = std::time::Instant::now();

    let k_dense = config.search.dense_k.unwrap_or(DEFAULT_K_DENSE);
    let k_sparse = config.search.sparse_k.unwrap_or(DEFAULT_K_SPARSE);
    let rerank_top_n = config.search.rerank_top_n.unwrap_or(DEFAULT_RERANK_TOP_N);
    let rrf_k = config.search.rrf_k.unwrap_or(DEFAULT_RRF_K);

    // 1. Embed the query. Same model as indexing — no asymmetric query/doc
    //    encoders here; jina-code-embeddings is symmetric.
    let embedder = embedding::Client::new(&config.embedding);
    let mut vecs = embedder
        .embed(vec![params.query.to_string()])
        .await
        .context("embedding query")?;
    let query_vec = vecs.pop().context("empty embedding response for query")?;
    debug!(
        elapsed_ms = started.elapsed().as_millis() as u64,
        "query embedded"
    );

    // 2. Dense + sparse in parallel.
    let vs = vector_store::Client::new(
        &config.vector_store,
        config.vector_store.resolve_collection_name(&config.project),
    )
    .await?;
    // Read-side marker check — bails if this collection's stored project
    // identity doesn't match the current config's, so search NEVER returns
    // chunks from a different project that happens to share the collection.
    vs.verify_marker_read_only(config).await?;
    let bm25 = Bm25Search::open(&config.bm25.index_path)?;

    let (dense_res, sparse_res) =
        tokio::join!(vs.search(&query_vec, k_dense, params.lang), async {
            bm25.search(params.query, k_sparse)
        });
    let dense_hits = dense_res.context("dense search")?;
    let sparse_hits = sparse_res.context("bm25 search")?;
    debug!(
        dense = dense_hits.len(),
        sparse = sparse_hits.len(),
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
                // BM25 doesn't carry kind/name (tantivy schema doesn't store
                // them). Stays None unless this same chunk_id also hit on
                // the dense side, where the insert above filled them in.
                kind: None,
                name: None,
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
    // it applies uniformly to both modalities.
    if let Some(prefix) = params.path {
        by_id.retain(|_, c| c.file.contains(prefix));
    }

    let mut merged: Vec<Candidate> = by_id.into_values().collect();
    merged.sort_by(|a, b| {
        b.rrf
            .partial_cmp(&a.rrf)
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    if merged.is_empty() {
        return Ok(Vec::new());
    }

    // Fill kind/name for sparse-only candidates. BM25's tantivy schema
    // doesn't carry the AST metadata (would require an incompatible
    // schema migration), so any chunk that came in via BM25 alone has
    // kind=None/name=None at this point. One Qdrant retrieve-by-IDs
    // call patches them up — the chunk DEFINITELY has these stored
    // (we indexed both stores together), it's just not in this
    // candidate's local copy.
    let missing_meta: Vec<String> = merged
        .iter()
        .filter(|c| c.kind.is_none() && c.name.is_none())
        .map(|c| c.chunk_id.clone())
        .collect();
    if !missing_meta.is_empty() {
        match vs.fetch_kind_name(&missing_meta).await {
            Ok(map) => {
                for c in merged.iter_mut() {
                    if let Some((k, n)) = map.get(&c.chunk_id) {
                        if c.kind.is_none() {
                            c.kind = k.clone();
                        }
                        if c.name.is_none() {
                            c.name = n.clone();
                        }
                    }
                }
            }
            Err(e) => {
                // Non-fatal — the search still works, results just
                // show without syntactic anchors for sparse-only hits.
                warn!(error = %e, "failed to fetch kind/name for sparse-only candidates");
            }
        }
    }

    debug!(
        merged = merged.len(),
        elapsed_ms = started.elapsed().as_millis() as u64,
        "merged"
    );

    // 4. Two-stage rerank:
    //    a) Split merged into head (top rerank_top_n by RRF) and tail (rest).
    //    b) Upgrade head's previews to full chunk content (bm25 lookup for
    //       dense-only candidates; BM25-side hits already have full text).
    //    c) Cross-encoder scores the head; its rank-vote is fused into the
    //       head's RRF scores. Tail keeps plain RRF score — consistent,
    //       since every head candidate's fused score ≥ its RRF score ≥
    //       any tail RRF score.
    //    d) Final list = fused_head + tail (in that order), trimmed to
    //       params.limit. For default config (limit=10, top_n=20), all
    //       returned results are reranked.
    //
    //    If the reranker call fails (server down, ctx overflow, timeout),
    //    we fall back to RRF-sorted results rather than failing the whole
    //    search. The user still gets something useful; warn logged.
    let want_rerank = params.use_rerank && config.reranker.as_ref().is_some_and(|r| r.enabled);

    let mut head = merged;
    let tail: Vec<Candidate> = if want_rerank && head.len() > rerank_top_n {
        head.split_off(rerank_top_n)
    } else {
        Vec::new()
    };

    let head_results: Vec<SearchResult> = if want_rerank {
        for c in &mut head {
            if c.content.is_none() {
                match bm25.lookup_content(&c.chunk_id) {
                    Ok(Some(text)) => c.content = Some(text),
                    Ok(None) => {
                        // Dense hit with no BM25 doc for chunk_id —
                        // shouldn't happen since we index both stores
                        // together, but be defensive.
                        warn!(
                            chunk_id = %c.chunk_id,
                            file = %c.file,
                            "no BM25 doc for chunk_id; reranker will see snippet only"
                        );
                    }
                    Err(e) => {
                        warn!(
                            chunk_id = %c.chunk_id,
                            error = %e,
                            "lookup_content failed; reranker will see snippet only"
                        );
                    }
                }
            }
        }
        let documents: Vec<String> = head
            .iter()
            .map(|c| c.content.clone().unwrap_or_else(|| c.preview.clone()))
            .collect();
        let rer = reranker::Client::new(config.reranker.as_ref().expect("reranker present"));
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

    let mut final_results = head_results;
    final_results.extend(tail_results);
    Ok(final_results.into_iter().take(params.limit).collect())
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
    fn fuse_handles_tied_rerank_scores() {
        // Ties in rerank scores must not panic and must keep RRF ordering
        // influence intact.
        let rrf = [0.02, 0.03];
        let rerank = [1.0, 1.0];
        let fused = fuse_rerank_votes(&rrf, &rerank, 60, 2.0);
        assert_eq!(ranking(&fused), vec![1, 0]);
    }
}
