use anyhow::{Context, Result};
use sha2::{Digest, Sha256};
use std::collections::{HashMap, HashSet};
use std::path::{Path, PathBuf};
use std::time::Duration;
use tracing::{debug, error, info, warn};
use uuid::Uuid;

use crate::adaptive_batcher::{AdaptiveBatcher, MIN_BUDGET};
use crate::bm25::{Bm25Index, ChunkDoc};
use crate::chunker::{Chunk, ChunkerSet};
use crate::config::Config;
use crate::embedding::{self, ErrorClass};
use crate::vector_store::{self, QdrantPoint};
use crate::walker;

/// Abort the run if this many files in a row fail with "server down" errors.
/// Beyond this, retry attempts are almost certainly wasted CPU and tokens —
/// the embedding server needs human attention (OOM, crash loop, network).
const MAX_CONSECUTIVE_SERVER_DOWN: u32 = 3;

/// How many recovery waits the adaptive embed loop will tolerate for a single
/// file before giving up. Beyond this, the file is failed and the caller can
/// decide what to do (typically: the indexer's consecutive_server_down kicks in).
const MAX_SERVER_RECOVERIES_PER_FILE: u32 = 2;

#[derive(Debug, Default)]
pub struct IndexStats {
    pub files_scanned: usize,
    pub files_indexed: usize,
    pub files_unchanged: usize,
    pub files_removed: usize,
    pub files_failed: usize,
    pub chunks_upserted: usize,
    pub chunks_skipped: usize,
    pub chunks_deleted: usize,
}

pub async fn run(config: &Config) -> Result<IndexStats> {
    let embedder = embedding::Client::new(&config.embedding);

    // Wait for embedding server to finish model load before doing anything
    // else. Without this, the indexer starts walking files immediately and
    // the first 1-3 minutes of files all fail with 503 "Loading model" or
    // connect timeout.
    if config.embedding.startup_wait_secs > 0 {
        info!(
            max_wait_s = config.embedding.startup_wait_secs,
            "waiting for embedding server to be ready"
        );
        embedder
            .wait_until_ready(Duration::from_secs(config.embedding.startup_wait_secs))
            .await?;
    }

    let vs = vector_store::Client::new(
        &config.vector_store,
        config.vector_store.resolve_collection_name(&config.project),
    )
    .await?;
    vs.ensure_collection(config.embedding.dimensions).await?;
    vs.ensure_payload_indexes().await;
    // Marker-driven config-change handling. Project identity mismatch
    // already bails inside check_marker_status. The remaining cases:
    //   Fresh        — first run, write the marker
    //   Match        — no change, proceed
    //   SoftChanged  — languages / exclude / gitignore changed; reindex
    //                  with stale-detection cleanup, no clear needed
    //   HardChanged  — chunking or embedding identity changed; chunk IDs
    //                  no longer align, so auto-clear the collection and
    //                  the tantivy directory, then continue
    match vs.check_marker_status(config).await? {
        vector_store::MarkerStatus::Fresh => {
            vs.write_current_marker(config, config.embedding.dimensions)
                .await?;
            info!(collection = %vs.collection_name(), "marker written (fresh collection)");
        }
        vector_store::MarkerStatus::Match => {
            debug!(collection = %vs.collection_name(), "marker matches");
        }
        vector_store::MarkerStatus::SoftChanged => {
            warn!(
                collection = %vs.collection_name(),
                "config-soft change detected (languages / exclude / gitignore). \
                 Reindexing with stale-detection — no `clear` needed."
            );
            vs.write_current_marker(config, config.embedding.dimensions)
                .await?;
        }
        vector_store::MarkerStatus::HardChanged => {
            warn!(
                collection = %vs.collection_name(),
                tantivy_path = %config.bm25.index_path.display(),
                "config-hard change detected (chunking strategy/params or embedding \
                 model/dimensions changed). Chunk IDs no longer align with stored data — \
                 AUTO-CLEARING the collection and tantivy index, then rebuilding from scratch."
            );
            vs.delete_collection().await?;
            if config.bm25.index_path.exists() {
                std::fs::remove_dir_all(&config.bm25.index_path).with_context(|| {
                    format!(
                        "auto-clear: removing tantivy dir {}",
                        config.bm25.index_path.display()
                    )
                })?;
            }
            // Recreate from scratch and rewrite marker for the new state.
            vs.ensure_collection(config.embedding.dimensions).await?;
            vs.ensure_payload_indexes().await;
            vs.write_current_marker(config, config.embedding.dimensions)
                .await?;
            info!("auto-clear complete; proceeding with fresh index");
        }
    }

    let mut bm25 = Bm25Index::open(&config.bm25.index_path)?;
    // Language-aware chunking: default strategy + per_language overrides.
    // Each FileEntry brings its language string; ChunkerSet dispatches.
    let chunker = ChunkerSet::from_config(&config.chunking)?;

    info!("loading index state from Qdrant");
    let mut cache: HashMap<PathBuf, String> = vs.scroll_files().await?;
    let qdrant_files = cache.len();

    // Local tantivy is the second store; on a fresh machine that shares a
    // Qdrant collection (e.g., same project tree mounted on a different
    // workstation), the Qdrant cache claims all files are indexed but the
    // local tantivy is empty — and a hybrid search needs both. Filter the
    // cache by what's actually in this machine's tantivy so a file present
    // in Qdrant but absent locally gets reprocessed (which refills tantivy
    // AND re-upserts to Qdrant idempotently).
    let tantivy_files = bm25.list_indexed_files()?;
    let before = cache.len();
    cache.retain(|path, _| tantivy_files.contains(path));
    let pruned = before - cache.len();
    if pruned > 0 {
        warn!(
            qdrant_files,
            tantivy_files = tantivy_files.len(),
            pruned,
            "tantivy is missing files that Qdrant has — they'll be reprocessed to refill tantivy locally"
        );
    }
    info!(
        already_indexed_files = cache.len(),
        "index state loaded (Qdrant ∩ tantivy)"
    );

    // Adaptive batcher — shared across all files in this run so it converges
    // to server capacity once and stays there. If config still has the
    // (now-advisory) max_input_chars set, use it as the starting hint;
    // AIMD will halve it down if it's too optimistic, grow it if conservative.
    let initial_budget = config.embedding.max_input_chars.unwrap_or(10_000);
    let mut batcher = AdaptiveBatcher::new(initial_budget);
    info!(
        initial_budget,
        min_budget = MIN_BUDGET,
        "adaptive batcher initialized"
    );

    let mut stats = IndexStats::default();
    let mut seen_paths: HashSet<PathBuf> = HashSet::new();

    let walk_result = process_walk(
        &embedder,
        &vs,
        &mut bm25,
        &chunker,
        config,
        &mut cache,
        &mut seen_paths,
        &mut stats,
        &mut batcher,
    )
    .await;

    // Best-effort trailing commit. Per-file work is already committed; this is the safety net.
    if let Err(e) = bm25.commit() {
        warn!(error = %e, "tantivy final commit failed");
    }

    walk_result?;

    // Stale detection — files we know about (cache) but didn't see during this walk.
    let stale: Vec<_> = cache
        .keys()
        .filter(|p| !seen_paths.contains(*p))
        .cloned()
        .collect();
    for path in stale {
        delete_file_from_indexes(&vs, &mut bm25, &mut cache, &path).await?;
        stats.files_removed += 1;
        debug!(file = %path.display(), "removed from index (stale)");
    }

    info!(
        final_budget = batcher.budget(),
        chars_per_sec = batcher.chars_per_sec().unwrap_or(0.0) as u64,
        "adaptive batcher final state"
    );

    Ok(stats)
}

#[allow(clippy::too_many_arguments)]
async fn process_walk(
    embedder: &embedding::Client,
    vs: &vector_store::Client,
    bm25: &mut Bm25Index,
    chunker: &ChunkerSet,
    config: &Config,
    cache: &mut HashMap<PathBuf, String>,
    seen_paths: &mut HashSet<PathBuf>,
    stats: &mut IndexStats,
    batcher: &mut AdaptiveBatcher,
) -> Result<()> {
    let mut consecutive_server_down: u32 = 0;

    for entry in walker::walk(&config.project, &config.index)? {
        let entry = entry?;
        stats.files_scanned += 1;
        seen_paths.insert(entry.relative.clone());

        // Per-file error isolation: a 500 on one file doesn't kill the whole run.
        let result: Result<()> =
            process_one_file(embedder, vs, bm25, chunker, cache, stats, &entry, batcher).await;

        match &result {
            Ok(_) => {
                consecutive_server_down = 0;
            }
            Err(e) => {
                if embedding::is_server_down(e) {
                    consecutive_server_down += 1;
                    if consecutive_server_down == 1 {
                        error!(
                            file = %entry.relative.display(),
                            error = ?e,
                            "EMBEDDING SERVER APPEARS UNAVAILABLE. Recovery wait exhausted. \
                             Investigate immediately to avoid wasting the rest of the run:\n  \
                             1. `docker ps -a | grep embedding`  (Restarting? Exited?)\n  \
                             2. `docker logs embedding --tail 100`  (OOM? segfault? model loading error?)\n  \
                             3. Common fixes: lower --ubatch-size / -c on docker run, \
                                 or add memory headroom (--memory or smaller model).\n  \
                             Indexer will abort after {} consecutive server-down failures.",
                            MAX_CONSECUTIVE_SERVER_DOWN
                        );
                    }
                    if consecutive_server_down >= MAX_CONSECUTIVE_SERVER_DOWN {
                        error!(
                            consecutive = consecutive_server_down,
                            indexed_this_run = stats.files_indexed,
                            failed_this_run = stats.files_failed + 1,
                            "ABORTING: embedding server unavailable for {} consecutive files. \
                             Fix the server and re-run; progress so far is persisted in Qdrant \
                             and will be picked up via cache on next start.",
                            consecutive_server_down
                        );
                        stats.files_failed += 1;
                        anyhow::bail!(
                            "embedding server chronically unavailable ({} consecutive failures) — see logs above",
                            consecutive_server_down
                        );
                    }
                } else {
                    // Per-file issue (oversized chunk, bad input, etc.) — not systemic.
                    consecutive_server_down = 0;
                }
                warn!(
                    file = %entry.relative.display(),
                    error = ?e,
                    "failed to index file; skipping (will be retried on next run)"
                );
                stats.files_failed += 1;
            }
        }

        let done = stats.files_indexed + stats.files_unchanged + stats.files_failed;
        if done > 0 && done.is_multiple_of(25) && stats.files_indexed > 0 {
            info!(
                indexed = stats.files_indexed,
                unchanged = stats.files_unchanged,
                failed = stats.files_failed,
                chunks_skipped = stats.chunks_skipped,
                budget = batcher.budget(),
                chars_per_sec = batcher.chars_per_sec().unwrap_or(0.0) as u64,
                "progress"
            );
        }
    }
    Ok(())
}

/// Remove all chunks for `relative_path` from both indexes and the cache.
/// Used by the indexer's stale-detection pass and by the watcher when it
/// observes a file deletion. `bm25.commit()` is called so the deletion is
/// durable as soon as this returns.
pub async fn delete_file_from_indexes(
    vs: &vector_store::Client,
    bm25: &mut Bm25Index,
    cache: &mut HashMap<PathBuf, String>,
    relative_path: &Path,
) -> Result<()> {
    let file_str = relative_path.to_string_lossy().to_string();
    vs.delete_by_file(&file_str).await?;
    bm25.delete_by_file(&file_str);
    bm25.commit()?;
    cache.remove(relative_path);
    Ok(())
}

/// Read a file as UTF-8 text, returning `Ok(None)` when it looks binary — a
/// NUL byte anywhere, or bytes that aren't valid UTF-8. Binary content is an
/// expected, non-fatal skip under the all-but-binary walk default, so we
/// distinguish it from a genuine I/O error (`Err`).
fn read_text_file(path: &Path) -> std::io::Result<Option<String>> {
    let bytes = std::fs::read(path)?;
    if bytes.contains(&0) {
        return Ok(None);
    }
    Ok(String::from_utf8(bytes).ok())
}

#[allow(clippy::too_many_arguments)]
pub async fn process_one_file(
    embedder: &embedding::Client,
    vs: &vector_store::Client,
    bm25: &mut Bm25Index,
    chunker: &ChunkerSet,
    cache: &mut HashMap<PathBuf, String>,
    stats: &mut IndexStats,
    entry: &walker::FileEntry,
    batcher: &mut AdaptiveBatcher,
) -> Result<()> {
    let content = match read_text_file(&entry.absolute) {
        Ok(Some(c)) => c,
        Ok(None) => {
            // Binary content (NUL byte or invalid UTF-8). Expected, not an
            // error: the all-but-binary walk default can admit an unlisted
            // binary type (extensionless executable, exotic extension).
            debug!(file = %entry.relative.display(), "skip binary file");
            return Ok(());
        }
        Err(e) => {
            warn!(file = %entry.relative.display(), error = %e, "skip unreadable file");
            return Ok(());
        }
    };
    let file_sha = sha256_hex(&content);
    let rel_str = entry.relative.to_string_lossy().to_string();

    if let Some(cached_sha) = cache.get(&entry.relative) {
        if cached_sha == &file_sha {
            stats.files_unchanged += 1;
            return Ok(());
        }
    }

    // The file is in the cache with a different sha: it's an edit, not a
    // first index, so its old vectors are still in Qdrant and most of its
    // chunks probably survived the edit unchanged. Fetch them before the
    // delete below wipes them. A failure here only costs re-embedding.
    let reusable: HashMap<String, Vec<f32>> = if cache.contains_key(&entry.relative) {
        match vs.fetch_file_vectors(&rel_str).await {
            Ok(map) => map,
            Err(e) => {
                warn!(
                    file = %entry.relative.display(),
                    error = %e,
                    "could not fetch previous vectors; re-embedding the whole file"
                );
                HashMap::new()
            }
        }
    } else {
        HashMap::new()
    };

    // Wipe any prior footprint of this file in both stores before reindexing.
    // Idempotent if the file has nothing yet (new/legacy).
    vs.delete_by_file(&rel_str).await?;
    bm25.delete_by_file(&rel_str);

    let chunks = chunker.chunk(&content, &entry.absolute, &entry.language);
    if chunks.is_empty() {
        return Ok(());
    }

    let chunk_shas: Vec<String> = chunks.iter().map(|c| sha256_hex(&c.text)).collect();
    let mut vectors_opt: Vec<Option<Vec<f32>>> = vec![None; chunks.len()];
    let mut embed_idx: Vec<usize> = Vec::new();
    let mut texts: Vec<String> = Vec::new();
    for (i, sha) in chunk_shas.iter().enumerate() {
        match reusable.get(sha) {
            Some(vector) => vectors_opt[i] = Some(vector.clone()),
            None => {
                embed_idx.push(i);
                texts.push(chunks[i].text.clone());
            }
        }
    }
    let reused = chunks.len() - embed_idx.len();

    let embed_start = std::time::Instant::now();
    if !texts.is_empty() {
        let total_chars: usize = texts.iter().map(|t| t.len()).sum();
        // Logged BEFORE the embed call so the user can see which file is "stuck"
        // on the GPU. Without this, a single 100+ KB file with many chunks looks
        // like a multi-minute hang at INFO level (tantivy logs cover the *previous*
        // file, not the one currently embedding).
        info!(
            file = %entry.relative.display(),
            chunks = chunks.len(),
            embedding = texts.len(),
            reused,
            chars = total_chars,
            budget = batcher.budget(),
            "embedding file"
        );
        let embedded = embed_with_adaptive_batching(embedder, texts, batcher)
            .await
            .with_context(|| format!("embed for {}", entry.relative.display()))?;
        if embedded.len() != embed_idx.len() {
            anyhow::bail!(
                "adaptive batcher returned {} slots for {} chunks",
                embedded.len(),
                embed_idx.len()
            );
        }
        for (slot, vector) in embed_idx.iter().zip(embedded) {
            vectors_opt[*slot] = vector;
        }
    } else {
        debug!(
            file = %entry.relative.display(),
            chunks = chunks.len(),
            "all chunks unchanged; reusing stored vectors"
        );
    }
    let embed_ms = embed_start.elapsed().as_millis() as u64;

    let mut points = Vec::new();
    let mut skipped_in_file = 0usize;
    for ((chunk, chunk_sha), vector_opt) in chunks.iter().zip(&chunk_shas).zip(vectors_opt) {
        let Some(vector) = vector_opt else {
            skipped_in_file += 1;
            continue;
        };
        let id = chunk_uuid(&entry.relative, chunk, chunk_sha);
        let snippet: String = chunk.text.chars().take(200).collect();
        // kind/name are populated by AST-aware chunkers (currently TreeSitter
        // for Rust). For other chunkers they're null in the payload.
        let payload = serde_json::json!({
            "file": rel_str,
            "file_sha256": file_sha,
            "chunk_sha": chunk_sha,
            "start_line": chunk.start_line,
            "end_line": chunk.end_line,
            "lang": entry.language,
            "snippet": snippet,
            "kind": chunk.kind,
            "name": chunk.name,
        });
        points.push(QdrantPoint {
            id: id.clone(),
            vector,
            payload,
        });
        bm25.upsert(&ChunkDoc {
            file: &rel_str,
            chunk_id: &id,
            start_line: chunk.start_line as u64,
            end_line: chunk.end_line as u64,
            lang: &entry.language,
            kind: chunk.kind.as_deref(),
            name: chunk.name.as_deref(),
            content: &chunk.text,
        })?;
    }

    if points.is_empty() {
        // All chunks were unembeddable. Cache the sha anyway so we don't
        // retry the same hopeless file every run — the warn is enough signal
        // for the user to decide whether to fix or exclude.
        warn!(
            file = %entry.relative.display(),
            total_chunks = chunks.len(),
            "all chunks unembeddable — file recorded as indexed-with-no-content"
        );
        cache.insert(entry.relative.clone(), file_sha);
        stats.chunks_skipped += skipped_in_file;
        return Ok(());
    }

    // tantivy commit FIRST, then Qdrant upsert (source of truth written last).
    // See the rationale in the longer comment in the old indexer: on crash
    // between these we get cache miss + reprocess, not silent inconsistency.
    bm25.commit()?;
    vs.upsert_points(points)
        .await
        .with_context(|| format!("qdrant upsert for {}", entry.relative.display()))?;
    cache.insert(entry.relative.clone(), file_sha);

    stats.chunks_upserted += chunks.len() - skipped_in_file;
    stats.chunks_skipped += skipped_in_file;
    stats.files_indexed += 1;

    if skipped_in_file > 0 {
        warn!(
            file = %entry.relative.display(),
            indexed_chunks = chunks.len() - skipped_in_file,
            skipped_chunks = skipped_in_file,
            embed_ms,
            "indexed file (partial — some chunks were unembeddable)"
        );
    } else {
        info!(
            file = %entry.relative.display(),
            chunks = chunks.len(),
            reused,
            embed_ms,
            "indexed file"
        );
    }
    Ok(())
}

/// Self-tuning embed loop. For each batch:
///   1. Pack from the adaptive batcher's current budget.
///   2. Send with a timeout scaled by batch size.
///   3. Classify failures:
///      - ServerDown    → wait for recovery, retry same batch
///      - WorkloadTooBig → halve budget, retry first half (or skip single chunk)
///      - PermanentBad  → skip chunk(s), don't retry
///      - Ambiguous     → quick probe to disambiguate
///
/// Returns `Vec<Option<Vec<f32>>>` of the same length as input — `None`
/// where a chunk was permanently skipped.
async fn embed_with_adaptive_batching(
    client: &embedding::Client,
    texts: Vec<String>,
    batcher: &mut AdaptiveBatcher,
) -> Result<Vec<Option<Vec<f32>>>> {
    let mut results: Vec<Option<Vec<f32>>> = (0..texts.len()).map(|_| None).collect();
    let mut start = 0usize;
    let mut server_recoveries = 0u32;

    while start < texts.len() {
        let end = batcher.pack(&texts, start);
        debug_assert!(end > start, "pack must advance");
        let batch_chars: usize = texts[start..end].iter().map(|t| t.len()).sum();
        // Throughput-aware timeout: derived from the batcher's observed
        // chars/sec EWMA. No static `timeout_secs` to guess.
        let timeout = batcher.estimate_timeout(batch_chars);

        let send_started = std::time::Instant::now();
        let batch = texts[start..end].to_vec();

        match client.embed_with_timeout(batch, timeout).await {
            Ok(vectors) => {
                if vectors.len() != end - start {
                    anyhow::bail!(
                        "embedding returned {} vectors for {} inputs",
                        vectors.len(),
                        end - start
                    );
                }
                let elapsed = send_started.elapsed();
                batcher.note_success(batch_chars, elapsed);
                for (i, v) in vectors.into_iter().enumerate() {
                    results[start + i] = Some(v);
                }
                debug!(
                    batch_size = end - start,
                    batch_chars,
                    elapsed_ms = elapsed.as_millis() as u64,
                    timeout_s = timeout.as_secs(),
                    budget = batcher.budget(),
                    chars_per_sec = batcher.chars_per_sec().unwrap_or(0.0) as u64,
                    "batch ok"
                );
                start = end;
                server_recoveries = 0;
            }
            Err(e) => {
                let class = match embedding::classify(&e) {
                    ErrorClass::Ambiguous => {
                        // Disambiguate: server alive → workload problem;
                        // server unreachable → server problem.
                        if client.is_alive_quick().await {
                            ErrorClass::WorkloadTooBig
                        } else {
                            ErrorClass::ServerDown
                        }
                    }
                    c => c,
                };

                match class {
                    ErrorClass::ServerDown => {
                        if server_recoveries >= MAX_SERVER_RECOVERIES_PER_FILE {
                            return Err(
                                e.context("embedding server unavailable after recovery attempts")
                            );
                        }
                        warn!(
                            error = %embedding::short_err(&e),
                            attempt = server_recoveries + 1,
                            max = MAX_SERVER_RECOVERIES_PER_FILE,
                            "embedding server unavailable, waiting for recovery"
                        );
                        client
                            .wait_until_ready(Duration::from_secs(300))
                            .await
                            .context("embedding server recovery wait")?;
                        server_recoveries += 1;
                    }
                    ErrorClass::WorkloadTooBig => {
                        if end - start > 1 {
                            // Multi-chunk: halve budget and cap the retry by
                            // the failing batch, so the next pack from the
                            // same start is strictly smaller.
                            let old = batcher.budget();
                            let new = batcher.note_failure(batch_chars);
                            warn!(
                                batch_size = end - start,
                                batch_chars,
                                old_budget = old,
                                new_budget = new,
                                retry_ceiling = ?batcher.retry_ceiling(),
                                error = %embedding::short_err(&e),
                                "batch too large for server; shrinking and retrying"
                            );
                        } else {
                            // Single chunk too big — halving doesn't help.
                            // Skip it permanently.
                            warn!(
                                chunk_index = start,
                                chunk_chars = texts[start].len(),
                                error = %embedding::short_err(&e),
                                "single chunk exceeds server capacity; skipping permanently"
                            );
                            results[start] = None;
                            start += 1;
                            // Don't halve the batcher — the issue is this chunk,
                            // not the budget. Other chunks may still pack normally.
                        }
                        server_recoveries = 0;
                    }
                    ErrorClass::PermanentBad => {
                        if end - start > 1 {
                            // Multi-chunk with a 4xx: somewhere in this batch
                            // is bad input. Shrink to bisect — the ceiling is
                            // what makes each pass actually narrow the search.
                            let new = batcher.note_failure(batch_chars);
                            warn!(
                                batch_size = end - start,
                                batch_chars,
                                new_budget = new,
                                retry_ceiling = ?batcher.retry_ceiling(),
                                error = %embedding::short_err(&e),
                                "batch rejected as bad input; bisecting"
                            );
                        } else {
                            warn!(
                                chunk_index = start,
                                chunk_chars = texts[start].len(),
                                error = %embedding::short_err(&e),
                                "chunk rejected as bad input; skipping"
                            );
                            results[start] = None;
                            start += 1;
                        }
                        server_recoveries = 0;
                    }
                    ErrorClass::Ambiguous => unreachable!("disambiguated above"),
                }
            }
        }
    }
    Ok(results)
}

fn sha256_hex(data: &str) -> String {
    let mut h = Sha256::new();
    h.update(data.as_bytes());
    hex::encode(h.finalize())
}

fn chunk_uuid(rel_path: &Path, chunk: &Chunk, chunk_sha: &str) -> String {
    let name = format!(
        "{}|{}-{}|{}",
        rel_path.display(),
        chunk.start_line,
        chunk.end_line,
        chunk_sha
    );
    Uuid::new_v5(&Uuid::NAMESPACE_OID, name.as_bytes()).to_string()
}
