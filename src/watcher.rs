//! File watcher: keeps the index in sync with filesystem changes.
//!
//! Flow:
//!   1. Initial sync via [`indexer::run`] — fast on subsequent starts
//!      because the sha256 cache means only changed files actually
//!      go through the embed pipeline.
//!   2. Construct long-running session state (embedder, vs, bm25, chunker,
//!      adaptive batcher, file→sha cache, path filter).
//!   3. Subscribe to a debounced notify watcher rooted at `project.root`.
//!   4. For each debounced batch of events: classify as
//!      `Modify`/`Create` (reprocess) or `Remove` (delete from indexes),
//!      apply the [`walker::PathFilter`] to drop noise (build artifacts,
//!      gitignored files), and dispatch to the indexer's per-file helpers.
//!   5. Ctrl-C performs a clean shutdown.
//!
//! State is shared across events: the adaptive batcher's throughput EWMA
//! persists, the cache stays current, and tantivy commits per file.

use anyhow::{Context, Result};
use notify::RecursiveMode;
use notify_debouncer_full::{new_debouncer, DebounceEventResult, DebouncedEvent};
use std::collections::{HashMap, HashSet};
use std::path::PathBuf;
use std::time::Duration;
use tokio::sync::mpsc;
use tracing::{debug, error, info, warn};

use crate::adaptive_batcher::AdaptiveBatcher;
use crate::bm25::Bm25Index;
use crate::chunker::ChunkerSet;
use crate::config::Config;
use crate::embedding;
use crate::indexer::{self, IndexStats};
use crate::vector_store;
use crate::walker::PathFilter;

pub async fn run(config: &Config) -> Result<()> {
    if !config.watcher.enabled {
        info!("watcher is disabled in config; nothing to do");
        return Ok(());
    }

    info!(
        root = %config.project.root.display(),
        debounce_ms = config.watcher.debounce_ms,
        "watcher: starting initial sync via indexer::run"
    );
    let initial: IndexStats = indexer::run(config).await?;
    info!(
        scanned = initial.files_scanned,
        indexed = initial.files_indexed,
        unchanged = initial.files_unchanged,
        failed = initial.files_failed,
        "watcher: initial sync done, switching to event mode"
    );

    // Long-running session — same pieces indexer::run uses, just held
    // across many events instead of a single walk.
    let embedder = embedding::Client::new(&config.embedding);
    let vs = vector_store::Client::new(
        &config.vector_store,
        config.vector_store.resolve_collection_name(&config.project),
    )
    .await?;
    vs.ensure_collection(config.embedding.dimensions).await?;
    // Marker was already validated (and updated, including any auto-clear)
    // by the initial indexer::run above. Read-only verify here is a cheap
    // sanity check.
    vs.verify_marker_read_only(config).await?;
    let mut bm25 = Bm25Index::open(&config.bm25.index_path)?;
    let chunker = ChunkerSet::from_config(&config.chunking)?;
    let initial_budget = config.embedding.max_input_chars.unwrap_or(10_000);
    let mut batcher = AdaptiveBatcher::new(initial_budget);
    // Rebuild cache from Qdrant — indexer::run dropped its local copy.
    // Cheap (one scroll, ~1 s for our usual corpus size).
    let mut cache: HashMap<PathBuf, String> = vs.scroll_files().await?;
    let filter = PathFilter::new(&config.project, &config.index)?;
    info!(cached_files = cache.len(), "watcher: cache rebuilt");

    // notify-debouncer-full uses a sync callback. Bridge to async via an
    // unbounded mpsc; the debouncer already aggregates events over the
    // debounce window, so the channel sees one message per quiet period
    // — bounded growth in practice.
    let (tx, mut rx) = mpsc::unbounded_channel::<Vec<DebouncedEvent>>();
    let mut debouncer = new_debouncer(
        Duration::from_millis(config.watcher.debounce_ms),
        None,
        move |result: DebounceEventResult| match result {
            Ok(events) => {
                let _ = tx.send(events);
            }
            Err(errs) => {
                for e in errs {
                    warn!(error = %e, "notify debouncer error");
                }
            }
        },
    )
    .context("creating notify debouncer")?;
    debouncer
        .watch(&config.project.root, RecursiveMode::Recursive)
        .context("starting filesystem watch")?;
    info!("watcher: subscribed to filesystem events");

    let ctrl_c = tokio::signal::ctrl_c();
    tokio::pin!(ctrl_c);

    loop {
        tokio::select! {
            biased;
            _ = &mut ctrl_c => {
                info!("watcher: ctrl-c received, shutting down");
                return Ok(());
            }
            maybe_events = rx.recv() => {
                let Some(events) = maybe_events else {
                    // Channel closed — debouncer dropped. Shouldn't happen
                    // while we hold it, but bail cleanly.
                    warn!("watcher: event channel closed; exiting");
                    return Ok(());
                };
                handle_batch(
                    events,
                    config,
                    &filter,
                    &embedder,
                    &vs,
                    &mut bm25,
                    &chunker,
                    &mut cache,
                    &mut batcher,
                )
                .await;
            }
        }
    }
}

/// Process one debounced batch. We deduplicate paths within the batch
/// (a file may be touched multiple times during the debounce window) and
/// classify each into either "process" (file currently exists & passes
/// the filter) or "delete" (file no longer exists or was outright removed).
#[allow(clippy::too_many_arguments)]
async fn handle_batch(
    events: Vec<DebouncedEvent>,
    config: &Config,
    filter: &PathFilter,
    embedder: &embedding::Client,
    vs: &vector_store::Client,
    bm25: &mut Bm25Index,
    chunker: &ChunkerSet,
    cache: &mut HashMap<PathBuf, String>,
    batcher: &mut AdaptiveBatcher,
) {
    // Dedup by absolute path. We don't differentiate by event kind here:
    // if the path exists post-debounce, treat as upsert; if it doesn't,
    // treat as delete. This naturally collapses "rename A B" into
    // "delete A + upsert B".
    let mut seen: HashSet<PathBuf> = HashSet::new();
    let mut paths: Vec<PathBuf> = Vec::new();
    for ev in &events {
        for path in &ev.paths {
            if seen.insert(path.clone()) {
                paths.push(path.clone());
            }
        }
    }
    debug!(
        events = events.len(),
        unique_paths = paths.len(),
        "watcher: batch received"
    );

    for path in paths {
        // Try to classify: does the path currently exist as something we
        // want indexed?
        match filter.check(&path) {
            Some(entry) => {
                // File present + indexable → reprocess.
                let mut stats = IndexStats::default();
                let started = std::time::Instant::now();
                match indexer::process_one_file(
                    embedder, vs, bm25, chunker, cache, &mut stats, &entry, batcher,
                )
                .await
                {
                    Ok(()) => {
                        let elapsed_ms = started.elapsed().as_millis() as u64;
                        if stats.files_indexed > 0 {
                            info!(
                                file = %entry.relative.display(),
                                elapsed_ms,
                                "watcher: indexed"
                            );
                        } else if stats.files_unchanged > 0 {
                            // Sha cache hit — content didn't actually change
                            // (event was spurious / formatter touched mtime).
                            debug!(
                                file = %entry.relative.display(),
                                "watcher: unchanged (sha hit)"
                            );
                        }
                    }
                    Err(e) => {
                        // Per-file failure is isolated; don't crash the watcher.
                        warn!(
                            file = %entry.relative.display(),
                            error = ?e,
                            "watcher: failed to index file"
                        );
                    }
                }
            }
            None => {
                // Either the path was deleted, was never indexable
                // (wrong extension / gitignored / excluded), or is a
                // directory event we don't care about. The only thing
                // that needs action is the deletion case: if this path
                // is in our cache, remove it.
                //
                // We key the cache by the *relative* path; convert.
                let relative = path
                    .strip_prefix(&config.project.root)
                    .unwrap_or(&path)
                    .to_path_buf();
                if cache.contains_key(&relative) {
                    match indexer::delete_file_from_indexes(vs, bm25, cache, &relative).await {
                        Ok(()) => info!(
                            file = %relative.display(),
                            "watcher: removed (file deleted)"
                        ),
                        Err(e) => error!(
                            file = %relative.display(),
                            error = ?e,
                            "watcher: failed to remove deleted file from indexes"
                        ),
                    }
                }
                // Otherwise: a directory event, an irrelevant extension,
                // or a noise event. Silently drop.
            }
        }
    }
}
