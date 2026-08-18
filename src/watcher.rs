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
    let mut batcher = AdaptiveBatcher::new(
        initial_budget,
        Duration::from_secs(config.embedding.timeout_secs),
    );
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

/// Which indexed files a non-indexable path event should remove.
///
/// The direct case is a tracked file that was deleted. The case that used
/// to be missed: `mv src/ old_src/` (or `rm -rf src/`) reports the
/// *directory*, and nothing ever reports its contents — so every chunk
/// under it survived in both stores until the next full `index` ran its
/// stale-detection pass. Treating a vanished path as a prefix over the
/// cache closes that gap.
///
/// The prefix scan is gated on the path no longer existing, because this
/// branch is also the hot path for pure noise (build artifacts, ignored
/// files), and those must not cost a scan of the whole cache.
fn vanished_files(
    cache: &HashMap<PathBuf, String>,
    absolute: &std::path::Path,
    relative: &std::path::Path,
) -> Vec<PathBuf> {
    if cache.contains_key(relative) {
        return vec![relative.to_path_buf()];
    }
    if absolute.exists() {
        return Vec::new();
    }
    // `starts_with` is component-wise, so `src/foo` never matches
    // `src/foobar`.
    cache
        .keys()
        .filter(|p| p.starts_with(relative))
        .cloned()
        .collect()
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
                // that needs action is the deletion case.
                //
                // We key the cache by the *relative* path; convert.
                let relative = path
                    .strip_prefix(&config.project.root)
                    .unwrap_or(&path)
                    .to_path_buf();
                if relative.as_os_str().is_empty() {
                    // An event on the project root itself. Treating it as a
                    // prefix would wipe the entire index.
                    continue;
                }
                for victim in vanished_files(cache, &path, &relative) {
                    match indexer::delete_file_from_indexes(vs, bm25, cache, &victim).await {
                        Ok(()) => info!(
                            file = %victim.display(),
                            "watcher: removed (file deleted)"
                        ),
                        Err(e) => error!(
                            file = %victim.display(),
                            error = ?e,
                            "watcher: failed to remove deleted file from indexes"
                        ),
                    }
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cache_with(paths: &[&str]) -> HashMap<PathBuf, String> {
        paths
            .iter()
            .map(|p| (PathBuf::from(p), "sha".to_string()))
            .collect()
    }

    #[test]
    fn tracked_file_is_removed_directly() {
        let cache = cache_with(&["src/a.rs", "src/b.rs"]);
        let got = vanished_files(
            &cache,
            std::path::Path::new("/nonexistent/src/a.rs"),
            std::path::Path::new("src/a.rs"),
        );
        assert_eq!(got, vec![PathBuf::from("src/a.rs")]);
    }

    #[test]
    fn vanished_directory_removes_everything_under_it() {
        // notify reports `mv src/ old_src/` as one event on the directory;
        // without prefix expansion its files would stay indexed forever.
        let cache = cache_with(&["src/a.rs", "src/deep/b.rs", "docs/c.md"]);
        let mut got = vanished_files(
            &cache,
            std::path::Path::new("/nonexistent/src"),
            std::path::Path::new("src"),
        );
        got.sort();
        assert_eq!(
            got,
            vec![PathBuf::from("src/a.rs"), PathBuf::from("src/deep/b.rs")]
        );
    }

    #[test]
    fn prefix_match_respects_path_components() {
        // `src` must not swallow `srcgen/`.
        let cache = cache_with(&["srcgen/a.rs"]);
        let got = vanished_files(
            &cache,
            std::path::Path::new("/nonexistent/src"),
            std::path::Path::new("src"),
        );
        assert!(
            got.is_empty(),
            "component-wise prefix expected, got {got:?}"
        );
    }

    #[test]
    fn existing_but_unindexable_path_is_noise() {
        // A build artifact that the filter rejected: it still exists, so no
        // cache scan and nothing to delete.
        let cache = cache_with(&["src/a.rs"]);
        let here = std::path::Path::new(file!());
        let got = vanished_files(&cache, here, std::path::Path::new("target"));
        assert!(got.is_empty());
    }
}
