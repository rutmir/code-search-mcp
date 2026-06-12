use anyhow::{Context, Result};
use clap::{Parser, Subcommand};
use std::path::PathBuf;
use tracing::{info, warn};

mod adaptive_batcher;
mod bm25;
mod chunker;
mod config;
mod embedding;
mod indexer;
mod reranker;
mod search;
mod serve;
mod vector_store;
mod walker;
mod watcher;

use config::Config;

#[derive(Parser)]
#[command(name = "code-search-mcp", version, about, long_about = None)]
struct Cli {
    #[arg(long, short = 'c', default_value = ".claude/code-search.toml")]
    config: PathBuf,

    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand)]
enum Command {
    /// Validate config, ping configured services, verify embedding dimensions match
    Check,
    /// Walk the project, (re)index changed files, sync Qdrant + tantivy + state.json
    Index,
    /// Hybrid search: dense (Qdrant) ∪ sparse (BM25) → RRF merge → optional rerank.
    Search {
        /// Query string. Quote it if it contains spaces / shell-special chars.
        query: String,
        /// Max results to display.
        #[arg(long, short = 'n', default_value_t = 10)]
        limit: usize,
        /// Skip the reranker stage (faster, lower quality).
        #[arg(long)]
        no_rerank: bool,
        /// Only show hits in this language (e.g. "rust", "markdown").
        #[arg(long)]
        lang: Option<String>,
        /// Only show hits whose file path contains this substring.
        #[arg(long)]
        path: Option<String>,
        /// Output as JSON instead of the pretty human-readable format.
        #[arg(long)]
        json: bool,
    },
    /// Wipe ALL indexed state for this project: drops the Qdrant collection
    /// and removes the tantivy index directory. Next `index` rebuilds from scratch.
    Clear {
        /// Skip the interactive confirmation prompt (for scripts / CI).
        #[arg(long)]
        yes: bool,
    },
    /// Run as an MCP server over stdio. Exposes a single tool
    /// `code_search`. Used by Claude Code via `.mcp.json` at the
    /// project root.
    Serve,
    /// Watch the project tree and incrementally reindex on file changes.
    /// First does a full sync via `index` (cheap if no files changed
    /// since last run, thanks to the sha256 cache), then subscribes to
    /// filesystem events and processes them in debounced batches.
    /// Ctrl-C to stop.
    Watch,
}

#[tokio::main]
async fn main() -> Result<()> {
    init_tracing();

    let cli = Cli::parse();
    let config = Config::load(&cli.config)?;
    info!(
        project = %config.project.id,
        root = %config.project.root.display(),
        "config loaded"
    );

    match cli.command {
        Command::Check => run_check(&config).await,
        Command::Index => run_index(&config).await,
        Command::Search {
            query,
            limit,
            no_rerank,
            lang,
            path,
            json,
        } => {
            run_search(
                &config,
                &query,
                limit,
                !no_rerank,
                lang.as_deref(),
                path.as_deref(),
                json,
            )
            .await
        }
        Command::Clear { yes } => run_clear(&config, yes).await,
        Command::Serve => serve::run(&config).await,
        Command::Watch => watcher::run(&config).await,
    }
}

async fn run_search(
    config: &Config,
    query: &str,
    limit: usize,
    use_rerank: bool,
    lang: Option<&str>,
    path: Option<&str>,
    json_out: bool,
) -> Result<()> {
    let started = std::time::Instant::now();
    let results = search::run(
        config,
        search::SearchParams {
            query,
            limit,
            use_rerank,
            lang,
            path,
        },
    )
    .await?;
    let elapsed_ms = started.elapsed().as_millis() as u64;

    if json_out {
        // Compact JSON for piping into other tools.
        let arr: Vec<serde_json::Value> = results
            .iter()
            .map(|r| {
                serde_json::json!({
                    "file": r.file,
                    "start_line": r.start_line,
                    "end_line": r.end_line,
                    "lang": r.lang,
                    "kind": r.kind,
                    "name": r.name,
                    "score": r.score,
                    "dense_score": r.dense_score,
                    "sparse_score": r.sparse_score,
                    "rerank_score": r.rerank_score,
                    "preview": r.preview,
                })
            })
            .collect();
        println!("{}", serde_json::to_string_pretty(&arr)?);
    } else {
        print_results_pretty(&results, query, elapsed_ms);
    }
    Ok(())
}

fn print_results_pretty(results: &[search::SearchResult], query: &str, elapsed_ms: u64) {
    eprintln!("query: {}", query);
    eprintln!("results: {} ({} ms)", results.len(), elapsed_ms);
    eprintln!();
    if results.is_empty() {
        eprintln!("(no results)");
        return;
    }
    for (i, r) in results.iter().enumerate() {
        // Two-line header: rank + score chain + location, then the preview.
        // Preview is indented and truncated for shell-friendliness.
        let mut score_parts = vec![format!("score={:.4}", r.score)];
        if let Some(d) = r.dense_score {
            score_parts.push(format!("dense={:.3}", d));
        }
        if let Some(s) = r.sparse_score {
            score_parts.push(format!("bm25={:.3}", s));
        }
        if let Some(rr) = r.rerank_score {
            score_parts.push(format!("rerank={:.3}", rr));
        }
        let symbol = match (&r.kind, &r.name) {
            (Some(k), Some(n)) => format!("  {} {}", k, n),
            (Some(k), None) => format!("  {}", k),
            _ => String::new(),
        };
        println!(
            "#{:<3}  {}:{}-{}  [{}]{}  {}",
            i + 1,
            r.file,
            r.start_line,
            r.end_line,
            r.lang,
            symbol,
            score_parts.join("  "),
        );
        // Compact preview: collapse whitespace runs to single spaces, trim,
        // cap at ~240 chars. Caller can `--json` for the full thing.
        let preview: String = r
            .preview
            .split_whitespace()
            .collect::<Vec<_>>()
            .join(" ")
            .chars()
            .take(240)
            .collect();
        println!("      {}", preview);
        println!();
    }
}

async fn run_index(config: &Config) -> Result<()> {
    let started = std::time::Instant::now();
    let stats = indexer::run(config).await?;
    info!(
        scanned = stats.files_scanned,
        indexed = stats.files_indexed,
        unchanged = stats.files_unchanged,
        removed = stats.files_removed,
        failed = stats.files_failed,
        chunks_added = stats.chunks_upserted,
        chunks_deleted = stats.chunks_deleted,
        elapsed_ms = started.elapsed().as_millis() as u64,
        "indexing complete"
    );
    Ok(())
}

async fn run_clear(config: &Config, yes: bool) -> Result<()> {
    // Inspect what's actually there so the prompt and the summary are honest.
    // Reading both stores is cheap and lets us skip work if nothing exists.
    let vs = vector_store::Client::new(
        &config.vector_store,
        config.vector_store.resolve_collection_name(&config.project),
    )
    .await?;
    let qdrant_points = if vs.collection_exists().await? {
        Some(vs.points_count().await.unwrap_or(0))
    } else {
        None
    };
    let tantivy_exists = config.bm25.index_path.exists();

    if qdrant_points.is_none() && !tantivy_exists {
        info!(
            project = %config.project.id,
            "nothing to clear — neither qdrant collection nor tantivy index dir exists"
        );
        return Ok(());
    }

    warn!(
        project = %config.project.id,
        qdrant_collection = %vs.collection_name(),
        qdrant_points = qdrant_points.unwrap_or(0),
        tantivy_path = %config.bm25.index_path.display(),
        "about to WIPE all indexed state for this project"
    );

    if !yes {
        // Interactive confirmation. Anything but exactly "yes" aborts.
        // stderr for prompt so it interleaves correctly with tracing logs (also stderr).
        use std::io::{BufRead, Write};
        eprint!("Type 'yes' to confirm: ");
        std::io::stderr().flush().ok();
        let stdin = std::io::stdin();
        let mut line = String::new();
        stdin.lock().read_line(&mut line).ok();
        if line.trim() != "yes" {
            info!("aborted; nothing was deleted");
            return Ok(());
        }
    }

    // Qdrant first. If this fails, tantivy stays intact and the user can retry.
    if qdrant_points.is_some() {
        vs.delete_collection().await?;
        info!(
            collection = %vs.collection_name(),
            "qdrant collection dropped"
        );
    }

    // Tantivy: remove the whole index directory. Next `index` will recreate it.
    // We can't reuse the open IndexWriter here because tantivy holds locks on
    // segment files; nuking the dir is simpler and equally correct.
    if tantivy_exists {
        std::fs::remove_dir_all(&config.bm25.index_path).with_context(|| {
            format!("removing tantivy dir: {}", config.bm25.index_path.display())
        })?;
        info!(
            path = %config.bm25.index_path.display(),
            "tantivy index directory removed"
        );
    }

    info!("clear complete — next `index` will rebuild from scratch");
    Ok(())
}

async fn run_check(config: &Config) -> Result<()> {
    info!("checking embedding endpoint");
    let embedder = embedding::Client::new(&config.embedding);
    let actual_dim = embedder.probe_dimensions().await?;
    if actual_dim != config.embedding.dimensions {
        anyhow::bail!(
            "embedding dimension mismatch: config declares {}, endpoint returned {}",
            config.embedding.dimensions,
            actual_dim
        );
    }
    info!(
        model = %config.embedding.model,
        dimensions = actual_dim,
        "embedding endpoint OK"
    );

    info!("checking qdrant");
    let vs = vector_store::Client::new(
        &config.vector_store,
        config.vector_store.resolve_collection_name(&config.project),
    )
    .await?;
    if vs.collection_exists().await? {
        let count = vs.points_count().await?;
        info!(
            collection = %vs.collection_name(),
            points = count,
            "qdrant collection exists"
        );
        // Read-only marker check — surfaces fingerprint mismatch in `check`
        // so the user catches a misconfigured `vector_store.collection`
        // BEFORE running index / search and corrupting the wrong DB.
        // (verify_marker_read_only logs its own status at debug level.)
        vs.verify_marker_read_only(config).await?;
    } else {
        warn!(
            collection = %vs.collection_name(),
            "qdrant collection does not exist yet — will be created on first `index`"
        );
    }

    if let Some(reranker_cfg) = &config.reranker {
        if reranker_cfg.enabled {
            info!("checking reranker endpoint");
            let rer = reranker::Client::new(reranker_cfg);
            // Tiny contrastive probe: a healthy cross-encoder must score the
            // on-topic document above the off-topic one. Catches not just
            // dead servers but the "loads cleanly, outputs garbage scores"
            // failure mode of broken classifier heads.
            let scores = rer
                .rerank(
                    "apple fruit",
                    vec![
                        "An apple is an edible fruit produced by an apple tree.".to_string(),
                        "The mutex guards the scheduler queue against data races.".to_string(),
                    ],
                )
                .await
                .context("reranker probe")?;
            if scores.len() != 2 {
                anyhow::bail!(
                    "reranker probe returned {} scores for 2 documents",
                    scores.len()
                );
            }
            if scores[0] <= scores[1] {
                warn!(
                    on_topic = scores[0],
                    off_topic = scores[1],
                    "reranker scored an off-topic document above an on-topic one — \
                     model or server is likely misconfigured (wrong pooling, broken \
                     classifier head, wrong model file)"
                );
            } else {
                info!(
                    model = %reranker_cfg.model,
                    on_topic = scores[0],
                    off_topic = scores[1],
                    "reranker endpoint OK"
                );
            }

            // Chunking vs reranker-truncation sanity: a chunk bigger than
            // max_document_chars gets cut before the cross-encoder sees it,
            // so the reranker judges a fragment while the LLM later reads
            // the full chunk. Legal, but worth knowing about.
            let limit = reranker_cfg.max_document_chars;
            let mut oversized: Vec<(String, usize)> = Vec::new();
            if config.chunking.max_chunk_chars > limit {
                oversized.push(("default".to_string(), config.chunking.max_chunk_chars));
            }
            for (lang, lc) in &config.chunking.per_language {
                let chars = lc
                    .max_chunk_chars
                    .unwrap_or(config.chunking.max_chunk_chars);
                if chars > limit {
                    oversized.push((lang.clone(), chars));
                }
            }
            for (lang, chars) in oversized {
                warn!(
                    lang = %lang,
                    max_chunk_chars = chars,
                    max_document_chars = limit,
                    "chunks can exceed the reranker's truncation limit — the \
                     cross-encoder will rank such chunks by their first \
                     max_document_chars chars only. Consider lowering \
                     chunking.max_chunk_chars or raising reranker.max_document_chars"
                );
            }
        }
    }

    info!("all checks passed");
    Ok(())
}

/// Tracing setup: everything goes to stderr.
/// stdout is reserved for MCP JSON-RPC framing in `serve` mode.
fn init_tracing() {
    use tracing_subscriber::{fmt, EnvFilter};
    fmt()
        .with_writer(std::io::stderr)
        .with_env_filter(
            EnvFilter::try_from_default_env()
                // Default: our crate at INFO, third-party at INFO, but tantivy
                // muzzled to WARN — its routine "Preparing commit / GC / save
                // metas / Deleted .idx" chatter at INFO drowns out our own logs
                // (5-10x ratio) and made apparent "log gaps" look like hangs.
                // Override with RUST_LOG=... for debugging.
                .unwrap_or_else(|_| EnvFilter::new("code_search_mcp=info,tantivy=warn,info")),
        )
        .init();
}
