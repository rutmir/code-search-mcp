use anyhow::{Context, Result};
use serde::Deserialize;
use std::collections::HashMap;
use std::path::{Path, PathBuf};

#[derive(Debug, Deserialize, Clone)]
pub struct Config {
    pub project: ProjectConfig,
    pub index: IndexConfig,
    pub embedding: EmbeddingConfig,
    pub vector_store: VectorStoreConfig,
    pub bm25: Bm25Config,
    pub reranker: Option<RerankerConfig>,
    pub watcher: WatcherConfig,
    pub chunking: ChunkingConfig,
    /// Optional search-tuning block. All fields default to the values
    /// hard-coded in `src/search.rs` (quality-first: K_DENSE=K_SPARSE=
    /// RERANK_TOP_N=30, RRF_K=60). Override per-project for
    /// hardware-specific tuning without a recompile.
    #[serde(default)]
    pub search: SearchConfig,
    /// Optional serve-mode block.
    #[serde(default)]
    pub serve: ServeConfig,
}

#[derive(Debug, Deserialize, Clone, Default)]
pub struct SearchConfig {
    /// Top-K candidates pulled from the dense (Qdrant) side. Bigger =
    /// better recall at linear reranker token cost.
    pub dense_k: Option<usize>,
    /// Top-K candidates pulled from the sparse (BM25/tantivy) side.
    pub sparse_k: Option<usize>,
    /// Only the top-N candidates by RRF score are reranked by the
    /// cross-encoder; the tail keeps RRF score. Cap on reranker work.
    pub rerank_top_n: Option<usize>,
    /// Reciprocal Rank Fusion constant. Smaller favors top-of-list
    /// matches more aggressively; 60 is the Cormack et al. default.
    pub rrf_k: Option<usize>,
    /// Weight of the reranker's rank-vote in the final fusion, relative
    /// to a single retrieval modality (dense or sparse each contribute
    /// with weight 1). Default 2.0: the cross-encoder counts as two
    /// modalities — a strong vote, not a veto. Set to 0.0 to rank purely
    /// by retrieval RRF while still reporting rerank scores.
    pub rerank_weight: Option<f32>,
    /// Boost for candidates whose AST symbol name is literally named in
    /// the query (e.g. query "AdaptiveBatcher::note_failure timeout"
    /// hitting the chunk named `AdaptiveBatcher::note_failure`), in units
    /// of a #1 rank-vote: `bonus = symbol_boost / (rrf_k + 1)`.
    /// Default 1.0; 0.0 disables. Makes `code_search` competitive with
    /// grep for exact-symbol lookups.
    pub symbol_boost: Option<f32>,
}

#[derive(Debug, Deserialize, Clone, Default)]
pub struct ServeConfig {
    /// When set, `serve` appends one JSON line per `code_search` call to
    /// this file: timestamp, query, filters, result count, latency, and
    /// the top hits. MCP clients (Claude Code included) do not persist
    /// the server's stderr beyond connection start, so this is the only
    /// durable record of what was asked and what came back — invaluable
    /// when tuning retrieval quality or diagnosing why the calling LLM
    /// stopped using the tool. Off by default; `~` and env vars expand.
    pub query_log_path: Option<PathBuf>,
}

#[derive(Debug, Deserialize, Clone)]
pub struct ProjectConfig {
    pub id: String,
    pub root: PathBuf,
}

#[derive(Debug, Deserialize, Clone)]
pub struct IndexConfig {
    /// Languages to index. **Optional.** When omitted or empty, the walker
    /// indexes every file that isn't a known-binary type or oversize (see
    /// `walker::index_all_default`) — zero-config for a new project. When
    /// non-empty, it acts as an explicit whitelist (only those languages'
    /// extensions are walked), which is the way to *narrow* a noisy repo.
    #[serde(default)]
    pub languages: Vec<String>,
    #[serde(default = "default_true")]
    pub respect_gitignore: bool,
    #[serde(default)]
    pub exclude: Vec<String>,
}

#[derive(Debug, Deserialize, Clone)]
pub struct EmbeddingConfig {
    /// Kept for TOML clarity (`"openai-compatible"` etc.) and as a
    /// forward-compatible discriminator if/when we support multiple
    /// backends. The code currently hardcodes the openai-compatible
    /// HTTP shape.
    #[allow(dead_code)]
    pub provider: String,
    pub url: String,
    pub model: String,
    pub dimensions: usize,
    /// DEPRECATED: ignored. Batch size is now determined dynamically by the
    /// AIMD adaptive batcher. Kept Option<...> so existing configs keep parsing;
    /// presence triggers an advisory log at startup.
    #[serde(default)]
    pub batch_size: Option<usize>,
    /// Outer HTTP timeout, and the transport's hard cap. The adaptive
    /// batcher derives its own per-request timeout from observed throughput
    /// and keeps it strictly under this value, so a slow batch always fails
    /// as the batcher's timeout rather than as an indistinguishable
    /// transport error. Lowering this therefore also lowers the largest
    /// batch the batcher will ever assemble.
    #[serde(default = "default_embedding_timeout")]
    pub timeout_secs: u64,
    /// Advisory: starting hint for the adaptive batcher's `budget_chars`.
    /// AIMD will halve it down if it's too optimistic, or grow it up to
    /// ~256 KB if conservative. If unset, defaults to 10_000.
    #[serde(default)]
    pub max_input_chars: Option<usize>,
    /// On `index` start, wait up to N seconds for the embedding server to become ready
    /// (it returns 503 "Loading model" while warming up — typically 1-3 min for jina-code).
    /// Set to 0 to disable the wait.
    #[serde(default = "default_startup_wait_secs")]
    pub startup_wait_secs: u64,
}

#[derive(Debug, Deserialize, Clone)]
pub struct VectorStoreConfig {
    pub provider: String,
    pub url: String,
    /// Collection name. **Recommended: leave unset** so it auto-derives
    /// from `{project.id}_{8-hex-sha256(project.id + canonical_root)}` —
    /// guarantees no accidental collisions between different users or
    /// different checkouts of the same project name. Explicit overrides
    /// are still supported (for migrations, intentional sharing) but
    /// trigger a marker-point verification at startup that hard-fails on
    /// fingerprint mismatch.
    #[serde(default)]
    pub collection: Option<String>,
    #[serde(default = "default_qdrant_timeout")]
    pub timeout_secs: u64,
}

impl VectorStoreConfig {
    /// Resolve the actual collection name: use the explicit override if
    /// present, otherwise auto-derive from project id + canonical root.
    /// The derivation is stable: same project at same path always yields
    /// the same name; same project at a different path yields a different
    /// name (which is the point — no accidental collisions).
    pub fn resolve_collection_name(&self, project: &ProjectConfig) -> String {
        if let Some(explicit) = self.collection.as_ref() {
            if !explicit.trim().is_empty() {
                return explicit.clone();
            }
        }
        derive_collection_name(project)
    }
}

fn derive_collection_name(project: &ProjectConfig) -> String {
    use sha2::{Digest, Sha256};
    // canonicalize in Config::load already ran, so project.root is absolute.
    let root_str = project.root.to_string_lossy();
    let mut hasher = Sha256::new();
    hasher.update(project.id.as_bytes());
    hasher.update(b"\0");
    hasher.update(root_str.as_bytes());
    let digest = hasher.finalize();
    let suffix = hex::encode(&digest[..4]); // 8 hex chars
    let slug = slugify_for_qdrant(&project.id);
    format!("{}_{}", slug, suffix)
}

/// Qdrant collection names: ASCII letters / digits / `-` / `_`. Replace
/// anything else with `_`, collapse runs.
fn slugify_for_qdrant(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    let mut last_underscore = false;
    for c in s.chars() {
        let safe = c.is_ascii_alphanumeric() || c == '-' || c == '_';
        if safe {
            out.push(c);
            last_underscore = false;
        } else if !last_underscore {
            out.push('_');
            last_underscore = true;
        }
    }
    out
}

#[derive(Debug, Deserialize, Clone)]
pub struct Bm25Config {
    #[allow(dead_code)]
    pub provider: String,
    pub index_path: PathBuf,
}

#[derive(Debug, Deserialize, Clone)]
pub struct RerankerConfig {
    #[allow(dead_code)]
    pub provider: String,
    pub url: String,
    pub model: String,
    #[serde(default = "default_true")]
    pub enabled: bool,
    #[serde(default = "default_reranker_timeout")]
    pub timeout_secs: u64,
    /// Truncate each candidate document to this many chars before sending
    /// to the reranker. Default 8000 — sized for bge-reranker-v2-m3 at its
    /// native ctx=8192 (8000 chars ≈ 2700 ASCII-code tokens + query +
    /// special tokens). The server's physical batch must fit it too; if a
    /// batch is still rejected as too large, the client halves the limit
    /// and retries (see `reranker::Client::rerank`).
    #[serde(default = "default_reranker_max_doc_chars")]
    pub max_document_chars: usize,
}

#[derive(Debug, Deserialize, Clone)]
pub struct WatcherConfig {
    #[serde(default = "default_true")]
    pub enabled: bool,
    #[serde(default = "default_debounce_ms")]
    pub debounce_ms: u64,
}

#[derive(Debug, Deserialize, Clone)]
pub struct ChunkingConfig {
    /// Default strategy for languages without an entry in `per_language`.
    /// "lines" — line-window chunker; "headings" — markdown H1/H2-aware
    /// chunker.
    pub strategy: String,
    #[serde(default = "default_max_chunk_lines")]
    pub max_chunk_lines: usize,
    #[serde(default = "default_overlap_lines")]
    pub overlap_lines: usize,
    /// Byte-size cap per chunk. The chunker stops accumulating lines whenever
    /// either `max_chunk_lines` OR `max_chunk_chars` is hit, whichever comes
    /// first. Keep this comfortably below `embedding.max_input_chars` so a
    /// produced chunk doesn't get rejected downstream.
    #[serde(default = "default_max_chunk_chars")]
    pub max_chunk_chars: usize,
    /// Per-language overrides. Keyed by walker's language string (`"rust"`,
    /// `"markdown"`, `"toml"`, etc — see `walker::language_from_ext`).
    /// Any field not specified here falls back to the top-level default.
    /// Typical use: `[chunking.per_language.markdown] strategy = "headings"`
    /// so docs cut on section boundaries instead of line windows.
    #[serde(default)]
    pub per_language: HashMap<String, LanguageChunkConfig>,
}

/// Per-language override block. Every field is optional; whatever's set
/// replaces the top-level `[chunking]` default for files in that language.
#[derive(Debug, Deserialize, Default, Clone)]
pub struct LanguageChunkConfig {
    pub strategy: Option<String>,
    pub max_chunk_lines: Option<usize>,
    pub overlap_lines: Option<usize>,
    pub max_chunk_chars: Option<usize>,
}

fn default_true() -> bool {
    true
}
fn default_debounce_ms() -> u64 {
    500
}
fn default_max_chunk_lines() -> usize {
    80
}
fn default_overlap_lines() -> usize {
    15
}
fn default_max_chunk_chars() -> usize {
    // Upper bound on a single chunk's byte size. Chunks above this would
    // dominate any batch; the chunker breaks on either max_chunk_lines or
    // max_chunk_chars (whichever hits first). Sub-splits on sentence
    // boundaries for very long lines (markdown, generated content).
    20_000
}
fn default_embedding_timeout() -> u64 {
    300
}
fn default_startup_wait_secs() -> u64 {
    300
}
fn default_qdrant_timeout() -> u64 {
    60
}
fn default_reranker_timeout() -> u64 {
    120
}
fn default_reranker_max_doc_chars() -> usize {
    // Quality-first: 8000 chars ≈ 2700 tokens leaves the cross-encoder
    // looking at full function bodies (most Rust functions ≤ 250 lines
    // ≈ 7000 chars), not just signatures. Fits bge-reranker-v2-m3 at
    // its native ctx=8192 with safe headroom for query + special tokens.
    // Drop to 4000 if your search latency is unacceptably long; drop to
    // 2000 if your reranker is stuck at legacy ctx=1024.
    8_000
}

impl Config {
    pub fn load(path: &Path) -> Result<Self> {
        let raw = std::fs::read_to_string(path)
            .with_context(|| format!("reading config: {}", path.display()))?;
        let mut cfg: Config =
            toml::from_str(&raw).with_context(|| format!("parsing config: {}", path.display()))?;

        cfg.project.root = expand_path(&cfg.project.root)?;
        cfg.bm25.index_path = expand_path(&cfg.bm25.index_path)?;
        if let Some(p) = &cfg.serve.query_log_path {
            cfg.serve.query_log_path = Some(expand_path(p)?);
        }

        if !cfg.project.root.exists() {
            anyhow::bail!(
                "project.root does not exist: {}",
                cfg.project.root.display()
            );
        }

        // Canonicalize project.root to an absolute, symlink-resolved path.
        // The walker is permissive about relative roots, but notify (used by
        // `watch`) reports absolute paths in events — so any relative root
        // creates a strip_prefix mismatch between the two code paths and
        // chunks end up stored under inconsistent `file=` keys. Canonicalize
        // once here and both code paths agree.
        cfg.project.root = std::fs::canonicalize(&cfg.project.root).with_context(|| {
            format!(
                "canonicalizing project.root: {}",
                cfg.project.root.display()
            )
        })?;

        // Advisory: knobs that were static in earlier versions are now derived
        // by the AIMD adaptive batcher at runtime. Don't error — just inform.
        if let Some(b) = cfg.embedding.batch_size {
            tracing::info!(
                value = b,
                "embedding.batch_size is now ignored — batch sizes are derived by the adaptive batcher"
            );
        }
        if let Some(m) = cfg.embedding.max_input_chars {
            tracing::info!(
                value = m,
                "embedding.max_input_chars is now advisory (used as the adaptive batcher's initial budget hint)"
            );
        }

        Ok(cfg)
    }
}

fn expand_path(p: &Path) -> Result<PathBuf> {
    let s = p.to_str().context("non-utf8 path in config")?;
    let expanded = shellexpand::full(s)
        .with_context(|| format!("expanding path: {}", s))?
        .into_owned();
    Ok(PathBuf::from(expanded))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn project(id: &str, root: &str) -> ProjectConfig {
        ProjectConfig {
            id: id.to_string(),
            root: PathBuf::from(root),
        }
    }

    fn store(collection: Option<&str>) -> VectorStoreConfig {
        VectorStoreConfig {
            provider: "qdrant".to_string(),
            url: "http://localhost:6333".to_string(),
            collection: collection.map(str::to_string),
            timeout_secs: 60,
        }
    }

    /// The derived name is the only thing keeping two checkouts from
    /// sharing a collection, and a change to it silently orphans every
    /// existing index. It must be stable and it must separate.
    #[test]
    fn derived_collection_name_is_stable_and_separating() {
        let a = derive_collection_name(&project("roex", "/home/u/proj"));
        assert_eq!(a, derive_collection_name(&project("roex", "/home/u/proj")));
        // Same project name, different checkout — must not collide.
        assert_ne!(a, derive_collection_name(&project("roex", "/home/u/proj2")));
        // Same path, renamed project — must not collide either.
        assert_ne!(a, derive_collection_name(&project("roex2", "/home/u/proj")));
    }

    #[test]
    fn derived_collection_name_is_qdrant_safe() {
        let name = derive_collection_name(&project("Ω my proj/v2!", "/tmp/x"));
        assert!(
            name.chars()
                .all(|c| c.is_ascii_alphanumeric() || c == '-' || c == '_'),
            "unsafe collection name: {name}"
        );
        // Ends in the 8-hex discriminator.
        let suffix = name.rsplit('_').next().unwrap();
        assert_eq!(suffix.len(), 8);
        assert!(suffix.chars().all(|c| c.is_ascii_hexdigit()));
    }

    #[test]
    fn slugify_collapses_unsafe_runs() {
        assert_eq!(slugify_for_qdrant("plain-name_1"), "plain-name_1");
        assert_eq!(slugify_for_qdrant("a  //  b"), "a_b");
        assert_eq!(slugify_for_qdrant("привет"), "_");
    }

    #[test]
    fn explicit_collection_overrides_derivation() {
        let p = project("roex", "/home/u/proj");
        assert_eq!(
            store(Some("legacy_name")).resolve_collection_name(&p),
            "legacy_name"
        );
        // A blank override is a config accident, not a request for a
        // collection named "  " — fall back to the derived name.
        for blank in [Some(""), Some("   "), None] {
            assert_eq!(
                store(blank).resolve_collection_name(&p),
                derive_collection_name(&p),
                "blank override {blank:?} should fall back"
            );
        }
    }
}
