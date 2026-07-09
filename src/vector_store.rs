use anyhow::{Context, Result};
use reqwest::Client as HttpClient;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::PathBuf;
use std::time::Duration;

use crate::config::{Config, ProjectConfig, VectorStoreConfig};

/// Qdrant REST client: ping, ensure_collection, upsert/delete points,
/// vector search, and the project-identity marker handling.
pub struct Client {
    http: HttpClient,
    base_url: String,
    collection: String,
}

/// Hash of config fields that affect **chunk identity** — changing any
/// of these means the existing collection's chunks no longer match what
/// a new index pass would produce, so the indexer must `clear` first.
///
/// Includes: chunking strategy/params (default + per_language overrides),
/// embedding model + dimensions. Excludes: server URLs, timeouts, ranks,
/// reranker config, watcher config (none of those change chunk content).
pub fn config_hard_fingerprint(config: &Config) -> String {
    use sha2::{Digest, Sha256};
    let mut h = Sha256::new();
    h.update(b"v1\0chunking\0");
    h.update(config.chunking.strategy.as_bytes());
    h.update(b"\0");
    h.update(config.chunking.max_chunk_lines.to_le_bytes());
    h.update(config.chunking.overlap_lines.to_le_bytes());
    h.update(config.chunking.max_chunk_chars.to_le_bytes());
    // Sort per_language for determinism — HashMap iteration order varies.
    let mut langs: Vec<_> = config.chunking.per_language.iter().collect();
    langs.sort_by_key(|(a, _)| *a);
    h.update(b"\0per_language\0");
    for (lang, lc) in langs {
        h.update(lang.as_bytes());
        h.update(b":");
        if let Some(s) = &lc.strategy {
            h.update(s.as_bytes());
        }
        h.update(b"/");
        if let Some(v) = lc.max_chunk_lines {
            h.update(v.to_le_bytes());
        }
        h.update(b"/");
        if let Some(v) = lc.overlap_lines {
            h.update(v.to_le_bytes());
        }
        h.update(b"/");
        if let Some(v) = lc.max_chunk_chars {
            h.update(v.to_le_bytes());
        }
        h.update(b"\0");
    }
    h.update(b"\0embedding\0");
    h.update(config.embedding.model.as_bytes());
    h.update(b"\0");
    h.update(config.embedding.dimensions.to_le_bytes());
    // Tantivy schema/tokenizer revision: a bump invalidates the local BM25
    // index, and the only rebuild path that refills tantivy is the full
    // reprocess (the indexer's cache is Qdrant ∩ tantivy), so it rides the
    // hard fingerprint.
    h.update(b"\0bm25_schema\0");
    h.update(crate::bm25::SCHEMA_VERSION.to_le_bytes());
    hex::encode(h.finalize())
}

/// Hash of config fields that affect **the file set** — changing any of
/// these means the indexer will see different files, but existing chunks
/// for files that survive the change are still valid. Reindex without
/// clear; stale-detection handles cleanup of files that left the set.
pub fn config_soft_fingerprint(config: &Config) -> String {
    use sha2::{Digest, Sha256};
    let mut h = Sha256::new();
    h.update(b"v1\0languages\0");
    let mut langs = config.index.languages.clone();
    langs.sort();
    for l in &langs {
        h.update(l.as_bytes());
        h.update(b",");
    }
    h.update(b"\0exclude\0");
    let mut ex = config.index.exclude.clone();
    ex.sort();
    for e in &ex {
        h.update(e.as_bytes());
        h.update(b",");
    }
    h.update(b"\0respect_gitignore\0");
    h.update([u8::from(config.index.respect_gitignore)]);
    hex::encode(h.finalize())
}

/// Deterministic UUID v5 for the project-identity marker point. Same
/// across all installations — the marker is uniquely addressable inside
/// any code-search-mcp collection. Computed at call time to keep dep
/// usage local; cheap and called only once per startup.
pub fn marker_point_id() -> String {
    uuid::Uuid::new_v5(&uuid::Uuid::NAMESPACE_OID, b"__codesearch_mcp_marker__").to_string()
}

/// Payload of the project-identity marker point. Written at first
/// `ensure_collection`, verified at every startup.
///
/// Two-level fingerprint scheme:
///   - `fingerprint` (project identity): hash of `project_id ‖ root`.
///     Mismatch → another project's collection → hard fail.
///   - `config_hard` (chunk identity): hash of chunking + embedding
///     parameters that affect chunk_uuid / vector content. Mismatch →
///     auto-clear + reindex.
///   - `config_soft` (file set): hash of languages + exclude + gitignore
///     toggle. Mismatch → just reindex, no clear; stale chunks get
///     cleaned via stale-detection.
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct MarkerPayload {
    pub project_id: String,
    pub root: String,
    pub fingerprint: String,
    #[serde(default)]
    pub config_hard: Option<String>,
    #[serde(default)]
    pub config_soft: Option<String>,
    #[serde(default)]
    pub host: Option<String>,
    #[serde(default)]
    pub user: Option<String>,
    #[serde(default)]
    pub created_at: Option<String>,
    #[serde(default)]
    pub version: Option<String>,
}

/// Result of comparing the live config against the marker stored in the
/// collection. Drives the indexer's reindex / clear decisions.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MarkerStatus {
    /// No marker present (fresh collection or pre-marker version). Caller
    /// should write the marker.
    Fresh,
    /// All fingerprints match; nothing to do.
    Match,
    /// `config_soft` differs (languages / exclude / gitignore toggle).
    /// Caller proceeds with normal `index` — stale chunks get cleaned by
    /// the indexer's stale-detection pass. Marker should be updated.
    SoftChanged,
    /// `config_hard` differs (chunking strategy / params, embedding model
    /// or dimensions). Chunk identity has changed: caller must drop the
    /// collection AND wipe tantivy before reindexing, otherwise old and
    /// new chunks coexist with mismatched IDs. After clear, marker is
    /// rewritten with current snapshots.
    HardChanged,
}

#[derive(Deserialize)]
struct MarkerFetchResp {
    result: MarkerFetchInner,
}

#[derive(Deserialize)]
struct MarkerFetchInner {
    #[serde(default)]
    payload: MarkerPayload,
}

impl MarkerPayload {
    /// Compute the expected marker payload for the given config. Includes
    /// the project identity fingerprint plus two-level config snapshots.
    pub fn for_config(config: &Config) -> Self {
        let mut payload = Self::for_project(&config.project);
        payload.config_hard = Some(config_hard_fingerprint(config));
        payload.config_soft = Some(config_soft_fingerprint(config));
        payload
    }

    /// Compute the expected marker payload for the given project. The
    /// fingerprint is a stable hash of (project.id + canonical root) — it
    /// changes only when the project ID is renamed OR the project moves
    /// to a different absolute path, both of which warrant `clear` +
    /// reindex anyway.
    pub fn for_project(project: &ProjectConfig) -> Self {
        use sha2::{Digest, Sha256};
        let root_str = project.root.to_string_lossy().to_string();
        let mut hasher = Sha256::new();
        hasher.update(project.id.as_bytes());
        hasher.update(b"\0");
        hasher.update(root_str.as_bytes());
        let fingerprint = hex::encode(hasher.finalize());

        let host = std::env::var("HOSTNAME").ok().or_else(|| {
            std::fs::read_to_string("/etc/hostname")
                .ok()
                .map(|s| s.trim().to_string())
        });
        let user = std::env::var("USER").ok();
        let created_at = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .ok()
            .map(|d| d.as_secs().to_string());

        Self {
            project_id: project.id.clone(),
            root: root_str,
            fingerprint,
            config_hard: None,
            config_soft: None,
            host,
            user,
            created_at,
            version: Some(env!("CARGO_PKG_VERSION").to_string()),
        }
    }
}

#[derive(Serialize)]
pub struct QdrantPoint {
    pub id: String,
    pub vector: Vec<f32>,
    pub payload: serde_json::Value,
}

impl Client {
    pub async fn new(config: &VectorStoreConfig, collection: String) -> Result<Self> {
        if config.provider != "qdrant" {
            anyhow::bail!("unsupported vector_store provider: {}", config.provider);
        }
        let http = HttpClient::builder()
            .timeout(Duration::from_secs(config.timeout_secs))
            .pool_idle_timeout(Duration::from_secs(30))
            .build()
            .expect("reqwest client");
        let base_url = config.url.trim_end_matches('/').to_string();

        let resp = http
            .get(&base_url)
            .send()
            .await
            .with_context(|| format!("connecting to qdrant: {}", base_url))?;
        resp.error_for_status()
            .with_context(|| "qdrant root endpoint returned non-2xx")?;

        Ok(Self {
            http,
            base_url,
            collection,
        })
    }

    /// Resolved collection name (after auto-derive / override).
    pub fn collection_name(&self) -> &str {
        &self.collection
    }

    pub async fn collection_exists(&self) -> Result<bool> {
        let url = format!("{}/collections/{}", self.base_url, self.collection);
        let resp = self.http.get(&url).send().await?;
        match resp.status().as_u16() {
            200 => Ok(true),
            404 => Ok(false),
            other => anyhow::bail!("qdrant collection check returned HTTP {}", other),
        }
    }

    pub async fn points_count(&self) -> Result<u64> {
        let url = format!("{}/collections/{}", self.base_url, self.collection);
        let resp = self.http.get(&url).send().await?.error_for_status()?;
        let body: CollectionInfoResp = resp.json().await?;
        Ok(body.result.points_count.unwrap_or(0))
    }

    /// Create the collection if it doesn't exist, with Cosine distance and the given vector size.
    pub async fn ensure_collection(&self, dimensions: usize) -> Result<()> {
        if self.collection_exists().await? {
            return Ok(());
        }
        let url = format!("{}/collections/{}", self.base_url, self.collection);
        let body = serde_json::json!({
            "vectors": {
                "size": dimensions,
                "distance": "Cosine"
            }
        });
        self.http
            .put(&url)
            .json(&body)
            .send()
            .await
            .context("qdrant PUT collection")?
            .error_for_status()
            .context("qdrant create_collection non-2xx")?;
        tracing::info!(
            collection = %self.collection,
            dimensions = dimensions,
            "qdrant collection created"
        );
        Ok(())
    }

    /// Drop the entire collection. Idempotent: returns Ok if the collection
    /// already doesn't exist. Used by `clear` to wipe all indexed state.
    pub async fn delete_collection(&self) -> Result<()> {
        let url = format!("{}/collections/{}", self.base_url, self.collection);
        let resp = self
            .http
            .delete(&url)
            .send()
            .await
            .context("qdrant DELETE collection")?;
        match resp.status().as_u16() {
            // Qdrant returns 200 on successful delete and (depending on version)
            // 404 if it never existed. Both are fine for `clear`.
            200 | 404 => Ok(()),
            other => {
                let body = resp.text().await.unwrap_or_default();
                anyhow::bail!("qdrant delete_collection HTTP {}: {}", other, body.trim())
            }
        }
    }

    pub async fn upsert_points(&self, points: Vec<QdrantPoint>) -> Result<()> {
        if points.is_empty() {
            return Ok(());
        }
        let url = format!(
            "{}/collections/{}/points?wait=true",
            self.base_url, self.collection
        );
        let body = serde_json::json!({ "points": points });
        send_with_retry("qdrant upsert_points", || self.http.put(&url).json(&body)).await
    }

    /// Currently unused — kept for a future chunk-level invalidation
    /// path (drop specific chunk IDs without doing a `delete_by_file`).
    /// Remove once that lands or `delete_by_file` settles as the only
    /// deletion primitive.
    #[allow(dead_code)]
    pub async fn delete_points(&self, ids: &[String]) -> Result<()> {
        if ids.is_empty() {
            return Ok(());
        }
        let url = format!(
            "{}/collections/{}/points/delete?wait=true",
            self.base_url, self.collection
        );
        let body = serde_json::json!({ "points": ids });
        send_with_retry("qdrant delete_points", || self.http.post(&url).json(&body)).await
    }

    /// Delete all points where payload.file == file. Used when re-indexing a changed file
    /// or removing a file deleted from disk. Filter-based delete is atomic in Qdrant.
    pub async fn delete_by_file(&self, file: &str) -> Result<()> {
        let url = format!(
            "{}/collections/{}/points/delete?wait=true",
            self.base_url, self.collection
        );
        let body = serde_json::json!({
            "filter": {
                "must": [
                    { "key": "file", "match": { "value": file } }
                ]
            }
        });
        send_with_retry("qdrant delete_by_file", || self.http.post(&url).json(&body)).await
    }

    /// Cosine-similarity search against `vector`, returning up to `limit`
    /// hits with id, score, and payload. Optional payload filters can narrow
    /// by language (exact match) or by path prefix (Qdrant `match` keyword
    /// `text` is only available on payload-indexed fields, so for the
    /// prefix case we filter client-side after the fetch).
    pub async fn search(
        &self,
        vector: &[f32],
        limit: usize,
        lang_filter: Option<&str>,
    ) -> Result<Vec<DenseHit>> {
        let url = format!(
            "{}/collections/{}/points/search",
            self.base_url, self.collection
        );
        // Always exclude the marker point from search results via a
        // must_not filter on the dedicated `kind=marker` tag. Combined
        // with `lang` (if set) as an additional must.
        let mut must: Vec<serde_json::Value> = Vec::new();
        if let Some(lang) = lang_filter {
            must.push(serde_json::json!({
                "key": "lang", "match": { "value": lang }
            }));
        }
        let mut filter = serde_json::json!({
            "must_not": [
                { "key": "kind", "match": { "value": "marker" } }
            ]
        });
        if !must.is_empty() {
            filter["must"] = serde_json::Value::Array(must);
        }
        let body = serde_json::json!({
            "vector": vector,
            "limit": limit,
            "with_payload": true,
            "with_vector": false,
            "filter": filter,
        });
        let resp = self
            .http
            .post(&url)
            .json(&body)
            .send()
            .await
            .context("qdrant POST search")?
            .error_for_status()
            .context("qdrant search non-2xx")?;
        let parsed: SearchResp = resp.json().await.context("parsing search response")?;
        Ok(parsed
            .result
            .into_iter()
            .map(|p| DenseHit {
                chunk_id: p.id,
                score: p.score,
                file: p.payload.file.unwrap_or_default(),
                start_line: p.payload.start_line.unwrap_or(0),
                end_line: p.payload.end_line.unwrap_or(0),
                lang: p.payload.lang.unwrap_or_default(),
                snippet: p.payload.snippet.unwrap_or_default(),
                kind: p.payload.kind,
                name: p.payload.name,
            })
            .collect())
    }

    /// Compare the current config against the marker stored in the
    /// collection. Returns a [`MarkerStatus`] the caller dispatches on:
    /// `Fresh` → write marker, `Match` → no-op, `SoftChanged` → reindex
    /// only, `HardChanged` → clear + reindex. Project-identity mismatch
    /// (different `project_id` or `root`) is the ONE case that hard-fails
    /// here — it means we're pointing at someone else's collection, which
    /// requires manual intervention, not auto-clear.
    pub async fn check_marker_status(&self, config: &Config) -> Result<MarkerStatus> {
        let expected = MarkerPayload::for_config(config);
        let actual = match self.fetch_marker().await? {
            None => return Ok(MarkerStatus::Fresh),
            Some(a) => a,
        };
        if actual.fingerprint != expected.fingerprint {
            anyhow::bail!(
                "Qdrant collection '{}' belongs to a different project.\n  \
                 stored:  project_id='{}'  root='{}'  fingerprint={}\n  \
                 current: project_id='{}'  root='{}'  fingerprint={}\n\
                 Either pick a unique `vector_store.collection` name (or remove the field to \
                 auto-derive a safe one), point project.root at the original location, or run \
                 `clear --yes` to wipe the existing collection.",
                self.collection,
                actual.project_id,
                actual.root,
                &actual.fingerprint[..16.min(actual.fingerprint.len())],
                expected.project_id,
                expected.root,
                &expected.fingerprint[..16],
            );
        }
        // Project identity matches. Decide between Match / Soft / Hard.
        let hard_match = actual.config_hard.as_deref() == expected.config_hard.as_deref();
        let soft_match = actual.config_soft.as_deref() == expected.config_soft.as_deref();
        if !hard_match {
            Ok(MarkerStatus::HardChanged)
        } else if !soft_match {
            Ok(MarkerStatus::SoftChanged)
        } else {
            Ok(MarkerStatus::Match)
        }
    }

    /// Write the current config's marker (overwriting any existing one).
    /// Used after `Fresh`, `SoftChanged`, or post-`HardChanged` clear.
    pub async fn write_current_marker(&self, config: &Config, dimensions: usize) -> Result<()> {
        let payload = MarkerPayload::for_config(config);
        self.write_marker(&payload, dimensions).await
    }

    /// Read-side identity check used by `search` / `serve`. Doesn't write
    /// if the marker is absent (search is non-mutating). Fails on
    /// project-identity fingerprint mismatch. Hard-config mismatch also
    /// fails — search results would be stale because chunk identity
    /// differs between what's stored and what the current config would
    /// produce. Soft-config mismatch logs a warning but allows search.
    pub async fn verify_marker_read_only(&self, config: &Config) -> Result<()> {
        match self.check_marker_status(config).await? {
            MarkerStatus::Fresh => {
                tracing::debug!(
                    collection = %self.collection,
                    "no project-identity marker present (pre-marker collection or fresh)"
                );
                Ok(())
            }
            MarkerStatus::Match => {
                tracing::debug!(
                    collection = %self.collection,
                    "project-identity marker verified"
                );
                Ok(())
            }
            MarkerStatus::SoftChanged => {
                tracing::warn!(
                    collection = %self.collection,
                    "config-soft change since last index (languages / exclude / gitignore). \
                     Search results reflect the previous file set until you run `index`."
                );
                Ok(())
            }
            MarkerStatus::HardChanged => {
                anyhow::bail!(
                    "Config-hard mismatch (chunking strategy/params or embedding model/\
                     dimensions changed since the index was built). Stored chunks no longer \
                     align with what the current config would produce — search would return \
                     stale results. Run `index` (it will auto-clear and rebuild)."
                )
            }
        }
    }

    /// Read the project-identity marker point from this collection.
    /// Returns `None` if the marker doesn't exist (fresh collection or
    /// pre-marker indexer version), `Some(payload)` otherwise.
    pub async fn fetch_marker(&self) -> Result<Option<MarkerPayload>> {
        let url = format!(
            "{}/collections/{}/points/{}",
            self.base_url,
            self.collection,
            marker_point_id()
        );
        let resp = self
            .http
            .get(&url)
            .send()
            .await
            .context("qdrant GET marker")?;
        match resp.status().as_u16() {
            200 => {
                let parsed: MarkerFetchResp =
                    resp.json().await.context("parsing marker response")?;
                Ok(Some(parsed.result.payload))
            }
            404 => Ok(None),
            other => {
                let body = resp.text().await.unwrap_or_default();
                anyhow::bail!("qdrant marker fetch HTTP {}: {}", other, body.trim())
            }
        }
    }

    /// Write (upsert) the project-identity marker point. Uses a fixed
    /// deterministic UUID so the point is addressable by anyone who knows
    /// the format. Vector is a single zero entry — Qdrant requires the
    /// vector to be the same dimensionality as the collection, so we
    /// take that as input. The marker is filtered out of search results
    /// and the scroll-build cache by its payload `kind: "marker"` tag.
    pub async fn write_marker(&self, payload: &MarkerPayload, dimensions: usize) -> Result<()> {
        let url = format!(
            "{}/collections/{}/points?wait=true",
            self.base_url, self.collection
        );
        let vector = vec![0.0_f32; dimensions];
        let mut payload_json = serde_json::to_value(payload).context("serialize marker")?;
        // Tag so it can be filtered out of search/scroll.
        payload_json["kind"] = serde_json::Value::String("marker".to_string());
        let body = serde_json::json!({
            "points": [{
                "id": marker_point_id(),
                "vector": vector,
                "payload": payload_json,
            }]
        });
        self.http
            .put(&url)
            .json(&body)
            .send()
            .await
            .context("qdrant PUT marker")?
            .error_for_status()
            .context("qdrant marker upsert non-2xx")?;
        Ok(())
    }

    /// Bulk-retrieve only the `kind` + `name` payload fields for the given
    /// chunk IDs. Used to fill in AST metadata for sparse-only candidates
    /// (BM25 hit but not in dense top-K): tantivy's schema doesn't store
    /// the AST fields, so without this they'd appear in results without
    /// the syntactic anchor.
    pub async fn fetch_kind_name(
        &self,
        ids: &[String],
    ) -> Result<HashMap<String, (Option<String>, Option<String>)>> {
        if ids.is_empty() {
            return Ok(HashMap::new());
        }
        let url = format!("{}/collections/{}/points", self.base_url, self.collection);
        let body = serde_json::json!({
            "ids": ids,
            "with_payload": { "include": ["kind", "name"] },
            "with_vector": false,
        });
        let resp = self
            .http
            .post(&url)
            .json(&body)
            .send()
            .await
            .context("qdrant POST points (retrieve)")?
            .error_for_status()
            .context("qdrant retrieve non-2xx")?;
        let parsed: RetrieveResp = resp.json().await.context("parsing retrieve response")?;
        let mut out = HashMap::with_capacity(parsed.result.len());
        for p in parsed.result {
            out.insert(p.id, (p.payload.kind, p.payload.name));
        }
        Ok(out)
    }

    /// Scroll the entire collection once at startup to build the file→sha cache.
    /// This replaces the old index_state.json: Qdrant IS the source of truth for
    /// what has been indexed. A chunk's payload carries `file` + `file_sha256`,
    /// and we collapse all chunks of one file into a single (file, sha) entry.
    ///
    /// Points missing `file_sha256` (legacy data from before this refactor) are
    /// ignored — their files will be detected as "not indexed" and reprocessed.
    pub async fn scroll_files(&self) -> Result<HashMap<PathBuf, String>> {
        let mut cache: HashMap<PathBuf, String> = HashMap::new();
        let url = format!(
            "{}/collections/{}/points/scroll",
            self.base_url, self.collection
        );
        let mut next_offset: Option<serde_json::Value> = None;
        loop {
            let body = serde_json::json!({
                "limit": 1024,
                "with_payload": { "include": ["file", "file_sha256"] },
                "with_vector": false,
                "offset": next_offset,
            });
            let resp = self
                .http
                .post(&url)
                .json(&body)
                .send()
                .await
                .context("qdrant POST scroll")?
                .error_for_status()
                .context("qdrant scroll non-2xx")?;
            let parsed: ScrollResp = resp.json().await.context("parsing scroll response")?;
            for p in parsed.result.points {
                let (file, sha) = match (p.payload.file, p.payload.file_sha256) {
                    (Some(f), Some(s)) => (f, s),
                    _ => continue, // skip marker point (no file/sha) and any legacy data
                };
                cache.entry(PathBuf::from(file)).or_insert(sha);
            }
            match parsed.result.next_page_offset {
                Some(v) if !v.is_null() => next_offset = Some(v),
                _ => break,
            }
        }
        Ok(cache)
    }
}

#[derive(Deserialize)]
struct ScrollResp {
    result: ScrollResultInner,
}

#[derive(Deserialize)]
struct ScrollResultInner {
    points: Vec<ScrollPoint>,
    #[serde(default)]
    next_page_offset: Option<serde_json::Value>,
}

#[derive(Deserialize)]
struct ScrollPoint {
    payload: ScrollPayload,
}

#[derive(Deserialize, Default)]
struct ScrollPayload {
    #[serde(default)]
    file: Option<String>,
    #[serde(default)]
    file_sha256: Option<String>,
}

#[derive(Deserialize)]
struct CollectionInfoResp {
    result: CollectionResultInner,
}

#[derive(Deserialize)]
struct CollectionResultInner {
    points_count: Option<u64>,
}

/// One result from a dense (vector) search. The snippet here is the short
/// preview stored in the Qdrant payload at index time; for reranking we
/// upgrade to the full chunk content via the tantivy lookup.
#[derive(Debug)]
pub struct DenseHit {
    pub chunk_id: String,
    pub score: f32,
    pub file: String,
    pub start_line: u64,
    pub end_line: u64,
    pub lang: String,
    pub snippet: String,
    /// Syntactic kind (`"fn"`, `"struct"`, `"impl_method"`, etc.) — present
    /// for AST-chunked sources, `None` for line-window / heading chunks.
    pub kind: Option<String>,
    /// Symbol name (e.g. `"Foo::bar"`). See `kind` for who populates it.
    pub name: Option<String>,
}

#[derive(Deserialize)]
struct SearchResp {
    result: Vec<SearchPoint>,
}

#[derive(Deserialize)]
struct SearchPoint {
    id: String,
    score: f32,
    #[serde(default)]
    payload: SearchPayload,
}

#[derive(Deserialize)]
struct RetrieveResp {
    result: Vec<RetrievePoint>,
}

#[derive(Deserialize)]
struct RetrievePoint {
    id: String,
    #[serde(default)]
    payload: KindNamePayload,
}

#[derive(Deserialize, Default)]
struct KindNamePayload {
    #[serde(default)]
    kind: Option<String>,
    #[serde(default)]
    name: Option<String>,
}

#[derive(Deserialize, Default)]
struct SearchPayload {
    #[serde(default)]
    file: Option<String>,
    #[serde(default)]
    start_line: Option<u64>,
    #[serde(default)]
    end_line: Option<u64>,
    #[serde(default)]
    lang: Option<String>,
    #[serde(default)]
    snippet: Option<String>,
    #[serde(default)]
    kind: Option<String>,
    #[serde(default)]
    name: Option<String>,
}

/// Retry an HTTP request on transient errors (connection drops, 5xx).
/// `build` is called fresh on each attempt because `send()` consumes the
/// RequestBuilder. The closure captures `&self.http`, `&url`, `&body` by
/// reference — no allocation per retry.
///
/// Mirrors `embedding.rs`'s retry logic for the same class of failures
/// (keep-alive race: `connection closed before message completed` etc.).
/// Backoff: 500ms, 1s, 2s for attempts 1..=3; 4th attempt has no backoff
/// and bubbles up on failure.
async fn send_with_retry<F>(label: &'static str, mut build: F) -> Result<()>
where
    F: FnMut() -> reqwest::RequestBuilder,
{
    const MAX_ATTEMPTS: u32 = 4;
    let mut last_err: Option<anyhow::Error> = None;
    for attempt in 1..=MAX_ATTEMPTS {
        let res: Result<()> = async {
            let resp = build().send().await.with_context(|| label.to_string())?;
            let status = resp.status();
            if !status.is_success() {
                // Capture body — Qdrant returns useful JSON diagnostics on errors.
                let body = resp.text().await.unwrap_or_default();
                anyhow::bail!("{} HTTP {}: {}", label, status.as_u16(), body.trim());
            }
            Ok(())
        }
        .await;

        match res {
            Ok(()) => {
                if attempt > 1 {
                    tracing::info!(label, attempt, "qdrant request succeeded after retry");
                }
                return Ok(());
            }
            Err(e) => {
                if attempt < MAX_ATTEMPTS && is_transient_http(&e) {
                    let backoff = Duration::from_millis(500u64 << (attempt - 1));
                    tracing::warn!(
                        label,
                        attempt,
                        max = MAX_ATTEMPTS,
                        backoff_ms = backoff.as_millis() as u64,
                        error = %short_err(&e),
                        "qdrant request failed (transient), retrying"
                    );
                    tokio::time::sleep(backoff).await;
                    last_err = Some(e);
                    continue;
                }
                return Err(e);
            }
        }
    }
    Err(last_err.unwrap_or_else(|| anyhow::anyhow!("qdrant retries exhausted")))
}

/// Heuristic: HTTP/connection-layer errors worth retrying for Qdrant.
/// 4xx and parsing errors are NOT transient — they fail identically on retry.
fn is_transient_http(err: &anyhow::Error) -> bool {
    let chain = err
        .chain()
        .map(|e| e.to_string())
        .collect::<Vec<_>>()
        .join(" | ")
        .to_lowercase();
    chain.contains("connection closed")
        || chain.contains("connection reset")
        || chain.contains("broken pipe")
        || chain.contains("operation timed out")
        || chain.contains("timeout")
        || chain.contains("http 500")
        || chain.contains("http 502")
        || chain.contains("http 503")
        || chain.contains("http 504")
}

fn short_err(e: &anyhow::Error) -> String {
    e.chain()
        .map(|x| x.to_string())
        .collect::<Vec<_>>()
        .join(" | ")
        .chars()
        .take(160)
        .collect()
}
