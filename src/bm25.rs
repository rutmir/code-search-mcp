use anyhow::{Context, Result};
use std::collections::{HashMap, HashSet};
use std::path::{Path, PathBuf};
use tantivy::{
    collector::TopDocs,
    query::{BooleanQuery, Occur, Query, QueryParser, TermQuery, TermSetQuery},
    schema::{
        Field, IndexRecordOption, Schema, TextFieldIndexing, TextOptions, Value, INDEXED, STORED,
        STRING,
    },
    tokenizer::{Token, TokenStream, Tokenizer},
    Index, IndexReader, IndexWriter, TantivyDocument, Term,
};

/// Bump this whenever the tantivy schema or the content tokenizer changes
/// in a way that makes existing indexes stale. It feeds the marker's
/// `config_hard` fingerprint, so the change triggers the auto-clear +
/// full-rebuild flow instead of silently searching a half-compatible index.
/// v2: code-aware content tokenizer (snake_case / camelCase splitting).
/// v3: AST metadata (`kind` / `name`) stored, `name` additionally indexed
///     and searched with a boost.
pub const SCHEMA_VERSION: u32 = 3;

const CODE_TOKENIZER_NAME: &str = "code";

/// Weight of the `name` field relative to `content` in the BM25 query.
/// A chunk whose symbol name matches the query is a much stronger signal
/// than the same tokens appearing somewhere in a body — but not a
/// certainty, so this stays a boost rather than a filter. The exact-match
/// case is handled separately by `search::query_names_symbol`.
const NAME_FIELD_BOOST: tantivy::Score = 3.0;

/// Tantivy BM25 index — sparse retrieval layer for hybrid search.
///
/// Schema:
///   file       — STRING + STORED (exact match on path, e.g. for `path_prefix` filter)
///   chunk_id   — STRING + STORED (used as delete key)
///   start_line — u64  + STORED + INDEXED
///   end_line   — u64  + STORED + INDEXED
///   lang       — STRING + STORED (exact match, drives the `lang` filter)
///   kind       — STORED only (AST node kind; carried for display, never queried)
///   name       — STORED + indexed with the code-aware tokenizer, searched
///                alongside `content` at [`NAME_FIELD_BOOST`]
///   content    — STORED + indexed with the code-aware tokenizer (primary
///                BM25 search target; see [`CodeTokenizer`])
pub struct Bm25Index {
    pub writer: IndexWriter,
    pub fields: SchemaFields,
    pub index: Index,
}

pub struct SchemaFields {
    pub file: Field,
    pub chunk_id: Field,
    pub start_line: Field,
    pub end_line: Field,
    pub lang: Field,
    pub kind: Field,
    pub name: Field,
    pub content: Field,
}

/// One chunk as written to the BM25 index. A struct rather than a long
/// positional argument list so the AST metadata can't be swapped with the
/// path fields at a call site.
pub struct ChunkDoc<'a> {
    pub file: &'a str,
    pub chunk_id: &'a str,
    pub start_line: u64,
    pub end_line: u64,
    pub lang: &'a str,
    pub kind: Option<&'a str>,
    pub name: Option<&'a str>,
    pub content: &'a str,
}

impl Bm25Index {
    pub fn open(path: &Path) -> Result<Self> {
        std::fs::create_dir_all(path)
            .with_context(|| format!("creating tantivy dir: {}", path.display()))?;

        let (schema, fields) = build_schema();

        let index = match Index::open_in_dir(path) {
            Ok(idx) => idx,
            Err(_) => Index::create_in_dir(path, schema.clone())
                .with_context(|| format!("creating tantivy index in {}", path.display()))?,
        };
        register_code_tokenizer(&index);

        let writer: IndexWriter = index
            .writer(50_000_000) // 50 MB heap
            .context("creating tantivy IndexWriter")?;

        Ok(Self {
            writer,
            fields,
            index,
        })
    }

    pub fn upsert(&mut self, doc: &ChunkDoc<'_>) -> Result<()> {
        // delete-then-insert (tantivy has no native upsert)
        self.writer
            .delete_term(Term::from_field_text(self.fields.chunk_id, doc.chunk_id));

        let mut d = TantivyDocument::new();
        d.add_text(self.fields.file, doc.file);
        d.add_text(self.fields.chunk_id, doc.chunk_id);
        d.add_u64(self.fields.start_line, doc.start_line);
        d.add_u64(self.fields.end_line, doc.end_line);
        d.add_text(self.fields.lang, doc.lang);
        // kind/name are absent for line-window and heading chunks; an
        // omitted field simply has no value in tantivy.
        if let Some(kind) = doc.kind {
            d.add_text(self.fields.kind, kind);
        }
        if let Some(name) = doc.name {
            d.add_text(self.fields.name, name);
        }
        d.add_text(self.fields.content, doc.content);
        self.writer
            .add_document(d)
            .context("adding doc to tantivy")?;
        Ok(())
    }

    /// Currently unused — kept for a future search-time chunk-id
    /// invalidation path (when we know specific stale chunk IDs to drop
    /// without redoing a `delete_by_file`). Remove once that lands or if
    /// we settle on `delete_by_file` as the only deletion primitive.
    #[allow(dead_code)]
    pub fn delete_chunks(&mut self, chunk_ids: &[String]) {
        for id in chunk_ids {
            self.writer
                .delete_term(Term::from_field_text(self.fields.chunk_id, id));
        }
    }

    /// Delete every document whose `file` field equals the given path.
    /// Used by the indexer to wipe a file's BM25 footprint before reindexing.
    pub fn delete_by_file(&mut self, file: &str) {
        self.writer
            .delete_term(Term::from_field_text(self.fields.file, file));
    }

    pub fn commit(&mut self) -> Result<()> {
        self.writer.commit().context("tantivy commit")?;
        Ok(())
    }

    /// Enumerate the distinct `file` field values currently in the
    /// tantivy index. Used by the indexer to detect "Qdrant has this
    /// file but tantivy doesn't" (which happens after moving to a new
    /// machine that shares Qdrant but has an empty local tantivy
    /// directory) so we can reprocess those files and refill tantivy
    /// without trusting the otherwise-misleading cache.
    pub fn list_indexed_files(&self) -> Result<HashSet<PathBuf>> {
        let reader = self
            .index
            .reader()
            .context("creating tantivy reader for file enumeration")?;
        list_files(&reader, self.fields.file)
    }
}

/// Distinct `file` terms across every segment. Shared by the writer-side
/// [`Bm25Index`] and the read-only [`Bm25Search`] so `status` can inspect
/// the index without taking the directory's write lock (which a running
/// `serve` or `watch` already holds).
fn list_files(reader: &IndexReader, file_field: Field) -> Result<HashSet<PathBuf>> {
    let searcher = reader.searcher();
    let mut out: HashSet<PathBuf> = HashSet::new();
    for seg in searcher.segment_readers() {
        let inv = seg
            .inverted_index(file_field)
            .context("inverted_index(file)")?;
        let mut terms = inv.terms().stream().context("terms stream")?;
        while let Some((term_bytes, _info)) = terms.next() {
            if let Ok(s) = std::str::from_utf8(term_bytes) {
                out.insert(PathBuf::from(s));
            }
        }
    }
    Ok(out)
}

fn build_schema() -> (Schema, SchemaFields) {
    let mut b = Schema::builder();
    let file = b.add_text_field("file", STRING | STORED);
    let chunk_id = b.add_text_field("chunk_id", STRING | STORED);
    let start_line = b.add_u64_field("start_line", STORED | INDEXED);
    let end_line = b.add_u64_field("end_line", STORED | INDEXED);
    let lang = b.add_text_field("lang", STRING | STORED);
    // Node kind ("fn", "struct", …) is display metadata only — storing it
    // without indexing keeps the term dictionary free of a handful of
    // ultra-high-frequency terms that would never make a useful query.
    let kind = b.add_text_field("kind", STORED);
    let text_opts = TextOptions::default().set_stored().set_indexing_options(
        TextFieldIndexing::default()
            .set_tokenizer(CODE_TOKENIZER_NAME)
            .set_index_option(IndexRecordOption::WithFreqsAndPositions),
    );
    let name = b.add_text_field("name", text_opts.clone());
    let content = b.add_text_field("content", text_opts);
    let schema = b.build();
    (
        schema,
        SchemaFields {
            file,
            chunk_id,
            start_line,
            end_line,
            lang,
            kind,
            name,
            content,
        },
    )
}

/// The tokenizer manager lives on the `Index` handle, not on disk — every
/// place that opens the index (writer or read-only searcher) must register
/// the code tokenizer or tantivy fails with "tokenizer not found".
fn register_code_tokenizer(index: &Index) {
    index
        .tokenizers()
        .register(CODE_TOKENIZER_NAME, CodeTokenizer);
}

/// Code-aware tokenizer: like tantivy's simple tokenizer (split on
/// non-alphanumeric, lowercase), but identifiers are additionally split on
/// `_` and camelCase humps — `build_si_portfolio` and `buildSiPortfolio`
/// both index as [`build`, `si`, `portfolio`], so a query naming either
/// form (or just `portfolio`) matches. Acronym runs keep their boundary
/// (`HTTPServer` → [`http`, `server`]); digits stay attached to the
/// preceding letters (`bm25` stays whole).
///
/// Only the split parts are emitted (no compound token): tantivy's
/// QueryParser turns a multi-token word into a phrase query, so the exact
/// identifier `build_si_portfolio` in a query still matches precisely —
/// as the consecutive phrase [`build`, `si`, `portfolio`] — while an
/// extra compound token would break that phrase's position sequence for
/// cross-style (camel ↔ snake) lookups.
#[derive(Clone)]
struct CodeTokenizer;

impl Tokenizer for CodeTokenizer {
    type TokenStream<'a> = VecTokenStream;

    fn token_stream<'a>(&'a mut self, text: &'a str) -> VecTokenStream {
        /// Parts longer than this are hash-/base64-like noise, not words.
        const MAX_TOKEN_BYTES: usize = 40;
        let mut tokens = Vec::new();
        let mut position = 0usize;

        // Runs of identifier chars ([alnum]+ including `_`).
        let mut runs: Vec<(usize, usize)> = Vec::new();
        let mut start: Option<usize> = None;
        for (bi, c) in text.char_indices() {
            let ident = c.is_alphanumeric() || c == '_';
            match (start, ident) {
                (None, true) => start = Some(bi),
                (Some(s), false) => {
                    runs.push((s, bi));
                    start = None;
                }
                _ => {}
            }
        }
        if let Some(s) = start {
            runs.push((s, text.len()));
        }

        for (rs, re) in runs {
            for (ps, pe) in split_identifier(&text[rs..re]) {
                let (from, to) = (rs + ps, rs + pe);
                if to - from > MAX_TOKEN_BYTES {
                    continue;
                }
                tokens.push(Token {
                    offset_from: from,
                    offset_to: to,
                    position,
                    text: text[from..to].to_lowercase(),
                    position_length: 1,
                });
                position += 1;
            }
        }
        VecTokenStream { tokens, idx: 0 }
    }
}

/// Split one identifier run into sub-word byte spans (relative to the run):
/// at `_` (dropped), at lower/digit→Upper boundaries, and at the last
/// capital of an acronym run when followed by lowercase.
fn split_identifier(word: &str) -> Vec<(usize, usize)> {
    let chars: Vec<(usize, char)> = word.char_indices().collect();
    let mut spans = Vec::new();
    let mut start: Option<usize> = None;
    for i in 0..chars.len() {
        let (bi, c) = chars[i];
        if c == '_' {
            if let Some(s) = start.take() {
                spans.push((s, bi));
            }
            continue;
        }
        let Some(s) = start else {
            start = Some(bi);
            continue;
        };
        let prev = chars[i - 1].1;
        let camel_boundary = (prev.is_lowercase() || prev.is_numeric()) && c.is_uppercase();
        let acronym_boundary = prev.is_uppercase()
            && c.is_uppercase()
            && chars.get(i + 1).is_some_and(|&(_, n)| n.is_lowercase());
        if camel_boundary || acronym_boundary {
            spans.push((s, bi));
            start = Some(bi);
        }
    }
    if let Some(s) = start {
        spans.push((s, word.len()));
    }
    spans
}

/// Pre-materialized token stream — the token set is computed eagerly in
/// `token_stream` (identifier splitting needs lookahead anyway).
struct VecTokenStream {
    tokens: Vec<Token>,
    idx: usize,
}

impl TokenStream for VecTokenStream {
    fn advance(&mut self) -> bool {
        if self.idx < self.tokens.len() {
            self.idx += 1;
            true
        } else {
            false
        }
    }
    fn token(&self) -> &Token {
        &self.tokens[self.idx - 1]
    }
    fn token_mut(&mut self) -> &mut Token {
        &mut self.tokens[self.idx - 1]
    }
}

// Suppress dead-code warnings for index/field accessors used only by `Bm25Search`.
#[allow(dead_code)]
impl Bm25Index {
    pub fn index(&self) -> &Index {
        &self.index
    }
    pub fn fields(&self) -> &SchemaFields {
        &self.fields
    }
}

/// One BM25 (sparse) search hit. `content` carries the full chunk text
/// (the `STORED` field), suitable for feeding into a reranker.
#[derive(Debug, Clone)]
pub struct SparseHit {
    pub chunk_id: String,
    pub score: f32,
    pub file: String,
    pub start_line: u64,
    pub end_line: u64,
    pub lang: String,
    pub kind: Option<String>,
    pub name: Option<String>,
    pub content: String,
}

/// Full text of one chunk, addressed by location rather than by chunk_id —
/// what the `code_read_chunk` MCP tool hands back.
#[derive(Debug, Clone)]
pub struct ChunkText {
    pub file: String,
    pub start_line: u64,
    pub end_line: u64,
    pub lang: String,
    pub kind: Option<String>,
    pub name: Option<String>,
    pub content: String,
}

/// Read-only view of the BM25 index. Unlike [`Bm25Index`], this does not
/// allocate an [`IndexWriter`], so it doesn't take a write lock on the
/// directory — multiple `search` invocations can coexist with the indexer.
///
/// The [`IndexReader`] is created once and held: it carries tantivy's
/// default `OnCommitWithDelay` reload policy, so a long-lived `serve`
/// process still picks up the background watcher's commits, while queries
/// stop paying for a reader (and its directory watch registration) each
/// time. `searcher()` per query is the cheap part.
pub struct Bm25Search {
    index: Index,
    reader: IndexReader,
    fields: SchemaFields,
}

impl Bm25Search {
    pub fn open(path: &Path) -> Result<Self> {
        let (_schema, fields) = build_schema();
        let index = Index::open_in_dir(path)
            .with_context(|| format!("opening tantivy index for search in {}", path.display()))?;
        register_code_tokenizer(&index);
        let reader = index.reader().context("creating tantivy reader")?;
        Ok(Self {
            index,
            reader,
            fields,
        })
    }

    /// BM25-rank documents by relevance to `query`, optionally restricted
    /// to one language.
    ///
    /// Two defenses against QueryParser syntax in natural-language queries:
    /// `:` is replaced by a space before parsing (`field:value` syntax is
    /// never useful here — but `Watcher::run` would otherwise parse as a
    /// clause on a nonexistent field and be dropped), and the rest is
    /// parsed leniently so stray quotes / `+` / `-` can't fail the whole
    /// search.
    ///
    /// The `lang` restriction is a hard `Must` clause rather than a
    /// post-filter: filtering after the fact would silently shrink the
    /// candidate pool the caller asked for.
    pub fn search(&self, query: &str, limit: usize, lang: Option<&str>) -> Result<Vec<SparseHit>> {
        let searcher = self.reader.searcher();
        let mut parser =
            QueryParser::for_index(&self.index, vec![self.fields.content, self.fields.name]);
        parser.set_field_boost(self.fields.name, NAME_FIELD_BOOST);
        let sanitized = query.replace(':', " ");
        let (parsed, parse_errors) = parser.parse_query_lenient(&sanitized);
        if !parse_errors.is_empty() {
            tracing::debug!(
                query = %query,
                errors = ?parse_errors,
                "bm25 query parsed leniently (some clauses dropped)"
            );
        }
        let q: Box<dyn Query> = match lang {
            Some(lang) => Box::new(BooleanQuery::new(vec![
                (Occur::Must, parsed),
                (
                    Occur::Must,
                    Box::new(TermQuery::new(
                        Term::from_field_text(self.fields.lang, lang),
                        IndexRecordOption::Basic,
                    )),
                ),
            ])),
            None => parsed,
        };
        let top = searcher
            .search(&q, &TopDocs::with_limit(limit))
            .context("running bm25 search")?;
        let mut hits = Vec::with_capacity(top.len());
        for (score, addr) in top {
            let doc: TantivyDocument = searcher.doc(addr).context("fetching bm25 doc")?;
            hits.push(SparseHit {
                chunk_id: extract_str(&doc, self.fields.chunk_id),
                score,
                file: extract_str(&doc, self.fields.file),
                start_line: extract_u64(&doc, self.fields.start_line),
                end_line: extract_u64(&doc, self.fields.end_line),
                lang: extract_str(&doc, self.fields.lang),
                kind: extract_opt_str(&doc, self.fields.kind),
                name: extract_opt_str(&doc, self.fields.name),
                content: extract_str(&doc, self.fields.content),
            });
        }
        Ok(hits)
    }

    /// Look up the full content of many chunks at once, keyed by chunk_id.
    /// Used to upgrade dense-only candidates (which carry only the 200-char
    /// snippet from the Qdrant payload) to full text before reranking.
    ///
    /// One `TermSetQuery` instead of a query per id: the rerank head is
    /// `rerank_top_n` candidates (30 by default), and a term-per-id round
    /// trip through the collector was the single most repeated piece of
    /// work in a search.
    pub fn lookup_contents(&self, chunk_ids: &[String]) -> Result<HashMap<String, String>> {
        if chunk_ids.is_empty() {
            return Ok(HashMap::new());
        }
        let searcher = self.reader.searcher();
        let terms = chunk_ids
            .iter()
            .map(|id| Term::from_field_text(self.fields.chunk_id, id));
        let query = TermSetQuery::new(terms);
        let top = searcher
            .search(&query, &TopDocs::with_limit(chunk_ids.len()))
            .context("chunk_id set query")?;
        let mut out = HashMap::with_capacity(top.len());
        for (_, addr) in top {
            let doc: TantivyDocument = searcher.doc(addr).context("fetching chunk doc")?;
            let id = extract_str(&doc, self.fields.chunk_id);
            if id.is_empty() {
                continue;
            }
            out.insert(id, extract_str(&doc, self.fields.content));
        }
        Ok(out)
    }

    /// Distinct file paths present in the index — the read-only twin of
    /// [`Bm25Index::list_indexed_files`], for inspecting an index that a
    /// running `serve` or `watch` holds the write lock on.
    pub fn list_indexed_files(&self) -> Result<HashSet<PathBuf>> {
        list_files(&self.reader, self.fields.file)
    }

    /// Fetch every indexed chunk of `file` that overlaps the inclusive line
    /// range `[start, end]`, ordered by start line. Backs the
    /// `code_read_chunk` tool: the caller has a `file:start-end` header
    /// from a previous search result and wants the untruncated text.
    ///
    /// A file's chunk count is bounded (large files are split, not
    /// unbounded), so collecting all of its docs and filtering in memory
    /// is cheaper than composing a range query per call.
    pub fn chunks_in_range(&self, file: &str, start: u64, end: u64) -> Result<Vec<ChunkText>> {
        let searcher = self.reader.searcher();
        let query = TermQuery::new(
            Term::from_field_text(self.fields.file, file),
            IndexRecordOption::Basic,
        );
        // A single file's chunks; the cap is a safety valve for pathological
        // generated files, not an expected limit.
        let top = searcher
            .search(&query, &TopDocs::with_limit(10_000))
            .context("file term query")?;
        let mut out = Vec::new();
        for (_, addr) in top {
            let doc: TantivyDocument = searcher.doc(addr).context("fetching chunk doc")?;
            let chunk_start = extract_u64(&doc, self.fields.start_line);
            let chunk_end = extract_u64(&doc, self.fields.end_line);
            if chunk_end < start || chunk_start > end {
                continue;
            }
            out.push(ChunkText {
                file: extract_str(&doc, self.fields.file),
                start_line: chunk_start,
                end_line: chunk_end,
                lang: extract_str(&doc, self.fields.lang),
                kind: extract_opt_str(&doc, self.fields.kind),
                name: extract_opt_str(&doc, self.fields.name),
                content: extract_str(&doc, self.fields.content),
            });
        }
        out.sort_by_key(|c| (c.start_line, c.end_line));
        Ok(out)
    }
}

fn extract_str(doc: &TantivyDocument, field: Field) -> String {
    doc.get_first(field)
        .and_then(|v| v.as_str())
        .map(|s| s.to_string())
        .unwrap_or_default()
}

/// Like [`extract_str`] but distinguishes "field absent" from "empty
/// string" — kind/name are genuinely optional (line-window and heading
/// chunks have neither).
fn extract_opt_str(doc: &TantivyDocument, field: Field) -> Option<String> {
    doc.get_first(field)
        .and_then(|v| v.as_str())
        .map(|s| s.to_string())
}

fn extract_u64(doc: &TantivyDocument, field: Field) -> u64 {
    doc.get_first(field).and_then(|v| v.as_u64()).unwrap_or(0)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tokens(text: &str) -> Vec<String> {
        let mut t = CodeTokenizer;
        let mut stream = t.token_stream(text);
        let mut out = Vec::new();
        while stream.advance() {
            out.push(stream.token().text.clone());
        }
        out
    }

    #[test]
    fn tok_snake_case_splits() {
        assert_eq!(
            tokens("build_si_portfolio"),
            vec!["build", "si", "portfolio"]
        );
    }

    #[test]
    fn tok_camel_case_splits() {
        assert_eq!(tokens("buildSiPortfolio"), vec!["build", "si", "portfolio"]);
        assert_eq!(tokens("AdaptiveBatcher"), vec!["adaptive", "batcher"]);
    }

    #[test]
    fn tok_acronym_boundary() {
        assert_eq!(tokens("HTTPServer"), vec!["http", "server"]);
        // Trailing acronym stays whole.
        assert_eq!(tokens("parseJSON"), vec!["parse", "json"]);
    }

    #[test]
    fn tok_digits_stay_attached() {
        assert_eq!(tokens("bm25"), vec!["bm25"]);
        assert_eq!(tokens("Server2"), vec!["server2"]);
    }

    #[test]
    fn tok_prose_unchanged() {
        assert_eq!(
            tokens("adaptive batching, halving on failure!"),
            vec!["adaptive", "batching", "halving", "on", "failure"]
        );
    }

    #[test]
    fn tok_qualified_path() {
        // `::` is a separator (not an identifier char), so qualified names
        // split into their segments' parts.
        assert_eq!(
            tokens("AdaptiveBatcher::note_failure"),
            vec!["adaptive", "batcher", "note", "failure"]
        );
    }

    #[test]
    fn tok_underscore_runs_and_empties() {
        assert_eq!(tokens("__init__"), vec!["init"]);
        assert_eq!(tokens("____"), Vec::<String>::new());
    }

    /// The tokenizer indexes whatever bytes a repo happens to contain —
    /// minified bundles, fixtures full of emoji, combining marks in
    /// comments. It slices `text` by byte offsets computed from
    /// `char_indices`, so a boundary mistake is a panic in the indexer,
    /// not a bad search result. Sweep a pool of adversarial characters and
    /// assert the invariants every emitted token must satisfy.
    #[test]
    fn tok_never_panics_and_emits_valid_offsets() {
        const PIECES: &[&str] = &[
            "_",
            "A",
            "a",
            "9",
            "Ж",
            "ß",
            "İ",
            "ﬀ",
            "🚀",
            "👩‍💻",
            "é",
            "e\u{0301}",
            "　",
            "\u{200b}",
            "::",
            ".",
            "-",
            "\n",
            "\t",
            "\u{feff}",
            "ｆｕｌｌ",
            "𝔘",
            "ᾈ",
        ];
        // Deterministic LCG — a fixed sweep beats a flaky random one.
        let mut seed: u64 = 0x5eed_1234_dead_beef;
        let mut next = move || {
            seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1);
            (seed >> 33) as usize
        };
        for _ in 0..2_000 {
            let len = next() % 12;
            let text: String = (0..len).map(|_| PIECES[next() % PIECES.len()]).collect();

            let mut t = CodeTokenizer;
            let mut stream = t.token_stream(&text);
            let mut prev_position = None;
            while stream.advance() {
                let tok = stream.token();
                assert!(
                    tok.offset_from < tok.offset_to,
                    "empty span {}..{} in {text:?}",
                    tok.offset_from,
                    tok.offset_to
                );
                assert!(
                    tok.offset_to <= text.len(),
                    "span past end in {text:?}: {}..{}",
                    tok.offset_from,
                    tok.offset_to
                );
                // Panics unless both offsets land on char boundaries.
                let slice = &text[tok.offset_from..tok.offset_to];
                assert_eq!(tok.text, slice.to_lowercase());
                if let Some(prev) = prev_position {
                    assert!(tok.position > prev, "positions must strictly increase");
                }
                prev_position = Some(tok.position);
            }
        }
    }

    #[test]
    fn tok_cyrillic_lowercased() {
        assert_eq!(tokens("Адаптивный Батчер"), vec!["адаптивный", "батчер"]);
    }

    /// End-to-end through a real tantivy index: snake_case-indexed content
    /// is found by camelCase, by a sub-word, and by the exact identifier;
    /// QueryParser-syntax queries (`::`) don't error.
    #[test]
    fn e2e_cross_style_identifier_search() {
        let dir = std::env::temp_dir().join(format!(
            "code-search-mcp-bm25-test-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        std::fs::create_dir_all(&dir).unwrap();

        let mut idx = Bm25Index::open(&dir).unwrap();
        idx.upsert(&ChunkDoc {
            file: "src/portfolio.rs",
            chunk_id: "chunk-1",
            start_line: 10,
            end_line: 40,
            lang: "rust",
            kind: Some("fn"),
            name: Some("build_si_portfolio"),
            content: "pub fn build_si_portfolio(deposit: Decimal) -> Engine { }",
        })
        .unwrap();
        idx.upsert(&ChunkDoc {
            file: "src/other.rs",
            chunk_id: "chunk-2",
            start_line: 1,
            end_line: 5,
            lang: "rust",
            kind: Some("fn"),
            name: Some("unrelated_helper"),
            content: "fn unrelated_helper() {}",
        })
        .unwrap();
        idx.upsert(&ChunkDoc {
            file: "docs/guide.md",
            chunk_id: "chunk-3",
            start_line: 1,
            end_line: 9,
            lang: "markdown",
            kind: None,
            name: None,
            content: "## Portfolio sizing\nHow build_si_portfolio decides deposit split.",
        })
        .unwrap();
        idx.commit().unwrap();
        drop(idx);

        let search = Bm25Search::open(&dir).unwrap();
        for query in [
            "build_si_portfolio",
            "buildSiPortfolio",
            "portfolio sizing",
            "Portfolio::build_si_portfolio",
        ] {
            let hits = search.search(query, 5, None).unwrap();
            assert!(
                hits.iter().any(|h| h.chunk_id == "chunk-1"),
                "query {:?} should find chunk-1, got {:?}",
                query,
                hits.iter().map(|h| h.chunk_id.clone()).collect::<Vec<_>>()
            );
        }
        // Garbage QueryParser syntax must not error out.
        let hits = search.search("\"unbalanced AND build_si_portfolio", 5, None);
        assert!(hits.is_ok());

        // AST metadata round-trips through the index.
        let hits = search.search("build_si_portfolio", 5, None).unwrap();
        let top = hits.iter().find(|h| h.chunk_id == "chunk-1").unwrap();
        assert_eq!(top.kind.as_deref(), Some("fn"));
        assert_eq!(top.name.as_deref(), Some("build_si_portfolio"));

        // lang filter is a hard clause: markdown-only search never returns
        // the Rust chunks even though they match the query better.
        let hits = search
            .search("build_si_portfolio portfolio", 5, Some("markdown"))
            .unwrap();
        assert!(!hits.is_empty(), "markdown chunk should still match");
        assert!(
            hits.iter().all(|h| h.lang == "markdown"),
            "lang filter leaked non-markdown hits: {:?}",
            hits.iter().map(|h| h.lang.clone()).collect::<Vec<_>>()
        );

        // Batch content lookup returns every requested id it knows about,
        // and silently omits the ones it doesn't.
        let map = search
            .lookup_contents(&[
                "chunk-1".to_string(),
                "chunk-2".to_string(),
                "no-such-chunk".to_string(),
            ])
            .unwrap();
        assert_eq!(map.len(), 2);
        assert!(map["chunk-1"].contains("build_si_portfolio"));
        assert!(search.lookup_contents(&[]).unwrap().is_empty());

        // Range lookup returns overlapping chunks only.
        let got = search.chunks_in_range("src/portfolio.rs", 10, 40).unwrap();
        assert_eq!(got.len(), 1);
        assert_eq!(got[0].name.as_deref(), Some("build_si_portfolio"));
        assert!(search
            .chunks_in_range("src/portfolio.rs", 100, 200)
            .unwrap()
            .is_empty());

        std::fs::remove_dir_all(&dir).ok();
    }
}
