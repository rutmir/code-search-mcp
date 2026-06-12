use anyhow::{Context, Result};
use std::path::Path;
use tantivy::{
    collector::TopDocs,
    query::{QueryParser, TermQuery},
    schema::{
        Field, IndexRecordOption, Schema, TextFieldIndexing, TextOptions, Value, INDEXED, STORED,
        STRING,
    },
    tokenizer::{Token, TokenStream, Tokenizer},
    Index, IndexWriter, TantivyDocument, Term,
};

/// Bump this whenever the tantivy schema or the content tokenizer changes
/// in a way that makes existing indexes stale. It feeds the marker's
/// `config_hard` fingerprint, so the change triggers the auto-clear +
/// full-rebuild flow instead of silently searching a half-compatible index.
/// v2: code-aware content tokenizer (snake_case / camelCase splitting).
pub const SCHEMA_VERSION: u32 = 2;

const CODE_TOKENIZER_NAME: &str = "code";

/// Tantivy BM25 index — sparse retrieval layer for hybrid search.
///
/// Schema:
///   file       — STRING + STORED (exact match on path, e.g. for `path_prefix` filter)
///   chunk_id   — STRING + STORED (used as delete key)
///   start_line — u64  + STORED + INDEXED
///   end_line   — u64  + STORED + INDEXED
///   lang       — STRING + STORED
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
    pub content: Field,
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

    pub fn upsert(
        &mut self,
        file: &str,
        chunk_id: &str,
        start_line: u64,
        end_line: u64,
        lang: &str,
        content: &str,
    ) -> Result<()> {
        // delete-then-insert (tantivy has no native upsert)
        self.writer
            .delete_term(Term::from_field_text(self.fields.chunk_id, chunk_id));

        let mut d = TantivyDocument::new();
        d.add_text(self.fields.file, file);
        d.add_text(self.fields.chunk_id, chunk_id);
        d.add_u64(self.fields.start_line, start_line);
        d.add_u64(self.fields.end_line, end_line);
        d.add_text(self.fields.lang, lang);
        d.add_text(self.fields.content, content);
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
    pub fn list_indexed_files(&self) -> Result<std::collections::HashSet<std::path::PathBuf>> {
        use std::collections::HashSet;
        use std::path::PathBuf;
        let reader = self
            .index
            .reader()
            .context("creating tantivy reader for file enumeration")?;
        let searcher = reader.searcher();
        let mut out: HashSet<PathBuf> = HashSet::new();
        for seg in searcher.segment_readers() {
            let inv = seg
                .inverted_index(self.fields.file)
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
}

fn build_schema() -> (Schema, SchemaFields) {
    let mut b = Schema::builder();
    let file = b.add_text_field("file", STRING | STORED);
    let chunk_id = b.add_text_field("chunk_id", STRING | STORED);
    let start_line = b.add_u64_field("start_line", STORED | INDEXED);
    let end_line = b.add_u64_field("end_line", STORED | INDEXED);
    let lang = b.add_text_field("lang", STRING | STORED);
    let content_opts = TextOptions::default().set_stored().set_indexing_options(
        TextFieldIndexing::default()
            .set_tokenizer(CODE_TOKENIZER_NAME)
            .set_index_option(IndexRecordOption::WithFreqsAndPositions),
    );
    let content = b.add_text_field("content", content_opts);
    let schema = b.build();
    (
        schema,
        SchemaFields {
            file,
            chunk_id,
            start_line,
            end_line,
            lang,
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
    pub content: String,
}

/// Read-only view of the BM25 index. Unlike [`Bm25Index`], this does not
/// allocate an [`IndexWriter`], so it doesn't take a write lock on the
/// directory — multiple `search` invocations can coexist with the indexer
/// (modulo whatever segments have been committed at the time of opening).
pub struct Bm25Search {
    index: Index,
    fields: SchemaFields,
}

impl Bm25Search {
    pub fn open(path: &Path) -> Result<Self> {
        let (_schema, fields) = build_schema();
        let index = Index::open_in_dir(path)
            .with_context(|| format!("opening tantivy index for search in {}", path.display()))?;
        register_code_tokenizer(&index);
        Ok(Self { index, fields })
    }

    /// BM25-rank documents by relevance to `query`.
    ///
    /// Two defenses against QueryParser syntax in natural-language queries:
    /// `:` is replaced by a space before parsing (we only ever search the
    /// `content` field, so `field:value` syntax is never useful — but
    /// `Watcher::run` would otherwise parse as a clause on a nonexistent
    /// field and be dropped), and the rest is parsed leniently so stray
    /// quotes / `+` / `-` can't fail the whole search.
    pub fn search(&self, query: &str, limit: usize) -> Result<Vec<SparseHit>> {
        let reader = self.index.reader().context("creating tantivy reader")?;
        let searcher = reader.searcher();
        let parser = QueryParser::for_index(&self.index, vec![self.fields.content]);
        let sanitized = query.replace(':', " ");
        let (q, parse_errors) = parser.parse_query_lenient(&sanitized);
        if !parse_errors.is_empty() {
            tracing::debug!(
                query = %query,
                errors = ?parse_errors,
                "bm25 query parsed leniently (some clauses dropped)"
            );
        }
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
                content: extract_str(&doc, self.fields.content),
            });
        }
        Ok(hits)
    }

    /// Look up a single chunk's full content by its chunk_id. Used to
    /// upgrade dense-only candidates (which only have the 200-char snippet
    /// from Qdrant payload) to full text before reranking.
    pub fn lookup_content(&self, chunk_id: &str) -> Result<Option<String>> {
        let reader = self.index.reader().context("creating tantivy reader")?;
        let searcher = reader.searcher();
        let term = Term::from_field_text(self.fields.chunk_id, chunk_id);
        let query = TermQuery::new(term, IndexRecordOption::Basic);
        let top = searcher
            .search(&query, &TopDocs::with_limit(1))
            .context("chunk_id term query")?;
        let Some((_, addr)) = top.into_iter().next() else {
            return Ok(None);
        };
        let doc: TantivyDocument = searcher.doc(addr).context("fetching chunk doc")?;
        Ok(doc
            .get_first(self.fields.content)
            .and_then(|v| v.as_str())
            .map(|s| s.to_string()))
    }
}

fn extract_str(doc: &TantivyDocument, field: Field) -> String {
    doc.get_first(field)
        .and_then(|v| v.as_str())
        .map(|s| s.to_string())
        .unwrap_or_default()
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
        idx.upsert(
            "src/portfolio.rs",
            "chunk-1",
            10,
            40,
            "rust",
            "pub fn build_si_portfolio(deposit: Decimal) -> Engine { }",
        )
        .unwrap();
        idx.upsert(
            "src/other.rs",
            "chunk-2",
            1,
            5,
            "rust",
            "fn unrelated_helper() {}",
        )
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
            let hits = search.search(query, 5).unwrap();
            assert!(
                hits.iter().any(|h| h.chunk_id == "chunk-1"),
                "query {:?} should find chunk-1, got {:?}",
                query,
                hits.iter().map(|h| h.chunk_id.clone()).collect::<Vec<_>>()
            );
        }
        // Garbage QueryParser syntax must not error out.
        let hits = search.search("\"unbalanced AND build_si_portfolio", 5);
        assert!(hits.is_ok());

        std::fs::remove_dir_all(&dir).ok();
    }
}
