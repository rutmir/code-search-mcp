use anyhow::{Context, Result};
use std::path::Path;
use tantivy::{
    collector::TopDocs,
    query::{QueryParser, TermQuery},
    schema::{Field, IndexRecordOption, Schema, Value, INDEXED, STORED, STRING, TEXT},
    Index, IndexWriter, TantivyDocument, Term,
};

/// Tantivy BM25 index — sparse retrieval layer for hybrid search.
///
/// Schema:
///   file       — STRING + STORED (exact match on path, e.g. for `path_prefix` filter)
///   chunk_id   — STRING + STORED (used as delete key)
///   start_line — u64  + STORED + INDEXED
///   end_line   — u64  + STORED + INDEXED
///   lang       — STRING + STORED
///   content    — TEXT + STORED (tokenized; primary BM25 search target)
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
    let content = b.add_text_field("content", TEXT | STORED);
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
        Ok(Self { index, fields })
    }

    /// BM25-rank documents by relevance to `query`. The query is parsed with
    /// tantivy's standard `QueryParser` on the `content` field — supports
    /// implicit OR over terms, quoted phrases, `field:value` syntax, etc.
    pub fn search(&self, query: &str, limit: usize) -> Result<Vec<SparseHit>> {
        let reader = self.index.reader().context("creating tantivy reader")?;
        let searcher = reader.searcher();
        let parser = QueryParser::for_index(&self.index, vec![self.fields.content]);
        let q = parser
            .parse_query(query)
            .with_context(|| format!("parsing bm25 query: {}", query))?;
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
