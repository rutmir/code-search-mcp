use anyhow::Result;
use std::collections::HashMap;
use std::path::Path;

use crate::config::ChunkingConfig;

pub struct Chunk {
    pub start_line: usize, // 1-indexed
    pub end_line: usize,   // 1-indexed, inclusive
    pub text: String,
    /// Syntactic kind of this chunk (`"fn"`, `"struct"`, `"impl_method"`,
    /// etc.) — populated by AST-aware chunkers, `None` for line-window.
    pub kind: Option<String>,
    /// Name of the syntactic item (function name, struct name, etc.).
    /// Populated by AST-aware chunkers; `None` for line-window. For
    /// impl-method chunks the format is `"TypeName::method_name"` so
    /// queries like "Foo::bar" can match the qualified path.
    pub name: Option<String>,
}

impl Chunk {
    /// Constructor for line-window / heading-based chunks that don't have
    /// syntactic metadata.
    pub fn plain(start_line: usize, end_line: usize, text: String) -> Self {
        Self {
            start_line,
            end_line,
            text,
            kind: None,
            name: None,
        }
    }
}

/// Dispatch wrapper holding one chunker per language plus a fallback default.
/// Built from [`ChunkingConfig`]'s `strategy` + `per_language` map.
pub struct ChunkerSet {
    default: Chunker,
    by_language: HashMap<String, Chunker>,
}

impl ChunkerSet {
    pub fn from_config(cfg: &ChunkingConfig) -> Result<Self> {
        let default = Chunker::build(
            &cfg.strategy,
            "default",
            cfg.max_chunk_lines,
            cfg.overlap_lines,
            cfg.max_chunk_chars,
        )?;
        let mut by_language = HashMap::new();
        for (lang, lc) in &cfg.per_language {
            let strategy = lc.strategy.as_deref().unwrap_or(&cfg.strategy);
            let max_lines = lc.max_chunk_lines.unwrap_or(cfg.max_chunk_lines);
            let overlap = lc.overlap_lines.unwrap_or(cfg.overlap_lines);
            let max_chars = lc.max_chunk_chars.unwrap_or(cfg.max_chunk_chars);
            let chunker = Chunker::build(strategy, lang, max_lines, overlap, max_chars)?;
            by_language.insert(lang.clone(), chunker);
        }
        Ok(Self {
            default,
            by_language,
        })
    }

    pub fn chunk(&self, content: &str, path: &Path, language: &str) -> Vec<Chunk> {
        let chunker = self.by_language.get(language).unwrap_or(&self.default);
        chunker.chunk(content, path)
    }
}

/// One chunker per strategy. Enum dispatch instead of `Box<dyn>` because
/// the set is small and known at compile time.
pub enum Chunker {
    Lines(LineChunker),
    Headings(HeadingsChunker),
    TreeSitter(TreeSitterChunker),
}

impl Chunker {
    /// Build a chunker. `language` is the walker language string used when
    /// this chunker handles a specific language (`"rust"`, `"python"`,
    /// `"markdown"`, etc.) — needed by `tree-sitter` to pick the grammar.
    /// For the top-level `[chunking]` fallback (which applies to whatever
    /// language doesn't have an override), pass `"default"`.
    pub fn build(
        strategy: &str,
        language: &str,
        max_lines: usize,
        overlap: usize,
        max_chars: usize,
    ) -> Result<Self> {
        match strategy {
            "lines" => Ok(Chunker::Lines(LineChunker::new(
                max_lines, overlap, max_chars,
            ))),
            "headings" => Ok(Chunker::Headings(HeadingsChunker::new(max_chars))),
            "tree-sitter" => {
                let fallback = LineChunker::new(max_lines, overlap, max_chars);
                let chunker = match language {
                    "rust" => TreeSitterChunker::new_rust(max_chars, fallback)?,
                    "python" => TreeSitterChunker::new_python(max_chars, fallback)?,
                    "dart" => TreeSitterChunker::new_dart(max_chars, fallback)?,
                    "cpp" => TreeSitterChunker::new_cpp(max_chars, fallback)?,
                    "typescript" => TreeSitterChunker::new_typescript(max_chars, fallback)?,
                    "javascript" => TreeSitterChunker::new_javascript(max_chars, fallback)?,
                    "go" => TreeSitterChunker::new_go(max_chars, fallback)?,
                    "csharp" => TreeSitterChunker::new_csharp(max_chars, fallback)?,
                    "java" => TreeSitterChunker::new_java(max_chars, fallback)?,
                    other => anyhow::bail!(
                        "tree-sitter strategy not supported for language '{}' \
                         (supported: 'rust', 'python', 'dart', 'cpp', \
                         'typescript', 'javascript', 'go', 'csharp', 'java')",
                        other
                    ),
                };
                Ok(Chunker::TreeSitter(chunker))
            }
            other => anyhow::bail!(
                "unknown chunking strategy: {} (supported: 'lines', 'headings', 'tree-sitter')",
                other
            ),
        }
    }

    pub fn chunk(&self, content: &str, path: &Path) -> Vec<Chunk> {
        match self {
            Chunker::Lines(c) => c.chunk(content, path),
            Chunker::Headings(c) => c.chunk(content, path),
            Chunker::TreeSitter(c) => c.chunk(content, path),
        }
    }
}

/// Byte-aware line-window chunker. V1 fallback before tree-sitter AST chunking.
///
/// Cuts on whichever limit is hit first:
///   * `max_lines` — typical case for code with normal-length lines
///   * `max_chars` — kicks in for files with very long lines (markdown tables,
///     minified code, prose) so we never emit a chunk the embedding server
///     can't process.
///
/// Overlap is preserved when chunks reach `max_lines`; when a chunk is cut
/// short by the byte limit (so its line count is already small), overlap is
/// proportionally reduced to keep forward progress.
///
/// Pathological case — a single line longer than `max_chars` — is emitted as
/// its own chunk and skipped downstream by the embedding-side guard, isolated
/// from the rest of the file.
pub struct LineChunker {
    max_lines: usize,
    overlap: usize,
    max_chars: usize,
}

impl LineChunker {
    pub fn new(max_lines: usize, overlap: usize, max_chars: usize) -> Self {
        assert!(max_lines > 0, "max_lines must be > 0");
        assert!(
            max_lines > overlap,
            "max_lines ({}) must be greater than overlap ({})",
            max_lines,
            overlap
        );
        assert!(max_chars > 0, "max_chars must be > 0");
        Self {
            max_lines,
            overlap,
            max_chars,
        }
    }

    pub fn chunk(&self, content: &str, _path: &Path) -> Vec<Chunk> {
        let lines: Vec<&str> = content.lines().collect();
        if lines.is_empty() {
            return Vec::new();
        }

        let mut chunks = Vec::new();
        let mut start = 0usize;

        while start < lines.len() {
            // Greedily accumulate lines, stopping at the FIRST of:
            //   - max_lines reached
            //   - adding the next line would exceed max_chars (only if we
            //     already have at least one line; otherwise we'd loop forever
            //     on a single huge line)
            let mut end = start;
            let mut bytes = 0usize;
            while end < lines.len() && (end - start) < self.max_lines {
                let line_bytes = lines[end].len() + 1; // +1 for separating newline
                if bytes + line_bytes > self.max_chars && end > start {
                    break;
                }
                bytes += line_bytes;
                end += 1;
            }

            // Inner loop always advances `end` by at least 1 (the byte-cap break
            // requires `end > start`), so `end > start` here.
            //
            // If the produced chunk is a single line that's still bigger than
            // max_chars (markdown table row, prose paragraph in one line, etc.),
            // sub-split on sentence boundaries instead of emitting one giant
            // chunk that the embedding guard would skip wholesale.
            //
            // Sentence boundary = `. ` / `! ` / `? ` (terminator + whitespace).
            // `obj.method()` and similar code patterns don't trigger this.
            let chunk_text = lines[start..end].join("\n");
            if end - start == 1 && chunk_text.len() > self.max_chars {
                for segment in split_on_sentences(&chunk_text, self.max_chars) {
                    chunks.push(Chunk::plain(start + 1, end, segment));
                }
            } else {
                chunks.push(Chunk::plain(start + 1, end, chunk_text));
            }

            if end >= lines.len() {
                break;
            }

            // Advance with overlap, but if the chunk was cut short by the byte
            // limit and is shorter than overlap, just skip past it (no overlap)
            // to keep forward progress.
            let actual_lines = end - start;
            let stride = if actual_lines > self.overlap {
                actual_lines - self.overlap
            } else {
                actual_lines
            };
            start += stride.max(1);
        }

        chunks
    }
}

/// Split `text` into segments at sentence boundaries (`.`, `!`, `?` followed
/// by space or newline), each segment at most `max_size` bytes when possible.
///
/// Greedy: accumulates sentences until the next one would exceed `max_size`,
/// then emits the current accumulation as a segment and starts fresh.
///
/// If a single sentence itself is larger than `max_size`, it's emitted as a
/// single oversized segment (downstream embedding guard will skip just that).
///
/// Safe on UTF-8: `.`, `!`, `?` are all single-byte ASCII (0x21/0x2E/0x3F)
/// and never appear as continuation bytes in any multi-byte UTF-8 sequence.
fn split_on_sentences(text: &str, max_size: usize) -> Vec<String> {
    let sentences = collect_sentences(text);
    if sentences.is_empty() {
        return vec![text.to_string()];
    }

    let mut out: Vec<String> = Vec::new();
    let mut current = String::new();
    for s in sentences {
        if !current.is_empty() && current.len() + s.len() > max_size {
            out.push(std::mem::take(&mut current));
        }
        current.push_str(&s);
    }
    if !current.is_empty() {
        out.push(current);
    }
    out
}

fn collect_sentences(text: &str) -> Vec<String> {
    let bytes = text.as_bytes();
    let mut result = Vec::new();
    let mut start = 0usize;
    let mut i = 0usize;
    while i < bytes.len() {
        let b = bytes[i];
        let is_terminator = b == b'.' || b == b'!' || b == b'?';
        let followed_by_break = i + 1 < bytes.len()
            && (bytes[i + 1] == b' ' || bytes[i + 1] == b'\t' || bytes[i + 1] == b'\n');
        if is_terminator && (followed_by_break || i + 1 == bytes.len()) {
            // Include trailing whitespace in the sentence so the rejoin reads naturally.
            let mut end = i + 1;
            while end < bytes.len()
                && (bytes[end] == b' ' || bytes[end] == b'\t' || bytes[end] == b'\n')
            {
                end += 1;
            }
            result.push(text[start..end].to_string());
            start = end;
            i = end;
            continue;
        }
        i += 1;
    }
    if start < text.len() {
        result.push(text[start..].to_string());
    }
    result
}

/// Markdown-aware chunker: cuts on H1/H2 ATX headings (`^#{1,2} `).
/// Each chunk = one section (heading + following content up to the next
/// heading). H3+ are treated as content, not boundaries — too granular
/// for retrieval. Fenced code blocks (``` ... ```) are tracked so a
/// `# python comment` inside one isn't mistaken for a heading.
///
/// Section length cap: `max_chars`. Sections exceeding this fall back to
/// greedy line-pack with sentence sub-split for monstrously long single
/// lines (markdown paragraphs).
pub struct HeadingsChunker {
    max_chars: usize,
}

impl HeadingsChunker {
    pub fn new(max_chars: usize) -> Self {
        assert!(max_chars > 0, "max_chars must be > 0");
        Self { max_chars }
    }

    pub fn chunk(&self, content: &str, _path: &Path) -> Vec<Chunk> {
        let lines: Vec<&str> = content.lines().collect();
        if lines.is_empty() {
            return Vec::new();
        }

        let boundaries = find_heading_boundaries(&lines);
        let mut chunks = Vec::new();
        for window in boundaries.windows(2) {
            let start = window[0];
            let end = window[1];
            if end == start {
                continue;
            }
            self.emit_section(&lines[start..end], start, &mut chunks);
        }
        chunks
    }

    /// Emit one section as one or more chunks, sub-splitting if it
    /// exceeds `max_chars`. Sub-split logic mirrors LineChunker's greedy
    /// line accumulator (without the overlap concept — heading
    /// boundaries are natural cuts, no need to repeat lines).
    fn emit_section(&self, lines: &[&str], offset: usize, out: &mut Vec<Chunk>) {
        let text = lines.join("\n");
        if text.len() <= self.max_chars {
            out.push(Chunk::plain(offset + 1, offset + lines.len(), text));
            return;
        }

        // Section too big — pack greedily by line, with sentence sub-split
        // for very long single lines (markdown paragraphs as one literal
        // line are common: tables, prose).
        let mut i = 0;
        while i < lines.len() {
            let mut j = i;
            let mut bytes = 0usize;
            while j < lines.len() {
                let line_bytes = lines[j].len() + 1; // +1 for joining newline
                if bytes + line_bytes > self.max_chars && j > i {
                    break;
                }
                bytes += line_bytes;
                j += 1;
            }
            let chunk_text = lines[i..j].join("\n");
            if j - i == 1 && chunk_text.len() > self.max_chars {
                // Single oversized line — sub-split on sentences. Each
                // segment gets the same line range (we can't know which
                // sentence is on which sub-line).
                for segment in split_on_sentences(&chunk_text, self.max_chars) {
                    out.push(Chunk::plain(offset + i + 1, offset + j, segment));
                }
            } else {
                out.push(Chunk::plain(offset + i + 1, offset + j, chunk_text));
            }
            i = j;
        }
    }
}

/// Walk lines tracking fenced-code-block state, return the line indices
/// at which a new chunk should start (= every H1/H2 heading, plus 0 and
/// `lines.len()` sentinels).
fn find_heading_boundaries(lines: &[&str]) -> Vec<usize> {
    let mut boundaries = vec![0usize];
    let mut in_code_block = false;
    for (i, line) in lines.iter().enumerate() {
        let trimmed = line.trim_start();
        // ``` toggles fenced-block state. We don't care about the language
        // hint after the backticks. Tilde fences (~~~) are rare in code
        // projects; skipping to keep this simple.
        if trimmed.starts_with("```") {
            in_code_block = !in_code_block;
            continue;
        }
        if in_code_block {
            continue;
        }
        if is_h1_or_h2_heading(trimmed) && i > 0 {
            boundaries.push(i);
        }
    }
    boundaries.push(lines.len());
    boundaries
}

fn is_h1_or_h2_heading(trimmed: &str) -> bool {
    let bytes = trimmed.as_bytes();
    let mut count = 0;
    for &b in bytes {
        if b == b'#' {
            count += 1;
        } else {
            break;
        }
    }
    if count == 0 || count > 2 {
        return false;
    }
    // ATX heading requires whitespace after the leading hashes.
    matches!(bytes.get(count).copied(), Some(b' ') | Some(b'\t'))
}

// ============================================================================
// TreeSitterChunker — AST-aware chunking for Rust
// ============================================================================

/// AST-aware chunker. Per language we extract the structurally
/// interesting top-level items as chunks:
///
/// **Rust** — fn / struct / enum / trait / impl / mod / const / static /
/// type / use / macro. `impl` blocks emit each contained method as its
/// own chunk with `kind = "impl_method"` and `name = "TypeName::method_name"`.
///
/// **Python** — top-level `function_definition`, `class_definition`,
/// `decorated_definition`. Class methods get qualified names
/// `name = "ClassName.method_name"` with `kind = "method"`. The class
/// itself is also emitted as one chunk (`kind = "class"`) so queries
/// about the class as a whole hit too.
///
/// On parse failure or for chunks exceeding `max_chars`, falls back to
/// the supplied [`LineChunker`] for that file/region.
pub struct TreeSitterChunker {
    max_chars: usize,
    lang_id: LangId,
    language: tree_sitter::Language,
    fallback: LineChunker,
}

/// Which language's grammar this chunker is wired to. Drives the node-type
/// dispatch in `emit_node` — each language has a different AST shape.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LangId {
    Rust,
    Python,
    Dart,
    Cpp,
    TypeScript,
    JavaScript,
    Go,
    CSharp,
    Java,
}

impl TreeSitterChunker {
    pub fn new_rust(max_chars: usize, fallback: LineChunker) -> Result<Self> {
        assert!(max_chars > 0);
        let language = tree_sitter_rust::LANGUAGE.into();
        Ok(Self {
            max_chars,
            lang_id: LangId::Rust,
            language,
            fallback,
        })
    }

    pub fn new_python(max_chars: usize, fallback: LineChunker) -> Result<Self> {
        assert!(max_chars > 0);
        let language = tree_sitter_python::LANGUAGE.into();
        Ok(Self {
            max_chars,
            lang_id: LangId::Python,
            language,
            fallback,
        })
    }

    pub fn new_dart(max_chars: usize, fallback: LineChunker) -> Result<Self> {
        assert!(max_chars > 0);
        let language = tree_sitter_dart::LANGUAGE.into();
        Ok(Self {
            max_chars,
            lang_id: LangId::Dart,
            language,
            fallback,
        })
    }

    pub fn new_cpp(max_chars: usize, fallback: LineChunker) -> Result<Self> {
        assert!(max_chars > 0);
        let language = tree_sitter_cpp::LANGUAGE.into();
        Ok(Self {
            max_chars,
            lang_id: LangId::Cpp,
            language,
            fallback,
        })
    }

    pub fn new_typescript(max_chars: usize, fallback: LineChunker) -> Result<Self> {
        assert!(max_chars > 0);
        // tree-sitter-typescript exposes both a TS grammar and a TSX one.
        // We use the TSX grammar because it's a strict superset (parses
        // plain .ts files identically while also handling .tsx).
        let language = tree_sitter_typescript::LANGUAGE_TSX.into();
        Ok(Self {
            max_chars,
            lang_id: LangId::TypeScript,
            language,
            fallback,
        })
    }

    pub fn new_javascript(max_chars: usize, fallback: LineChunker) -> Result<Self> {
        assert!(max_chars > 0);
        let language = tree_sitter_javascript::LANGUAGE.into();
        Ok(Self {
            max_chars,
            lang_id: LangId::JavaScript,
            language,
            fallback,
        })
    }

    pub fn new_go(max_chars: usize, fallback: LineChunker) -> Result<Self> {
        assert!(max_chars > 0);
        let language = tree_sitter_go::LANGUAGE.into();
        Ok(Self {
            max_chars,
            lang_id: LangId::Go,
            language,
            fallback,
        })
    }

    pub fn new_csharp(max_chars: usize, fallback: LineChunker) -> Result<Self> {
        assert!(max_chars > 0);
        let language = tree_sitter_c_sharp::LANGUAGE.into();
        Ok(Self {
            max_chars,
            lang_id: LangId::CSharp,
            language,
            fallback,
        })
    }

    pub fn new_java(max_chars: usize, fallback: LineChunker) -> Result<Self> {
        assert!(max_chars > 0);
        let language = tree_sitter_java::LANGUAGE.into();
        Ok(Self {
            max_chars,
            lang_id: LangId::Java,
            language,
            fallback,
        })
    }

    pub fn chunk(&self, content: &str, path: &Path) -> Vec<Chunk> {
        let mut parser = tree_sitter::Parser::new();
        if parser.set_language(&self.language).is_err() {
            // Should never happen — language is fixed at construction time.
            tracing::warn!(file = %path.display(), "tree-sitter set_language failed; falling back to lines");
            return self.fallback.chunk(content, path);
        }
        let Some(tree) = parser.parse(content, None) else {
            tracing::warn!(file = %path.display(), "tree-sitter parse returned None; falling back to lines");
            return self.fallback.chunk(content, path);
        };
        let root = tree.root_node();
        if root.has_error() {
            // Partial parse: tree-sitter is permissive and produces a tree
            // anyway. We try to use it but expect some items to be missed.
            // For files with serious syntax errors we still get *some*
            // structure rather than reverting to line windows.
            tracing::debug!(file = %path.display(), "tree-sitter parse has errors; using best-effort tree");
        }

        let mut out = Vec::new();
        let mut cursor = root.walk();
        let children: Vec<tree_sitter::Node> = root.children(&mut cursor).collect();

        // Python-specific pre-pass: group the leading top-level imports
        // into one "imports" chunk so they're retrievable as a unit
        // (queries like "where does this module import os" hit). The
        // per-statement emit_python_node skips them so we don't double-count.
        if matches!(self.lang_id, LangId::Python) {
            self.emit_python_imports_group(content.as_bytes(), &children, &mut out);
        }

        for child in children {
            self.emit_node(content.as_bytes(), child, path, None, &mut out);
        }

        // If we got nothing useful (e.g. file is comments-only or pure
        // imports + script logic with no def/class), fall back so the
        // file still gets indexed via line windows.
        if out.is_empty() && !content.trim().is_empty() {
            return self.fallback.chunk(content, path);
        }
        out
    }

    /// Collect the leading run of top-level imports into a single chunk
    /// with `kind = "imports"`. We only group the first contiguous run
    /// (the file's import prelude); imports buried mid-file are ignored
    /// here and skipped entirely by emit_python_node — rare in practice.
    fn emit_python_imports_group(
        &self,
        source: &[u8],
        children: &[tree_sitter::Node],
        out: &mut Vec<Chunk>,
    ) {
        let is_import = |k: &str| {
            matches!(
                k,
                "import_statement" | "import_from_statement" | "future_import_statement"
            )
        };
        let mut start_idx: Option<usize> = None;
        let mut end_idx: Option<usize> = None;
        for (i, child) in children.iter().enumerate() {
            if is_import(child.kind()) {
                start_idx.get_or_insert(i);
                end_idx = Some(i);
            } else if start_idx.is_some() {
                break; // first non-import after the prelude — stop scanning
            }
        }
        let (Some(s), Some(e)) = (start_idx, end_idx) else {
            return;
        };
        let first = children[s];
        let last = children[e];
        let start_byte = first.start_byte();
        let end_byte = last.end_byte();
        if end_byte <= start_byte || end_byte > source.len() {
            return;
        }
        let Some(text_str) = std::str::from_utf8(&source[start_byte..end_byte]).ok() else {
            return;
        };
        out.push(Chunk {
            start_line: first.start_position().row + 1,
            end_line: last.end_position().row + 1,
            text: text_str.to_string(),
            kind: Some("imports".to_string()),
            name: None,
        });
    }

    /// Dispatch entry: pick the per-language emitter based on `lang_id`.
    fn emit_node(
        &self,
        source: &[u8],
        node: tree_sitter::Node,
        path: &Path,
        parent_type: Option<String>,
        out: &mut Vec<Chunk>,
    ) {
        match self.lang_id {
            LangId::Rust => self.emit_rust_node(source, node, path, parent_type, out),
            LangId::Python => self.emit_python_node(source, node, path, parent_type, out),
            LangId::Dart => self.emit_dart_node(source, node, path, parent_type, out),
            LangId::Cpp => self.emit_cpp_node(source, node, path, parent_type, out),
            LangId::TypeScript | LangId::JavaScript => {
                // TS and JS share emitter; TS-only nodes (interface_declaration,
                // type_alias_declaration, enum_declaration) are skipped when
                // parsed via the JS grammar because they don't appear.
                self.emit_js_node(source, node, path, parent_type, out)
            }
            LangId::Go => self.emit_go_node(source, node, path, parent_type, out),
            LangId::CSharp => self.emit_csharp_node(source, node, path, parent_type, out),
            LangId::Java => self.emit_java_node(source, node, path, parent_type, out),
        }
    }

    /// Rust node emitter. For `impl_item` we descend into the impl body and
    /// emit each contained `function_item` as its own `impl_method` chunk;
    /// the impl block itself is NOT also emitted (would duplicate the
    /// methods' content).
    fn emit_rust_node(
        &self,
        source: &[u8],
        node: tree_sitter::Node,
        path: &Path,
        parent_type: Option<String>,
        out: &mut Vec<Chunk>,
    ) {
        let kind = node.kind();
        match kind {
            "impl_item" => self.emit_impl(source, node, path, out),
            "function_item" => {
                let name = field_text(source, node, "name");
                let qualified = match (parent_type, name) {
                    (Some(parent), Some(method)) => Some(format!("{}::{}", parent, method)),
                    (None, Some(method)) => Some(method),
                    _ => None,
                };
                let chunk_kind = if
                /* in impl body */
                qualified.as_deref().is_some_and(|s| s.contains("::")) {
                    "impl_method"
                } else {
                    "fn"
                };
                self.emit_one(source, node, path, chunk_kind, qualified, out);
            }
            "struct_item" | "enum_item" | "trait_item" | "union_item" => {
                let name = field_text(source, node, "name");
                self.emit_one(source, node, path, item_kind_short(kind), name, out);
            }
            "type_item" => {
                let name = field_text(source, node, "name");
                self.emit_one(source, node, path, "type", name, out);
            }
            "const_item" | "static_item" => {
                let name = field_text(source, node, "name");
                self.emit_one(source, node, path, item_kind_short(kind), name, out);
            }
            "use_declaration" | "extern_crate_declaration" => {
                // Group consecutive use/extern lines into one chunk? For
                // now, one chunk each — usually they're cheap and rarely
                // get retrieved standalone.
                self.emit_one(source, node, path, "use", None, out);
            }
            "macro_definition" | "macro_invocation" => {
                let name = field_text(source, node, "name");
                self.emit_one(source, node, path, "macro", name, out);
            }
            "mod_item" => {
                // Descend: emit a header chunk + the contained items.
                // Simpler choice: just emit each contained item with no
                // parent prefix (mod boundary lost). For now go simple.
                if let Some(body) = node.child_by_field_name("body") {
                    let mut cursor = body.walk();
                    for child in body.children(&mut cursor) {
                        self.emit_node(source, child, path, None, out);
                    }
                } else {
                    // `mod foo;` — file-level declaration, emit as one chunk.
                    let name = field_text(source, node, "name");
                    self.emit_one(source, node, path, "mod", name, out);
                }
            }
            // Catch-all for things we don't structure individually:
            // attribute_item, line_comment, block_comment, etc. Skip.
            _ => {}
        }
    }

    fn emit_impl(&self, source: &[u8], node: tree_sitter::Node, path: &Path, out: &mut Vec<Chunk>) {
        // Identify the "Self type" — what `impl X { ... }` or
        // `impl Trait for X { ... }` is implementing.
        let target_type =
            field_text(source, node, "type").or_else(|| field_text(source, node, "name"));
        // Descend into the impl body and emit each function as
        // "TypeName::method_name". Non-function items inside impl
        // (associated types/consts) currently get emitted too.
        if let Some(body) = node.child_by_field_name("body") {
            let mut cursor = body.walk();
            for child in body.children(&mut cursor) {
                self.emit_node(source, child, path, target_type.clone(), out);
            }
        }
        // If the impl was empty, emit it as a "impl" chunk so the user can
        // still find it.
        // (Skipped for now — empty impls are rare.)
    }

    /// Python node emitter. Extracts the structurally interesting top-level
    /// items: function/class definitions plus their decorated variants.
    /// Class methods get qualified names `ClassName.method`. The class is
    /// emitted as a whole AND its methods are emitted separately — small
    /// duplication of body content, but lets queries about the class as a
    /// whole AND queries about specific methods both find what they need.
    ///
    /// Top-level imports / module-level assignments / `if __name__` blocks
    /// are NOT separately emitted. Files that consist entirely of those
    /// (no def/class at all) fall back to line chunker via the empty-out
    /// check in [`Self::chunk`].
    fn emit_python_node(
        &self,
        source: &[u8],
        node: tree_sitter::Node,
        path: &Path,
        parent_type: Option<String>,
        out: &mut Vec<Chunk>,
    ) {
        let kind = node.kind();
        match kind {
            "function_definition" => {
                let name = field_text(source, node, "name");
                let qualified = match (parent_type, name) {
                    (Some(parent), Some(method)) => Some(format!("{}.{}", parent, method)),
                    (None, Some(method)) => Some(method),
                    _ => None,
                };
                // Inside a class body → "method", otherwise → "fn".
                let chunk_kind = if qualified.as_deref().is_some_and(|s| s.contains('.')) {
                    "method"
                } else {
                    "fn"
                };
                self.emit_one(source, node, path, chunk_kind, qualified, out);
            }
            "class_definition" => self.emit_python_class(source, node, path, out),
            "decorated_definition" => {
                // `decorated_definition` wraps a function or class with
                // its `@decorator` lines. We emit using the OUTER node's
                // range (so decorators are included in the chunk text)
                // but the kind/name come from the INNER definition.
                let Some(inner) = node.child_by_field_name("definition") else {
                    return;
                };
                match inner.kind() {
                    "function_definition" => {
                        let name = field_text(source, inner, "name");
                        let qualified = match (parent_type, name) {
                            (Some(parent), Some(method)) => Some(format!("{}.{}", parent, method)),
                            (None, Some(method)) => Some(method),
                            _ => None,
                        };
                        let chunk_kind = if qualified.as_deref().is_some_and(|s| s.contains('.')) {
                            "method"
                        } else {
                            "fn"
                        };
                        self.emit_one(source, node, path, chunk_kind, qualified, out);
                    }
                    "class_definition" => {
                        // Emit the decorated class block as one chunk, then
                        // descend into the inner class's body for methods.
                        let class_name = field_text(source, inner, "name");
                        self.emit_one(source, node, path, "class", class_name.clone(), out);
                        if let Some(body) = inner.child_by_field_name("body") {
                            let mut cursor = body.walk();
                            for child in body.children(&mut cursor) {
                                self.emit_python_node(source, child, path, class_name.clone(), out);
                            }
                        }
                    }
                    _ => {}
                }
            }
            // Everything else at module level: skip. Imports, top-level
            // assignments, `if __name__ == "__main__":` blocks, etc. all
            // get unindexed individually. They DO get matched when their
            // file falls through to the line chunker because no structural
            // items were extracted (see `chunk`'s empty-out fallback).
            _ => {}
        }
    }

    fn emit_python_class(
        &self,
        source: &[u8],
        node: tree_sitter::Node,
        path: &Path,
        out: &mut Vec<Chunk>,
    ) {
        let class_name = field_text(source, node, "name");
        // Class as a whole.
        self.emit_one(source, node, path, "class", class_name.clone(), out);
        // Then each method (and any nested function) inside the body.
        if let Some(body) = node.child_by_field_name("body") {
            let mut cursor = body.walk();
            for child in body.children(&mut cursor) {
                self.emit_python_node(source, child, path, class_name.clone(), out);
            }
        }
    }

    /// Dart node emitter. Handles top-level class / enum / mixin / extension
    /// declarations. Methods inside a class body get qualified names
    /// `ClassName.method`. The class itself is also emitted (kind="class").
    ///
    /// Limitations of the `tree-sitter-dart 0.0.4` grammar we depend on:
    /// it's old and incomplete around top-level functions. Files with
    /// nothing but top-level functions may fall back to line-window
    /// chunking; class methods do work cleanly.
    fn emit_dart_node(
        &self,
        source: &[u8],
        node: tree_sitter::Node,
        path: &Path,
        parent_type: Option<String>,
        out: &mut Vec<Chunk>,
    ) {
        let kind = node.kind();
        match kind {
            "class_declaration" => self.emit_dart_class(source, node, path, out),
            "enum_declaration" => {
                let name = field_text(source, node, "name");
                self.emit_one(source, node, path, "enum", name, out);
            }
            "mixin_declaration" => {
                let name = field_text(source, node, "name");
                self.emit_one(source, node, path, "mixin", name, out);
            }
            "extension_declaration" => {
                let name = field_text(source, node, "name");
                self.emit_one(source, node, path, "extension", name, out);
            }
            // A top-level function. `function_declaration` is what the
            // 0.2 grammar produces and it carries the body; bare
            // `function_signature` is kept because some inputs still
            // surface a signature without one.
            "function_declaration" | "function_signature" => {
                let name = node
                    .child_by_field_name("signature")
                    .and_then(|sig| field_text(source, sig, "name"))
                    .or_else(|| field_text(source, node, "name"));
                let qualified = match (parent_type, name) {
                    (Some(parent), Some(method)) => Some(format!("{}.{}", parent, method)),
                    (None, Some(method)) => Some(method),
                    _ => None,
                };
                let chunk_kind = if qualified.as_deref().is_some_and(|s| s.contains('.')) {
                    "method"
                } else {
                    "fn"
                };
                self.emit_one(source, node, path, chunk_kind, qualified, out);
            }
            _ => {}
        }
    }

    fn emit_dart_class(
        &self,
        source: &[u8],
        node: tree_sitter::Node,
        path: &Path,
        out: &mut Vec<Chunk>,
    ) {
        let class_name = field_text(source, node, "name");
        // Whole class as one chunk.
        self.emit_one(source, node, path, "class", class_name.clone(), out);
        // Each method/getter/setter/constructor inside class_body.
        let Some(body) = node.child_by_field_name("body") else {
            return;
        };
        let mut cursor = body.walk();
        for member in body.children(&mut cursor) {
            if member.kind() != "class_member" {
                continue;
            }
            // Inside a class_member_definition we look for a method_signature
            // (which wraps a function_signature / constructor_signature /
            // getter_signature / setter_signature / operator_signature). The
            // chunk text uses the whole class_member_definition range so we
            // capture the body too.
            let method_name = find_dart_method_name(source, member);
            let qualified = match (&class_name, method_name) {
                (Some(c), Some(m)) => Some(format!("{}.{}", c, m)),
                (None, Some(m)) => Some(m),
                _ => continue, // no name → skip (e.g. field declarations)
            };
            self.emit_one(source, member, path, "method", qualified, out);
        }
    }

    /// C++ node emitter. Handles:
    ///   - `function_definition` at top level → `fn`. For outline method
    ///     definitions like `void Foo::bar() { ... }`, the function's name
    ///     is a `qualified_identifier` (`Foo::bar`) and the chunk kind
    ///     stays `fn` (the `::` in the name is the structural anchor).
    ///   - `class_specifier` / `struct_specifier` → `class` / `struct`,
    ///     descend into body for inline methods (`method`).
    ///   - `namespace_definition` → `namespace`, descend.
    ///   - `template_declaration` → wraps a function/class. Emit using the
    ///     OUTER range (so template parameters are in the chunk text), but
    ///     kind/name from the wrapped declaration.
    ///   - `enum_specifier`, `union_specifier` → emit as is.
    fn emit_cpp_node(
        &self,
        source: &[u8],
        node: tree_sitter::Node,
        path: &Path,
        parent_type: Option<String>,
        out: &mut Vec<Chunk>,
    ) {
        let kind = node.kind();
        match kind {
            "function_definition" => {
                let name = extract_cpp_declarator_name(source, node);
                let (chunk_kind, qualified) = match (parent_type, name) {
                    (Some(parent), Some(method)) => {
                        ("method", Some(format!("{}::{}", parent, method)))
                    }
                    (None, Some(name)) => ("fn", Some(name)),
                    _ => ("fn", None),
                };
                self.emit_one(source, node, path, chunk_kind, qualified, out);
            }
            "class_specifier" | "struct_specifier" | "union_specifier" => {
                self.emit_cpp_record(source, node, path, kind, out);
            }
            "enum_specifier" => {
                let name = field_text(source, node, "name");
                self.emit_one(source, node, path, "enum", name, out);
            }
            "namespace_definition" => {
                let name = field_text(source, node, "name");
                // Big namespaces are containers — emit them as a chunk AND
                // descend so their internal items also surface. (Emitting
                // both gives mild duplication, OK for code search recall.)
                self.emit_one(source, node, path, "namespace", name.clone(), out);
                if let Some(body) = node.child_by_field_name("body") {
                    let mut cursor = body.walk();
                    for child in body.children(&mut cursor) {
                        self.emit_cpp_node(source, child, path, name.clone(), out);
                    }
                }
            }
            "template_declaration" => {
                // template_declaration wraps a function_definition,
                // class_specifier, struct_specifier, etc. Find the wrapped
                // node, take its kind/name, but emit using the template's
                // outer byte range so template parameters land in the chunk.
                let mut cursor = node.walk();
                for child in node.children(&mut cursor) {
                    match child.kind() {
                        "function_definition" => {
                            let name = extract_cpp_declarator_name(source, child);
                            self.emit_one(source, node, path, "fn", name, out);
                            return;
                        }
                        "class_specifier" => {
                            let name = field_text(source, child, "name");
                            self.emit_one(source, node, path, "class", name.clone(), out);
                            if let Some(body) = child.child_by_field_name("body") {
                                let mut c = body.walk();
                                for inner in body.children(&mut c) {
                                    self.emit_cpp_node(source, inner, path, name.clone(), out);
                                }
                            }
                            return;
                        }
                        "struct_specifier" => {
                            let name = field_text(source, child, "name");
                            self.emit_one(source, node, path, "struct", name.clone(), out);
                            if let Some(body) = child.child_by_field_name("body") {
                                let mut c = body.walk();
                                for inner in body.children(&mut c) {
                                    self.emit_cpp_node(source, inner, path, name.clone(), out);
                                }
                            }
                            return;
                        }
                        _ => {}
                    }
                }
            }
            // Skip: preproc_*, declaration (top-level extern decls),
            // using_declaration, alias_declaration, type_definition, etc.
            // For headers consisting only of declarations the empty-out
            // fallback to line chunker kicks in.
            _ => {}
        }
    }

    fn emit_cpp_record(
        &self,
        source: &[u8],
        node: tree_sitter::Node,
        path: &Path,
        node_kind: &str,
        out: &mut Vec<Chunk>,
    ) {
        let kind_short = match node_kind {
            "class_specifier" => "class",
            "struct_specifier" => "struct",
            "union_specifier" => "union",
            _ => "record",
        };
        let name = field_text(source, node, "name");
        // Whole record as one chunk.
        self.emit_one(source, node, path, kind_short, name.clone(), out);
        // Descend body for inline methods. tree-sitter-cpp's class body is
        // `field_declaration_list`; method declarations / definitions live
        // alongside fields.
        if let Some(body) = node.child_by_field_name("body") {
            let mut cursor = body.walk();
            for child in body.children(&mut cursor) {
                self.emit_cpp_node(source, child, path, name.clone(), out);
            }
        }
    }

    /// JavaScript / TypeScript node emitter (shared — the TS grammar is a
    /// superset of JS). Handles:
    ///   - `function_declaration` → `fn`
    ///   - `class_declaration` → `class`, descend body
    ///   - `method_definition` (inside class body) → `method` with qualified
    ///     name `ClassName.method_name`
    ///   - `interface_declaration` → `interface` (TS only)
    ///   - `type_alias_declaration` → `type` (TS only)
    ///   - `enum_declaration` → `enum` (TS only)
    ///   - `export_statement` / `ambient_declaration` → unwrap inner
    ///     declaration and recurse (so `export class Foo {}` produces a
    ///     class chunk with the `export` keyword in its text)
    fn emit_js_node(
        &self,
        source: &[u8],
        node: tree_sitter::Node,
        path: &Path,
        parent_type: Option<String>,
        out: &mut Vec<Chunk>,
    ) {
        let kind = node.kind();
        match kind {
            "function_declaration" => {
                let name = field_text(source, node, "name");
                let qualified = match (parent_type, name) {
                    (Some(parent), Some(method)) => Some(format!("{}.{}", parent, method)),
                    (None, Some(method)) => Some(method),
                    _ => None,
                };
                let chunk_kind = if qualified.as_deref().is_some_and(|s| s.contains('.')) {
                    "method"
                } else {
                    "fn"
                };
                self.emit_one(source, node, path, chunk_kind, qualified, out);
            }
            "class_declaration" => self.emit_js_class(source, node, path, out),
            "method_definition" => {
                let name = field_text(source, node, "name");
                let qualified = match (parent_type, name) {
                    (Some(parent), Some(method)) => Some(format!("{}.{}", parent, method)),
                    (None, Some(method)) => Some(method),
                    _ => None,
                };
                self.emit_one(source, node, path, "method", qualified, out);
            }
            "interface_declaration" => {
                let name = field_text(source, node, "name");
                self.emit_one(source, node, path, "interface", name, out);
            }
            "type_alias_declaration" => {
                let name = field_text(source, node, "name");
                self.emit_one(source, node, path, "type", name, out);
            }
            "enum_declaration" => {
                let name = field_text(source, node, "name");
                self.emit_one(source, node, path, "enum", name, out);
            }
            "export_statement" | "ambient_declaration" => {
                // Unwrap and emit using the OUTER range (so `export` lands
                // in the chunk text) with kind/name from the inner decl.
                let mut cursor = node.walk();
                for child in node.children(&mut cursor) {
                    match child.kind() {
                        "function_declaration" => {
                            let name = field_text(source, child, "name");
                            self.emit_one(source, node, path, "fn", name, out);
                            return;
                        }
                        "class_declaration" => {
                            let name = field_text(source, child, "name");
                            self.emit_one(source, node, path, "class", name.clone(), out);
                            if let Some(body) = child.child_by_field_name("body") {
                                let mut bc = body.walk();
                                for member in body.children(&mut bc) {
                                    self.emit_js_node(source, member, path, name.clone(), out);
                                }
                            }
                            return;
                        }
                        "interface_declaration" => {
                            let name = field_text(source, child, "name");
                            self.emit_one(source, node, path, "interface", name, out);
                            return;
                        }
                        "type_alias_declaration" => {
                            let name = field_text(source, child, "name");
                            self.emit_one(source, node, path, "type", name, out);
                            return;
                        }
                        "enum_declaration" => {
                            let name = field_text(source, child, "name");
                            self.emit_one(source, node, path, "enum", name, out);
                            return;
                        }
                        _ => {}
                    }
                }
            }
            // Skip: imports, top-level expression statements, lexical
            // declarations (`const x = ...`). Files of only these fall
            // back to line chunker via the empty-out path in `chunk`.
            _ => {}
        }
    }

    fn emit_js_class(
        &self,
        source: &[u8],
        node: tree_sitter::Node,
        path: &Path,
        out: &mut Vec<Chunk>,
    ) {
        let class_name = field_text(source, node, "name");
        self.emit_one(source, node, path, "class", class_name.clone(), out);
        let Some(body) = node.child_by_field_name("body") else {
            return;
        };
        let mut cursor = body.walk();
        for member in body.children(&mut cursor) {
            self.emit_js_node(source, member, path, class_name.clone(), out);
        }
    }

    /// Go node emitter. Handles:
    ///   - `function_declaration` (no receiver) → `fn`
    ///   - `method_declaration` (has receiver) → `method` qualified
    ///     `ReceiverType.method_name`. Pointer (`*Foo`) and value (`Foo`)
    ///     receivers normalize to the same base type name.
    ///   - `type_declaration` containing one or more `type_spec`s. We
    ///     inspect each spec's `type` field:
    ///       - `struct_type` → `struct`
    ///       - `interface_type` → `interface`
    ///       - other (aliases, named primitives) → `type`
    ///   - Skip: `package_clause`, `import_declaration`, top-level
    ///     `const_declaration` / `var_declaration` (too small / noise).
    fn emit_go_node(
        &self,
        source: &[u8],
        node: tree_sitter::Node,
        path: &Path,
        _parent_type: Option<String>,
        out: &mut Vec<Chunk>,
    ) {
        let kind = node.kind();
        match kind {
            "function_declaration" => {
                let name = field_text(source, node, "name");
                self.emit_one(source, node, path, "fn", name, out);
            }
            "method_declaration" => {
                let method_name = field_text(source, node, "name");
                let receiver_type = extract_go_receiver_type(source, node);
                let qualified = match (receiver_type, method_name) {
                    (Some(recv), Some(method)) => Some(format!("{}.{}", recv, method)),
                    (None, Some(method)) => Some(method),
                    _ => None,
                };
                self.emit_one(source, node, path, "method", qualified, out);
            }
            "type_declaration" => {
                // type_declaration → one or more `type_spec` children
                // (Go allows `type ( ... )` blocks declaring several types).
                let mut cursor = node.walk();
                for child in node.children(&mut cursor) {
                    if child.kind() != "type_spec" {
                        continue;
                    }
                    let spec_name = field_text(source, child, "name");
                    let type_node = child.child_by_field_name("type");
                    let kind_short = match type_node.map(|n| n.kind()) {
                        Some("struct_type") => "struct",
                        Some("interface_type") => "interface",
                        _ => "type",
                    };
                    self.emit_one(source, child, path, kind_short, spec_name, out);
                }
            }
            _ => {}
        }
    }

    /// C# node emitter. Handles:
    ///   - `class_declaration` / `interface_declaration` / `struct_declaration`
    ///     / `enum_declaration` / `record_declaration` — emit as the whole
    ///     type, then descend into `declaration_list` body for methods.
    ///   - `method_declaration` / `constructor_declaration` /
    ///     `destructor_declaration` — when inside a class body these
    ///     become `method` with `ClassName.method_name`. At top level
    ///     (rare in C#) they become `fn`.
    ///   - `delegate_declaration` → `delegate`
    ///   - `namespace_declaration` / `file_scoped_namespace_declaration` →
    ///     `namespace`, descend.
    fn emit_csharp_node(
        &self,
        source: &[u8],
        node: tree_sitter::Node,
        path: &Path,
        parent_type: Option<String>,
        out: &mut Vec<Chunk>,
    ) {
        let kind = node.kind();
        match kind {
            "class_declaration"
            | "interface_declaration"
            | "struct_declaration"
            | "enum_declaration"
            | "record_declaration" => {
                let chunk_kind = match kind {
                    "class_declaration" => "class",
                    "interface_declaration" => "interface",
                    "struct_declaration" => "struct",
                    "enum_declaration" => "enum",
                    "record_declaration" => "record",
                    _ => "type",
                };
                let name = field_text(source, node, "name");
                self.emit_one(source, node, path, chunk_kind, name.clone(), out);
                // Descend body. The C# grammar names the body field varies
                // by node type — try common ones.
                let body = node.child_by_field_name("body").or_else(|| {
                    // Some node types use a different field; scan for a
                    // declaration_list / enum_member_declaration_list.
                    let mut c = node.walk();
                    let mut found = None;
                    for ch in node.children(&mut c) {
                        if matches!(
                            ch.kind(),
                            "declaration_list" | "enum_member_declaration_list"
                        ) {
                            found = Some(ch);
                            break;
                        }
                    }
                    found
                });
                if let Some(body) = body {
                    let mut cursor = body.walk();
                    for member in body.children(&mut cursor) {
                        self.emit_csharp_node(source, member, path, name.clone(), out);
                    }
                }
            }
            "method_declaration"
            | "constructor_declaration"
            | "destructor_declaration"
            | "local_function_statement" => {
                let name = field_text(source, node, "name");
                let qualified = match (parent_type, name) {
                    (Some(parent), Some(method)) => Some(format!("{}.{}", parent, method)),
                    (None, Some(method)) => Some(method),
                    _ => None,
                };
                let chunk_kind = if qualified.as_deref().is_some_and(|s| s.contains('.')) {
                    "method"
                } else {
                    "fn"
                };
                self.emit_one(source, node, path, chunk_kind, qualified, out);
            }
            "delegate_declaration" => {
                let name = field_text(source, node, "name");
                self.emit_one(source, node, path, "delegate", name, out);
            }
            "namespace_declaration" | "file_scoped_namespace_declaration" => {
                let name = field_text(source, node, "name");
                // Don't emit the whole namespace as one chunk — they tend
                // to span the entire file and would duplicate everything.
                // Just descend for the contained type declarations.
                let body = node.child_by_field_name("body").or_else(|| {
                    let mut c = node.walk();
                    let mut found = None;
                    for ch in node.children(&mut c) {
                        if ch.kind() == "declaration_list" {
                            found = Some(ch);
                            break;
                        }
                    }
                    found
                });
                if let Some(body) = body {
                    let mut cursor = body.walk();
                    for child in body.children(&mut cursor) {
                        self.emit_csharp_node(source, child, path, name.clone(), out);
                    }
                } else {
                    // File-scoped namespace declarations don't have a body —
                    // the rest of the file is the namespace's content. Defer
                    // to the caller's top-level walk to pick those up.
                }
            }
            _ => {}
        }
    }

    /// Java node emitter. Handles:
    ///   - `class_declaration` / `interface_declaration` / `enum_declaration`
    ///     / `record_declaration` / `annotation_type_declaration` — emit
    ///     the whole type then descend body for methods.
    ///   - `method_declaration` / `constructor_declaration` — when inside
    ///     class body, `method` with `ClassName.method_name`. Otherwise `fn`.
    fn emit_java_node(
        &self,
        source: &[u8],
        node: tree_sitter::Node,
        path: &Path,
        parent_type: Option<String>,
        out: &mut Vec<Chunk>,
    ) {
        let kind = node.kind();
        match kind {
            "class_declaration"
            | "interface_declaration"
            | "enum_declaration"
            | "record_declaration"
            | "annotation_type_declaration" => {
                let chunk_kind = match kind {
                    "class_declaration" => "class",
                    "interface_declaration" => "interface",
                    "enum_declaration" => "enum",
                    "record_declaration" => "record",
                    "annotation_type_declaration" => "annotation",
                    _ => "type",
                };
                let name = field_text(source, node, "name");
                self.emit_one(source, node, path, chunk_kind, name.clone(), out);
                // Descend body. tree-sitter-java's class body is field-named.
                if let Some(body) = node.child_by_field_name("body") {
                    let mut cursor = body.walk();
                    for member in body.children(&mut cursor) {
                        self.emit_java_node(source, member, path, name.clone(), out);
                    }
                }
            }
            "method_declaration" | "constructor_declaration" => {
                let name = field_text(source, node, "name");
                let qualified = match (parent_type, name) {
                    (Some(parent), Some(method)) => Some(format!("{}.{}", parent, method)),
                    (None, Some(method)) => Some(method),
                    _ => None,
                };
                let chunk_kind = if qualified.as_deref().is_some_and(|s| s.contains('.')) {
                    "method"
                } else {
                    "fn"
                };
                self.emit_one(source, node, path, chunk_kind, qualified, out);
            }
            _ => {}
        }
    }

    /// Emit ONE chunk for the given node, splitting it via the line
    /// fallback if it exceeds max_chars.
    fn emit_one(
        &self,
        source: &[u8],
        node: tree_sitter::Node,
        path: &Path,
        kind: &str,
        name: Option<String>,
        out: &mut Vec<Chunk>,
    ) {
        let start_byte = node.start_byte();
        let end_byte = node.end_byte();
        if end_byte <= start_byte || end_byte > source.len() {
            return;
        }
        let text = match std::str::from_utf8(&source[start_byte..end_byte]) {
            Ok(s) => s.to_string(),
            Err(_) => return, // non-UTF8 region; skip
        };
        let start_row = node.start_position().row + 1; // 1-indexed
        let end_row = node.end_position().row + 1;

        if text.len() <= self.max_chars {
            out.push(Chunk {
                start_line: start_row,
                end_line: end_row,
                text,
                kind: Some(kind.to_string()),
                name,
            });
            return;
        }

        // Oversized item: fall back to line-window split, carrying kind/name
        // forward on each sub-chunk so the LLM still sees the structural
        // anchor.
        let sub_chunks = self.fallback.chunk(&text, path);
        let row_offset = start_row.saturating_sub(1);
        for sub in sub_chunks {
            out.push(Chunk {
                start_line: row_offset + sub.start_line,
                end_line: row_offset + sub.end_line,
                text: sub.text,
                kind: Some(kind.to_string()),
                name: name.clone(),
            });
        }
    }
}

/// Extract the UTF-8 text of `node`'s `field` (by field name). Returns
/// None if the field is missing or non-UTF-8.
fn field_text(source: &[u8], node: tree_sitter::Node, field: &str) -> Option<String> {
    let n = node.child_by_field_name(field)?;
    let bytes = source.get(n.start_byte()..n.end_byte())?;
    std::str::from_utf8(bytes).ok().map(|s| s.to_string())
}

fn item_kind_short(kind: &str) -> &'static str {
    match kind {
        "struct_item" => "struct",
        "enum_item" => "enum",
        "trait_item" => "trait",
        "union_item" => "union",
        "const_item" => "const",
        "static_item" => "static",
        "function_item" => "fn",
        _ => "item",
    }
}

/// Find the method/getter/setter/constructor name inside a Dart
/// `class_member_definition`. Returns None for non-method members like
/// plain field declarations (which have no identifiable "method name").
fn find_dart_method_name(source: &[u8], member: tree_sitter::Node) -> Option<String> {
    /// How deep to look. A method reaches its name in three steps
    /// (`class_member` → `method_declaration` → `method_signature` →
    /// `function_signature`), a constructor in two (`class_member` →
    /// `declaration` → `constructor_signature`). Four leaves headroom
    /// without wandering into method bodies.
    const MAX_DEPTH: usize = 4;

    fn search(source: &[u8], node: tree_sitter::Node, depth: usize) -> Option<String> {
        if depth > MAX_DEPTH {
            return None;
        }
        // Any `*_signature` that names something. `operator_signature` has
        // no `name` field and is skipped by exactly that test rather than
        // by being listed — one less thing to keep in sync with the
        // grammar.
        if node.kind().ends_with("_signature") {
            if let Some(name) = field_text(source, node, "name") {
                return Some(name);
            }
        }
        let mut cursor = node.walk();
        for child in node.children(&mut cursor) {
            if let Some(found) = search(source, child, depth + 1) {
                return Some(found);
            }
        }
        None
    }

    search(source, member, 0)
}

/// Extract the receiver type name from a Go `method_declaration`. The
/// receiver field is a `parameter_list` containing one
/// `parameter_declaration` whose `type` field is either a `type_identifier`
/// (value receiver `func (f Foo) ...`) or a `pointer_type` (pointer
/// receiver `func (f *Foo) ...`). We normalize both to the base
/// `type_identifier` text so qualified names stay consistent.
fn extract_go_receiver_type(source: &[u8], method_node: tree_sitter::Node) -> Option<String> {
    let receiver = method_node.child_by_field_name("receiver")?;
    let mut cursor = receiver.walk();
    for param in receiver.children(&mut cursor) {
        if param.kind() != "parameter_declaration" {
            continue;
        }
        let type_node = param.child_by_field_name("type")?;
        let name_node = match type_node.kind() {
            "pointer_type" => {
                // pointer_type may expose its inner type via the "type" field
                // OR as a plain child. Try the field first; fall back to a
                // child-walk if not set.
                if let Some(inner) = type_node.child_by_field_name("type") {
                    inner
                } else {
                    let mut c = type_node.walk();
                    let mut found: Option<tree_sitter::Node> = None;
                    for n in type_node.children(&mut c) {
                        if matches!(n.kind(), "type_identifier" | "qualified_type") {
                            found = Some(n);
                            break;
                        }
                    }
                    found?
                }
            }
            "type_identifier" => type_node,
            _ => return None,
        };
        let bytes = source.get(name_node.start_byte()..name_node.end_byte())?;
        return std::str::from_utf8(bytes).ok().map(|s| s.to_string());
    }
    None
}

/// Walk through a C++ `function_definition`'s declarator chain to find
/// the innermost identifier — handles `(*pfn)(...)`,
/// `void Foo::bar(...)`, `template<T> void f(...)`, etc. Returns the
/// fully-qualified name as written in source (so `Foo::bar` comes back
/// verbatim with the `::`).
fn extract_cpp_declarator_name(source: &[u8], func_def: tree_sitter::Node) -> Option<String> {
    let mut current = func_def.child_by_field_name("declarator")?;
    loop {
        match current.kind() {
            "identifier"
            | "field_identifier"
            | "qualified_identifier"
            | "destructor_name"
            | "operator_name" => {
                let bytes = source.get(current.start_byte()..current.end_byte())?;
                return std::str::from_utf8(bytes).ok().map(|s| s.to_string());
            }
            _ => {
                // Descend through nested declarators (function_declarator,
                // parenthesized_declarator, pointer_declarator,
                // reference_declarator, etc.) all of which expose a
                // `declarator` field one level deeper.
                current = current.child_by_field_name("declarator")?;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    fn p() -> PathBuf {
        PathBuf::from("x")
    }

    #[test]
    fn empty_input_produces_no_chunks() {
        let c = LineChunker::new(10, 2, 1000);
        assert!(c.chunk("", &p()).is_empty());
    }

    #[test]
    fn small_file_one_chunk() {
        let c = LineChunker::new(60, 10, 24000);
        let chunks = c.chunk("a\nb\nc", &p());
        assert_eq!(chunks.len(), 1);
        assert_eq!(chunks[0].start_line, 1);
        assert_eq!(chunks[0].end_line, 3);
    }

    #[test]
    fn overlap_preserved_when_line_limit_hits_first() {
        // 100 short lines, max_lines=10, overlap=2 → stride=8, chunks 1..10, 9..18, ...
        let content: String = (1..=100)
            .map(|i| i.to_string())
            .collect::<Vec<_>>()
            .join("\n");
        let c = LineChunker::new(10, 2, 1_000_000);
        let chunks = c.chunk(&content, &p());
        assert_eq!(chunks[0].start_line, 1);
        assert_eq!(chunks[0].end_line, 10);
        assert_eq!(chunks[1].start_line, 9);
        assert_eq!(chunks[1].end_line, 18);
    }

    #[test]
    fn byte_limit_cuts_chunk_short() {
        // Each line is 100 chars; max_lines=60, max_chars=300 → ~3 lines per chunk
        let line = "x".repeat(100);
        let content: Vec<String> = (0..30).map(|_| line.clone()).collect();
        let content = content.join("\n");
        let c = LineChunker::new(60, 1, 300);
        let chunks = c.chunk(&content, &p());
        // Should be many small chunks, not one big one
        assert!(
            chunks.len() > 5,
            "expected many chunks, got {}",
            chunks.len()
        );
        for ch in &chunks {
            assert!(
                ch.text.len() <= 300 || (ch.end_line - ch.start_line + 1) == 1,
                "chunk exceeds max_chars without being single-line: {} chars",
                ch.text.len()
            );
        }
    }

    #[test]
    fn oversized_line_without_sentences_emitted_alone() {
        // One huge line with NO sentence boundaries (just `z`s). Should be emitted
        // as a single oversized chunk; the embedding guard will skip just that one.
        let huge = "z".repeat(5000);
        let content = format!("a\nb\n{}\nc\nd", huge);
        let c = LineChunker::new(60, 1, 1000);
        let chunks = c.chunk(&content, &p());
        let huge_chunk = chunks.iter().find(|ch| ch.text.contains("zzzz")).unwrap();
        assert_eq!(huge_chunk.start_line, huge_chunk.end_line);
        // Normal lines around it still get indexed
        let total_normal: usize = chunks
            .iter()
            .filter(|ch| !ch.text.contains('z'))
            .map(|ch| ch.end_line - ch.start_line + 1)
            .sum();
        assert!(total_normal >= 2);
    }

    #[test]
    fn oversized_line_with_sentences_sub_split() {
        // One long line consisting of many sentences, each fitting well under
        // max_chars. Should be sub-split into per-sentence (or grouped) chunks.
        let sentence = "This is a fact about a thing. ";
        let huge_line: String = sentence.repeat(50); // ~1500 chars
        let content = format!("intro line\n{}\noutro line", huge_line);
        let c = LineChunker::new(60, 1, 300); // max_chars=300, sentence~30 chars
        let chunks = c.chunk(&content, &p());

        // The huge line should produce MULTIPLE chunks, all on the same line index
        let huge_chunks: Vec<_> = chunks
            .iter()
            .filter(|ch| ch.text.contains("fact"))
            .collect();
        assert!(
            huge_chunks.len() >= 3,
            "expected sub-split, got {} chunks",
            huge_chunks.len()
        );
        for ch in &huge_chunks {
            assert_eq!(
                ch.start_line, ch.end_line,
                "all sub-chunks share the source line"
            );
            assert!(
                ch.text.len() <= 300 + 30,
                "each sub-chunk ~max_chars: {}",
                ch.text.len()
            );
        }
    }

    #[test]
    fn code_dots_do_not_trigger_split() {
        // `obj.method()` patterns in code should NOT be split. A period without
        // following whitespace isn't a sentence boundary.
        let code = "let x = foo.bar().baz();";
        let segs = split_on_sentences(code, 5);
        assert_eq!(
            segs.len(),
            1,
            "code-style dots should not split: {:?}",
            segs
        );
    }

    // ---- HeadingsChunker tests ----

    #[test]
    fn headings_simple_split() {
        let md = "\
# Title

Intro paragraph.

## Section One

Content one.

## Section Two

Content two.";
        let c = HeadingsChunker::new(10_000);
        let chunks = c.chunk(md, &p());
        // Expect 3 chunks: H1 block, H2-one block, H2-two block.
        assert_eq!(
            chunks.len(),
            3,
            "got chunks: {:?}",
            chunks.iter().map(|c| &c.text).collect::<Vec<_>>()
        );
        assert!(chunks[0].text.starts_with("# Title"));
        assert!(chunks[1].text.starts_with("## Section One"));
        assert!(chunks[2].text.starts_with("## Section Two"));
    }

    #[test]
    fn headings_inside_code_fence_ignored() {
        // `# something` inside ``` ... ``` is code, not a markdown heading.
        let md = "\
# Real Title

Intro.

```python
# this is a python comment
def foo():
    pass
```

## Real Section

After fence.";
        let c = HeadingsChunker::new(10_000);
        let chunks = c.chunk(md, &p());
        // Two chunks: H1 block (containing the fenced code), and H2 block.
        assert_eq!(
            chunks.len(),
            2,
            "fence-internal '#' must not trigger boundary"
        );
        assert!(chunks[0].text.contains("python comment"));
        assert!(chunks[1].text.starts_with("## Real Section"));
    }

    #[test]
    fn headings_h3_does_not_split() {
        let md = "\
# H1

intro

### H3 sub
content under H3

### Another H3
more content

## H2

other";
        let c = HeadingsChunker::new(10_000);
        let chunks = c.chunk(md, &p());
        // H3s should NOT split — only the H2 does. So 2 chunks.
        assert_eq!(chunks.len(), 2);
        assert!(chunks[0].text.contains("### H3 sub"));
        assert!(chunks[0].text.contains("### Another H3"));
        assert!(chunks[1].text.starts_with("## H2"));
    }

    #[test]
    fn headings_no_headings_one_chunk() {
        let md = "Just a paragraph.\nWith two lines.";
        let c = HeadingsChunker::new(10_000);
        let chunks = c.chunk(md, &p());
        assert_eq!(chunks.len(), 1);
        assert_eq!(chunks[0].start_line, 1);
        assert_eq!(chunks[0].end_line, 2);
    }

    #[test]
    fn headings_oversized_section_falls_back_to_line_pack() {
        // Section bigger than max_chars must be sub-split.
        let big = "## big\n".to_string() + &"line of text\n".repeat(50); // ~ 600+ chars total
        let c = HeadingsChunker::new(150);
        let chunks = c.chunk(&big, &p());
        assert!(
            chunks.len() > 1,
            "expected multi-chunk split, got {}",
            chunks.len()
        );
        for ch in &chunks {
            assert!(
                ch.text.len() <= 200,
                "sub-chunk too large: {} bytes",
                ch.text.len()
            );
        }
    }

    #[test]
    fn headings_lines_without_space_after_hash_not_heading() {
        // `#not-a-heading` (no space) is NOT an ATX heading; should not split.
        let md = "\
# Real

content

#tag-like
more content

## Section
ok";
        let c = HeadingsChunker::new(10_000);
        let chunks = c.chunk(md, &p());
        assert_eq!(chunks.len(), 2, "tag-like '#tag' must not be a heading");
        assert!(chunks[0].text.contains("#tag-like"));
    }

    // ---- TreeSitterChunker tests (Rust) ----

    fn ts_rust() -> TreeSitterChunker {
        TreeSitterChunker::new_rust(50_000, LineChunker::new(60, 10, 50_000)).unwrap()
    }

    #[test]
    fn ts_extracts_top_level_fn() {
        let src = "\
fn alpha() {
    println!(\"a\");
}

fn beta(x: u32) -> u32 {
    x + 1
}
";
        let c = ts_rust();
        let chunks = c.chunk(src, &p());
        assert_eq!(chunks.len(), 2);
        assert_eq!(chunks[0].kind.as_deref(), Some("fn"));
        assert_eq!(chunks[0].name.as_deref(), Some("alpha"));
        assert_eq!(chunks[1].kind.as_deref(), Some("fn"));
        assert_eq!(chunks[1].name.as_deref(), Some("beta"));
    }

    #[test]
    fn ts_extracts_struct_and_enum() {
        let src = "\
struct Foo {
    x: u32,
}

enum Bar {
    A,
    B(u32),
}
";
        let c = ts_rust();
        let chunks = c.chunk(src, &p());
        assert_eq!(chunks.len(), 2);
        let kinds: Vec<&str> = chunks.iter().filter_map(|c| c.kind.as_deref()).collect();
        let names: Vec<&str> = chunks.iter().filter_map(|c| c.name.as_deref()).collect();
        assert_eq!(kinds, vec!["struct", "enum"]);
        assert_eq!(names, vec!["Foo", "Bar"]);
    }

    #[test]
    fn ts_impl_methods_get_qualified_name() {
        let src = "\
struct Foo;

impl Foo {
    fn new() -> Self { Foo }
    fn bar(&self) -> u32 { 42 }
}
";
        let c = ts_rust();
        let chunks = c.chunk(src, &p());
        // Expect: struct Foo + 2 methods (new, bar). impl block itself is NOT
        // a separate chunk — its content lives in the method chunks.
        assert_eq!(
            chunks.len(),
            3,
            "got: {:?}",
            chunks
                .iter()
                .map(|c| (c.kind.clone(), c.name.clone()))
                .collect::<Vec<_>>()
        );
        let methods: Vec<&str> = chunks
            .iter()
            .filter(|c| c.kind.as_deref() == Some("impl_method"))
            .filter_map(|c| c.name.as_deref())
            .collect();
        assert!(methods.contains(&"Foo::new"));
        assert!(methods.contains(&"Foo::bar"));
    }

    #[test]
    fn ts_impl_trait_for_target() {
        // `impl Display for Foo` — methods get `Foo::method` qualification.
        let src = "\
struct Foo;

impl std::fmt::Display for Foo {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        write!(f, \"Foo\")
    }
}
";
        let c = ts_rust();
        let chunks = c.chunk(src, &p());
        let qualified: Vec<String> = chunks
            .iter()
            .filter(|c| c.kind.as_deref() == Some("impl_method"))
            .filter_map(|c| c.name.clone())
            .collect();
        // tree-sitter-rust exposes the target type via the "type" field on
        // impl_item. Most simple Self types come through cleanly.
        assert!(
            qualified
                .iter()
                .any(|n| n.starts_with("Foo::") && n.ends_with("::fmt")),
            "expected Foo::fmt-style qualification, got {:?}",
            qualified
        );
    }

    #[test]
    fn ts_traits_and_types() {
        let src = "\
trait Shape {
    fn area(&self) -> f64;
}

type Pair<T> = (T, T);

const MAX: usize = 100;

static GLOBAL: u32 = 0;
";
        let c = ts_rust();
        let chunks = c.chunk(src, &p());
        let kinds: Vec<&str> = chunks.iter().filter_map(|c| c.kind.as_deref()).collect();
        assert!(kinds.contains(&"trait"));
        assert!(kinds.contains(&"type"));
        assert!(kinds.contains(&"const"));
        assert!(kinds.contains(&"static"));
    }

    #[test]
    fn ts_oversized_fn_splits_with_kind_preserved() {
        // A massive function — should split into multiple chunks, all
        // tagged with the same kind/name.
        let body = "    let x = 1;\n".repeat(500);
        let src = format!("fn huge() {{\n{}}}", body);
        let c = TreeSitterChunker::new_rust(500, LineChunker::new(20, 0, 500)).unwrap();
        let chunks = c.chunk(&src, &p());
        assert!(chunks.len() > 1, "expected multi-chunk split");
        for ch in &chunks {
            assert_eq!(ch.kind.as_deref(), Some("fn"));
            assert_eq!(ch.name.as_deref(), Some("huge"));
        }
    }

    #[test]
    fn ts_parse_error_falls_back_to_lines() {
        // Genuinely broken — tree-sitter produces a tree with errors, but
        // some structure usually survives. For a file that's *just*
        // garbage with no recognizable items at all, we expect fallback.
        let src = "this is not rust at all just random text without any structure";
        let c = ts_rust();
        let chunks = c.chunk(src, &p());
        // We get at least one chunk (from fallback or partial parse).
        assert!(!chunks.is_empty());
    }

    #[test]
    fn ts_empty_source_no_chunks() {
        let c = ts_rust();
        assert!(c.chunk("", &p()).is_empty());
    }

    // ---- TreeSitterChunker (Python) tests ----

    fn ts_python() -> TreeSitterChunker {
        TreeSitterChunker::new_python(50_000, LineChunker::new(60, 10, 50_000)).unwrap()
    }

    #[test]
    fn ts_py_top_level_fn() {
        let src = "\
def alpha():
    print('a')

def beta(x):
    return x + 1
";
        let c = ts_python();
        let chunks = c.chunk(src, &p());
        assert_eq!(chunks.len(), 2);
        assert_eq!(chunks[0].kind.as_deref(), Some("fn"));
        assert_eq!(chunks[0].name.as_deref(), Some("alpha"));
        assert_eq!(chunks[1].kind.as_deref(), Some("fn"));
        assert_eq!(chunks[1].name.as_deref(), Some("beta"));
    }

    #[test]
    fn ts_py_class_with_methods() {
        let src = "\
class Foo:
    def __init__(self, x):
        self.x = x

    def bar(self):
        return self.x
";
        let c = ts_python();
        let chunks = c.chunk(src, &p());
        // Expect: class Foo chunk + 2 method chunks (__init__, bar).
        assert_eq!(
            chunks.len(),
            3,
            "got: {:?}",
            chunks
                .iter()
                .map(|c| (c.kind.clone(), c.name.clone()))
                .collect::<Vec<_>>()
        );
        // Class
        assert!(chunks
            .iter()
            .any(|c| c.kind.as_deref() == Some("class") && c.name.as_deref() == Some("Foo")));
        // Qualified methods
        let methods: Vec<&str> = chunks
            .iter()
            .filter(|c| c.kind.as_deref() == Some("method"))
            .filter_map(|c| c.name.as_deref())
            .collect();
        assert!(methods.contains(&"Foo.__init__"));
        assert!(methods.contains(&"Foo.bar"));
    }

    #[test]
    fn ts_py_decorated_fn() {
        let src = "\
@staticmethod
def helper(x):
    return x * 2
";
        let c = ts_python();
        let chunks = c.chunk(src, &p());
        assert_eq!(chunks.len(), 1);
        assert_eq!(chunks[0].kind.as_deref(), Some("fn"));
        assert_eq!(chunks[0].name.as_deref(), Some("helper"));
        // The chunk text must include the decorator line.
        assert!(chunks[0].text.contains("@staticmethod"));
        assert!(chunks[0].text.contains("def helper"));
    }

    #[test]
    fn ts_py_decorated_class_with_methods() {
        let src = "\
@dataclass
class Point:
    x: float
    y: float

    def distance(self):
        return (self.x ** 2 + self.y ** 2) ** 0.5
";
        let c = ts_python();
        let chunks = c.chunk(src, &p());
        // Expect: class Point (the decorated one) + method Point.distance.
        let class_chunk = chunks
            .iter()
            .find(|c| c.kind.as_deref() == Some("class") && c.name.as_deref() == Some("Point"));
        assert!(
            class_chunk.is_some(),
            "no class chunk in {:?}",
            chunks
                .iter()
                .map(|c| (c.kind.clone(), c.name.clone()))
                .collect::<Vec<_>>()
        );
        assert!(class_chunk.unwrap().text.contains("@dataclass"));
        assert!(chunks
            .iter()
            .any(|c| c.kind.as_deref() == Some("method")
                && c.name.as_deref() == Some("Point.distance")));
    }

    #[test]
    fn ts_py_module_with_only_imports_and_script_yields_imports_chunk() {
        // No def/class — the imports prelude is captured as kind="imports";
        // the script-level statements (assignment + call) get skipped by
        // emit_python_node but the imports chunk alone keeps the file
        // out of the line-chunker fallback path.
        let src = "\
import os
import sys

X = 10
print(\"hello\")
";
        let c = ts_python();
        let chunks = c.chunk(src, &p());
        assert!(!chunks.is_empty());
        assert!(chunks.iter().any(|c| c.kind.as_deref() == Some("imports")));
    }

    #[test]
    fn ts_py_imports_grouped() {
        let src = r#"
import os
import sys
from collections import defaultdict

def helper():
    pass
"#;
        let c = ts_python();
        let chunks = c.chunk(src, &p());
        let imports = chunks
            .iter()
            .find(|c| c.kind.as_deref() == Some("imports"))
            .expect("imports chunk missing");
        assert!(imports.text.contains("import os"));
        assert!(imports.text.contains("import sys"));
        assert!(imports.text.contains("from collections import defaultdict"));
        // helper() also indexed
        assert!(chunks
            .iter()
            .any(|c| c.kind.as_deref() == Some("fn") && c.name.as_deref() == Some("helper")));
    }

    #[test]
    fn ts_py_imports_only_file() {
        // No def/class — imports chunk alone should suffice (no fallback).
        let src = r#"
import os
from typing import List
"#;
        let c = ts_python();
        let chunks = c.chunk(src, &p());
        assert_eq!(chunks.len(), 1);
        assert_eq!(chunks[0].kind.as_deref(), Some("imports"));
    }

    #[test]
    fn ts_py_oversized_class_sub_splits() {
        // Class body bigger than max_chars — fallback line-chunker splits it.
        let body = "        x = 1\n".repeat(500);
        let src = format!("class Big:\n{}", body);
        let c = TreeSitterChunker::new_python(500, LineChunker::new(20, 0, 500)).unwrap();
        let chunks = c.chunk(&src, &p());
        let class_chunks: Vec<&Chunk> = chunks
            .iter()
            .filter(|ch| ch.kind.as_deref() == Some("class"))
            .collect();
        assert!(
            class_chunks.len() > 1,
            "expected class to sub-split, got {}",
            class_chunks.len()
        );
        for ch in &class_chunks {
            assert_eq!(ch.name.as_deref(), Some("Big"));
        }
    }

    // ---- TreeSitterChunker (Dart) tests ----

    fn ts_dart() -> TreeSitterChunker {
        TreeSitterChunker::new_dart(50_000, LineChunker::new(60, 10, 50_000)).unwrap()
    }

    #[test]
    fn ts_dart_class_with_methods() {
        let src = "\
class Counter {
  int value = 0;

  void increment() {
    value++;
  }

  int doubled() => value * 2;
}
";
        let c = ts_dart();
        let chunks = c.chunk(src, &p());
        // Expect: class + at least one method chunk.
        assert!(chunks.iter().any(|c|
            c.kind.as_deref() == Some("class") && c.name.as_deref() == Some("Counter")),
            "no class chunk in {:?}",
            chunks.iter().map(|c| (c.kind.clone(), c.name.clone())).collect::<Vec<_>>());
        let methods: Vec<&str> = chunks
            .iter()
            .filter(|c| c.kind.as_deref() == Some("method"))
            .filter_map(|c| c.name.as_deref())
            .collect();
        // At least one of increment/doubled should be captured.
        assert!(
            methods.iter().any(|m| m.starts_with("Counter.")),
            "no Counter.* methods in {:?}",
            methods
        );
    }

    #[test]
    fn ts_dart_top_level_function() {
        // Something the 0.0.4 grammar could not do reliably — its top-level
        // coverage was fuzzy enough that standalone functions fell through
        // to line-window chunking. The 0.2 grammar gives them a
        // `function_declaration` with a named signature, so they get a real
        // syntactic anchor like every other language.
        let src = "\
int add(int a, int b) => a + b;

Future<void> main() async {
  print(add(1, 2));
}
";
        let c = ts_dart();
        let chunks = c.chunk(src, &p());
        let fns: Vec<&str> = chunks
            .iter()
            .filter(|c| c.kind.as_deref() == Some("fn"))
            .filter_map(|c| c.name.as_deref())
            .collect();
        assert!(fns.contains(&"add"), "top-level `add` missing from {fns:?}");
        assert!(
            fns.contains(&"main"),
            "top-level `main` missing from {fns:?}"
        );
    }

    #[test]
    fn ts_dart_constructor_is_named() {
        // Constructors sit under `declaration` rather than
        // `method_declaration`, which is why the name lookup searches for
        // any `*_signature` carrying a `name` field instead of matching on
        // node kinds.
        let src = "class Point {\n  final int x;\n  Point(this.x);\n}\n";
        let c = ts_dart();
        let chunks = c.chunk(src, &p());
        let names: Vec<&str> = chunks.iter().filter_map(|c| c.name.as_deref()).collect();
        assert!(
            names.contains(&"Point.Point"),
            "constructor not named in {names:?}"
        );
    }

    #[test]
    fn ts_dart_enum() {
        let src = "\
enum Color {
  red,
  green,
  blue,
}
";
        let c = ts_dart();
        let chunks = c.chunk(src, &p());
        assert!(
            chunks
                .iter()
                .any(|c| c.kind.as_deref() == Some("enum") && c.name.as_deref() == Some("Color")),
            "got {:?}",
            chunks
                .iter()
                .map(|c| (c.kind.clone(), c.name.clone()))
                .collect::<Vec<_>>()
        );
    }

    #[test]
    fn ts_dart_mixin() {
        let src = "\
mixin Walkable {
  void walk() {}
}
";
        let c = ts_dart();
        let chunks = c.chunk(src, &p());
        assert!(chunks.iter().any(|c| c.kind.as_deref() == Some("mixin")));
    }

    #[test]
    fn ts_dart_empty_no_chunks() {
        let c = ts_dart();
        assert!(c.chunk("", &p()).is_empty());
    }

    // ---- TreeSitterChunker (C++) tests ----

    fn ts_cpp() -> TreeSitterChunker {
        TreeSitterChunker::new_cpp(50_000, LineChunker::new(60, 10, 50_000)).unwrap()
    }

    #[test]
    fn ts_cpp_top_level_fn() {
        let src = "\
int square(int x) {
    return x * x;
}

void greet() {
    printf(\"hi\");
}
";
        let c = ts_cpp();
        let chunks = c.chunk(src, &p());
        let fns: Vec<&str> = chunks
            .iter()
            .filter(|ch| ch.kind.as_deref() == Some("fn"))
            .filter_map(|ch| ch.name.as_deref())
            .collect();
        assert!(fns.contains(&"square"), "got {:?}", fns);
        assert!(fns.contains(&"greet"), "got {:?}", fns);
    }

    #[test]
    fn ts_cpp_class_inline_methods() {
        let src = "\
class Foo {
public:
    int x;
    int get_x() const { return x; }
    void set_x(int v) { x = v; }
};
";
        let c = ts_cpp();
        let chunks = c.chunk(src, &p());
        // Class itself
        assert!(
            chunks
                .iter()
                .any(|c| c.kind.as_deref() == Some("class") && c.name.as_deref() == Some("Foo")),
            "no class Foo in {:?}",
            chunks
                .iter()
                .map(|c| (c.kind.clone(), c.name.clone()))
                .collect::<Vec<_>>()
        );
        // Inline methods qualified Foo::get_x, Foo::set_x
        let methods: Vec<String> = chunks
            .iter()
            .filter(|c| c.kind.as_deref() == Some("method"))
            .filter_map(|c| c.name.clone())
            .collect();
        assert!(
            methods.iter().any(|n| n == "Foo::get_x"),
            "got {:?}",
            methods
        );
        assert!(
            methods.iter().any(|n| n == "Foo::set_x"),
            "got {:?}",
            methods
        );
    }

    #[test]
    fn ts_cpp_outline_method_keeps_qualified_name() {
        // Outline definition: name is `Foo::bar` via qualified_identifier.
        // Even at top level (parent=None), the qualified form survives.
        let src = "\
class Foo { void bar(); };

void Foo::bar() {
    return;
}
";
        let c = ts_cpp();
        let chunks = c.chunk(src, &p());
        // Should have a top-level fn chunk with name = "Foo::bar".
        let fns: Vec<String> = chunks
            .iter()
            .filter(|c| c.kind.as_deref() == Some("fn"))
            .filter_map(|c| c.name.clone())
            .collect();
        assert!(
            fns.iter().any(|n| n == "Foo::bar"),
            "expected fn Foo::bar in {:?}",
            fns
        );
    }

    #[test]
    fn ts_cpp_struct_and_enum_and_namespace() {
        let src = "\
namespace util {
    struct Point { int x; int y; };
    enum Mode { On, Off };
}
";
        let c = ts_cpp();
        let chunks = c.chunk(src, &p());
        let kinds: Vec<&str> = chunks.iter().filter_map(|c| c.kind.as_deref()).collect();
        assert!(kinds.contains(&"namespace"));
        assert!(kinds.contains(&"struct"));
        assert!(kinds.contains(&"enum"));
    }

    #[test]
    fn ts_cpp_template_function() {
        let src = "\
template<typename T>
T identity(T x) {
    return x;
}
";
        let c = ts_cpp();
        let chunks = c.chunk(src, &p());
        let fn_chunk = chunks.iter().find(|c| c.kind.as_deref() == Some("fn"));
        assert!(
            fn_chunk.is_some(),
            "no fn chunk in {:?}",
            chunks
                .iter()
                .map(|c| (c.kind.clone(), c.name.clone()))
                .collect::<Vec<_>>()
        );
        let fc = fn_chunk.unwrap();
        assert_eq!(fc.name.as_deref(), Some("identity"));
        // Template parameters are included in the chunk text.
        assert!(fc.text.contains("template"));
    }

    #[test]
    fn ts_cpp_empty_no_chunks() {
        let c = ts_cpp();
        assert!(c.chunk("", &p()).is_empty());
    }

    // ---- TreeSitterChunker (TypeScript) tests ----

    fn ts_ts() -> TreeSitterChunker {
        TreeSitterChunker::new_typescript(50_000, LineChunker::new(60, 10, 50_000)).unwrap()
    }

    #[test]
    fn ts_ts_top_level_fn_and_export() {
        let src = r#"
function helper(x: number): number {
    return x * 2;
}

export function exported(y: string): string {
    return y;
}
"#;
        let c = ts_ts();
        let chunks = c.chunk(src, &p());
        let fns: Vec<&str> = chunks
            .iter()
            .filter(|c| c.kind.as_deref() == Some("fn"))
            .filter_map(|c| c.name.as_deref())
            .collect();
        assert!(fns.contains(&"helper"), "got {:?}", fns);
        assert!(fns.contains(&"exported"), "got {:?}", fns);
        // Exported fn's chunk text must include the `export` keyword.
        let exported = chunks
            .iter()
            .find(|c| c.name.as_deref() == Some("exported"))
            .unwrap();
        assert!(exported.text.contains("export"));
    }

    #[test]
    fn ts_ts_class_with_methods() {
        let src = r#"
class Foo {
    x: number = 0;
    constructor(x: number) { this.x = x; }
    getX(): number { return this.x; }
}
"#;
        let c = ts_ts();
        let chunks = c.chunk(src, &p());
        assert!(chunks
            .iter()
            .any(|c| c.kind.as_deref() == Some("class") && c.name.as_deref() == Some("Foo")));
        let methods: Vec<String> = chunks
            .iter()
            .filter(|c| c.kind.as_deref() == Some("method"))
            .filter_map(|c| c.name.clone())
            .collect();
        assert!(
            methods.iter().any(|n| n == "Foo.constructor"),
            "got {:?}",
            methods
        );
        assert!(methods.iter().any(|n| n == "Foo.getX"), "got {:?}", methods);
    }

    #[test]
    fn ts_ts_interface_type_enum() {
        let src = r#"
interface Shape { area(): number; }
type Pair<T> = [T, T];
enum Color { Red, Green, Blue }
"#;
        let c = ts_ts();
        let chunks = c.chunk(src, &p());
        let kinds: Vec<&str> = chunks.iter().filter_map(|c| c.kind.as_deref()).collect();
        assert!(kinds.contains(&"interface"));
        assert!(kinds.contains(&"type"));
        assert!(kinds.contains(&"enum"));
    }

    #[test]
    fn ts_ts_empty_no_chunks() {
        let c = ts_ts();
        assert!(c.chunk("", &p()).is_empty());
    }

    // ---- TreeSitterChunker (JavaScript) tests ----

    fn ts_js() -> TreeSitterChunker {
        TreeSitterChunker::new_javascript(50_000, LineChunker::new(60, 10, 50_000)).unwrap()
    }

    #[test]
    fn ts_js_fn_and_class() {
        let src = r#"
function add(a, b) {
    return a + b;
}

class Counter {
    constructor() { this.n = 0; }
    increment() { this.n += 1; }
}
"#;
        let c = ts_js();
        let chunks = c.chunk(src, &p());
        let kinds_names: Vec<(String, String)> = chunks
            .iter()
            .filter_map(|c| Some((c.kind.clone()?, c.name.clone()?)))
            .collect();
        assert!(
            kinds_names.iter().any(|(k, n)| k == "fn" && n == "add"),
            "got {:?}",
            kinds_names
        );
        assert!(
            kinds_names
                .iter()
                .any(|(k, n)| k == "class" && n == "Counter"),
            "got {:?}",
            kinds_names
        );
        assert!(
            kinds_names
                .iter()
                .any(|(k, n)| k == "method" && n == "Counter.increment"),
            "got {:?}",
            kinds_names
        );
    }

    // ---- TreeSitterChunker (Go) tests ----

    fn ts_go() -> TreeSitterChunker {
        TreeSitterChunker::new_go(50_000, LineChunker::new(60, 10, 50_000)).unwrap()
    }

    #[test]
    fn ts_go_top_level_fn() {
        let src = r#"
package main

func square(x int) int {
    return x * x
}

func main() {
    println(square(3))
}
"#;
        let c = ts_go();
        let chunks = c.chunk(src, &p());
        let fns: Vec<&str> = chunks
            .iter()
            .filter(|c| c.kind.as_deref() == Some("fn"))
            .filter_map(|c| c.name.as_deref())
            .collect();
        assert!(fns.contains(&"square"));
        assert!(fns.contains(&"main"));
    }

    #[test]
    fn ts_go_struct_and_method_value_receiver() {
        let src = r#"
package main

type Point struct {
    X int
    Y int
}

func (p Point) Sum() int {
    return p.X + p.Y
}
"#;
        let c = ts_go();
        let chunks = c.chunk(src, &p());
        assert!(chunks
            .iter()
            .any(|c| c.kind.as_deref() == Some("struct") && c.name.as_deref() == Some("Point")));
        // method qualified with value receiver
        assert!(
            chunks
                .iter()
                .any(|c| c.kind.as_deref() == Some("method")
                    && c.name.as_deref() == Some("Point.Sum")),
            "got {:?}",
            chunks
                .iter()
                .map(|c| (c.kind.clone(), c.name.clone()))
                .collect::<Vec<_>>()
        );
    }

    #[test]
    fn ts_go_pointer_receiver_method() {
        let src = r#"
package main

type Counter struct {
    n int
}

func (c *Counter) Increment() {
    c.n++
}
"#;
        let c = ts_go();
        let chunks = c.chunk(src, &p());
        // Pointer-receiver method should still qualify with base type name.
        assert!(
            chunks.iter().any(|c| c.kind.as_deref() == Some("method")
                && c.name.as_deref() == Some("Counter.Increment")),
            "got {:?}",
            chunks
                .iter()
                .map(|c| (c.kind.clone(), c.name.clone()))
                .collect::<Vec<_>>()
        );
    }

    #[test]
    fn ts_go_interface_and_type_alias() {
        let src = r#"
package main

type Reader interface {
    Read(p []byte) (int, error)
}

type Bytes []byte
"#;
        let c = ts_go();
        let chunks = c.chunk(src, &p());
        let kinds_names: Vec<(String, String)> = chunks
            .iter()
            .filter_map(|c| Some((c.kind.clone()?, c.name.clone()?)))
            .collect();
        assert!(
            kinds_names
                .iter()
                .any(|(k, n)| k == "interface" && n == "Reader"),
            "got {:?}",
            kinds_names
        );
        assert!(
            kinds_names.iter().any(|(k, n)| k == "type" && n == "Bytes"),
            "got {:?}",
            kinds_names
        );
    }

    #[test]
    fn ts_go_multi_type_block() {
        // `type ( ... )` block declaring several types in one declaration.
        let src = r#"
package main

type (
    Email string
    Age   int
    Name  string
)
"#;
        let c = ts_go();
        let chunks = c.chunk(src, &p());
        let names: Vec<&str> = chunks
            .iter()
            .filter(|c| c.kind.as_deref() == Some("type"))
            .filter_map(|c| c.name.as_deref())
            .collect();
        for expected in ["Email", "Age", "Name"] {
            assert!(
                names.contains(&expected),
                "missing {} in {:?}",
                expected,
                names
            );
        }
    }

    #[test]
    fn ts_go_empty_no_chunks() {
        let c = ts_go();
        assert!(c.chunk("", &p()).is_empty());
    }

    // ---- TreeSitterChunker (C#) tests ----

    fn ts_cs() -> TreeSitterChunker {
        TreeSitterChunker::new_csharp(50_000, LineChunker::new(60, 10, 50_000)).unwrap()
    }

    #[test]
    fn ts_cs_class_with_methods() {
        let src = r#"
public class Foo {
    public int X { get; set; }

    public Foo(int x) {
        X = x;
    }

    public int GetX() {
        return X;
    }
}
"#;
        let c = ts_cs();
        let chunks = c.chunk(src, &p());
        assert!(
            chunks
                .iter()
                .any(|c| c.kind.as_deref() == Some("class") && c.name.as_deref() == Some("Foo")),
            "got {:?}",
            chunks
                .iter()
                .map(|c| (c.kind.clone(), c.name.clone()))
                .collect::<Vec<_>>()
        );
        let methods: Vec<String> = chunks
            .iter()
            .filter(|c| c.kind.as_deref() == Some("method"))
            .filter_map(|c| c.name.clone())
            .collect();
        assert!(
            methods.iter().any(|n| n == "Foo.Foo"),
            "no Foo.Foo (ctor) in {:?}",
            methods
        );
        assert!(
            methods.iter().any(|n| n == "Foo.GetX"),
            "no Foo.GetX in {:?}",
            methods
        );
    }

    #[test]
    fn ts_cs_interface_struct_enum_record() {
        let src = r#"
public interface IShape { double Area(); }
public struct Point { public int X; public int Y; }
public enum Color { Red, Green, Blue }
public record Person(string Name, int Age);
"#;
        let c = ts_cs();
        let chunks = c.chunk(src, &p());
        let kinds_names: Vec<(String, String)> = chunks
            .iter()
            .filter_map(|c| Some((c.kind.clone()?, c.name.clone()?)))
            .collect();
        assert!(
            kinds_names
                .iter()
                .any(|(k, n)| k == "interface" && n == "IShape"),
            "got {:?}",
            kinds_names
        );
        assert!(
            kinds_names
                .iter()
                .any(|(k, n)| k == "struct" && n == "Point"),
            "got {:?}",
            kinds_names
        );
        assert!(
            kinds_names.iter().any(|(k, n)| k == "enum" && n == "Color"),
            "got {:?}",
            kinds_names
        );
        assert!(
            kinds_names
                .iter()
                .any(|(k, n)| k == "record" && n == "Person"),
            "got {:?}",
            kinds_names
        );
    }

    #[test]
    fn ts_cs_namespace_descends() {
        let src = r#"
namespace MyApp.Domain {
    public class Order {
        public int Id;
        public void Submit() { }
    }
}
"#;
        let c = ts_cs();
        let chunks = c.chunk(src, &p());
        // Namespace is NOT emitted as a chunk (would duplicate everything);
        // but we descend so the class Order shows up.
        assert!(
            chunks
                .iter()
                .any(|c| c.kind.as_deref() == Some("class") && c.name.as_deref() == Some("Order")),
            "got {:?}",
            chunks
                .iter()
                .map(|c| (c.kind.clone(), c.name.clone()))
                .collect::<Vec<_>>()
        );
        assert!(chunks
            .iter()
            .any(|c| c.kind.as_deref() == Some("method")
                && c.name.as_deref() == Some("Order.Submit")));
    }

    #[test]
    fn ts_cs_empty_no_chunks() {
        let c = ts_cs();
        assert!(c.chunk("", &p()).is_empty());
    }

    // ---- TreeSitterChunker (Java) tests ----

    fn ts_java() -> TreeSitterChunker {
        TreeSitterChunker::new_java(50_000, LineChunker::new(60, 10, 50_000)).unwrap()
    }

    #[test]
    fn ts_java_class_with_methods() {
        let src = r#"
public class Counter {
    private int n;

    public Counter() {
        this.n = 0;
    }

    public void increment() {
        n++;
    }

    public int value() {
        return n;
    }
}
"#;
        let c = ts_java();
        let chunks = c.chunk(src, &p());
        assert!(chunks
            .iter()
            .any(|c| c.kind.as_deref() == Some("class") && c.name.as_deref() == Some("Counter")));
        let methods: Vec<String> = chunks
            .iter()
            .filter(|c| c.kind.as_deref() == Some("method"))
            .filter_map(|c| c.name.clone())
            .collect();
        assert!(
            methods.iter().any(|n| n == "Counter.Counter"),
            "missing Counter.Counter in {:?}",
            methods
        );
        assert!(
            methods.iter().any(|n| n == "Counter.increment"),
            "missing Counter.increment in {:?}",
            methods
        );
        assert!(
            methods.iter().any(|n| n == "Counter.value"),
            "missing Counter.value in {:?}",
            methods
        );
    }

    #[test]
    fn ts_java_interface_and_enum() {
        let src = r#"
public interface Shape {
    double area();
}

public enum Status {
    ACTIVE, INACTIVE
}
"#;
        let c = ts_java();
        let chunks = c.chunk(src, &p());
        let kinds_names: Vec<(String, String)> = chunks
            .iter()
            .filter_map(|c| Some((c.kind.clone()?, c.name.clone()?)))
            .collect();
        assert!(kinds_names
            .iter()
            .any(|(k, n)| k == "interface" && n == "Shape"));
        assert!(kinds_names
            .iter()
            .any(|(k, n)| k == "enum" && n == "Status"));
    }

    #[test]
    fn ts_java_record() {
        let src = r#"
public record Point(int x, int y) {}
"#;
        let c = ts_java();
        let chunks = c.chunk(src, &p());
        assert!(
            chunks
                .iter()
                .any(|c| c.kind.as_deref() == Some("record") && c.name.as_deref() == Some("Point")),
            "got {:?}",
            chunks
                .iter()
                .map(|c| (c.kind.clone(), c.name.clone()))
                .collect::<Vec<_>>()
        );
    }

    #[test]
    fn ts_java_empty_no_chunks() {
        let c = ts_java();
        assert!(c.chunk("", &p()).is_empty());
    }
}
