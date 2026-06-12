# Example project-level CLAUDE.md

> Copy this file to **`<your-project>/CLAUDE.md`** (project root). Claude Code reads it at session start, and instructions here take precedence over MCP tool descriptions and over auto-memory hints. Use it to make Claude Code prefer `code_search` over Grep / Read / Bash for any project-content query.
>
> Customize the "Project-specific notes" section at the bottom to point at your own canonical files.

---

## Code search → use `code_search` MCP first

For **any** question about this project's content, use the `code_search` MCP tool **BEFORE** Grep / Glob / Read / Bash(`grep`|`sed`|`cat`).

This covers (but is not limited to):

- **Semantic code lookup** — "how does X work", "where is Y implemented", "what calls Z", "show me the implementation of feature F"
- **Documentation** — "what does the ROADMAP say about phase N", "what's in the design notes about subsystem M", "which section explains feature K"
- **Project orientation / navigation** — "where are we", "what are the active tasks", "what's done so far", "what's the current state". These are questions about **project content**; the answer comes from `code_search` over the indexed docs, not from reading specific files blindly
- **Config / architecture** — "where is option O configured", "which modules handle subsystem S"
- **Cross-cutting searches** — "all references to LIBOR", "what depends on module M", "where is constant C used"

### Why this rule

`code_search` indexes the **entire project** — source code (tree-sitter AST chunks with `fn` / `struct` / `impl_method` anchors), markdown documentation (heading-aware chunker — each `## Section` becomes a chunk with the heading as anchor), TOML / YAML / JSON configs, ROADMAPs / CHANGELOGs / STATUS files / design notes.

One tool call returns ranked `file:line` chunks (~1500 tokens, structured with syntactic anchors) — **30-100× cheaper** than iterative grep+read for exploratory queries, with equal or better recall.

### When to fall back to direct file tools

Use Grep / Read / Bash directly only when:

1. **Known specific path**: `Read docs/ROADMAP.md:42-50` — yes. `code_search "ROADMAP what does it say"` → then Read the resulting chunk — no, that's the right flow
2. **Exact-bytes operations**: hex dumps, log tails, line counts, diffs between files
3. **Non-indexed content**: build artifacts, generated files, paths in `[index].exclude` or gitignored (`target/`, `build/`, `*.lock`, etc.)
4. **Trivial shell / git operations**: `git log`, `git status`, `ls`, `wc -l` — cheaper than `code_search` and not about file content

### Workflow pattern for exploratory queries

```
1. Received a user question
2. Is it about project content?  →  YES → code_search "<3-10 word query>"
                                  →  NO  → use direct tools (git/ls/etc.)
3. Optional scope hints:
   - docs only:   code_search "..." {"lang": "markdown"}
   - one subdir:  code_search "..." {"path": "src/feature/"}
4. Top-3-5 hits show where to look. Use kind/name anchors in headers
   (e.g. `fn Foo::bar`, `## Section Title`) to navigate
5. If full file content needed → Read the specific file:line range
6. If cross-references needed → another code_search with a different query
```

**Tip**: concise queries (3-10 words) outperform long descriptions. The reranker sees more signal in a tight phrase.

**Tip**: phrase queries in **English**, even when discussing the project in another language. The reference embedding model (`jina-code-embeddings`) is trained on English-code pairs — the dense leg is noticeably weaker on non-English queries, while identifiers and keywords in the query work equally well either way (BM25 leg).

---

## Project-specific notes

> Replace the lines below with pointers to your project's actual canonical files. Examples:

- `docs/INDEX.md` — top-level docs router. For exploratory queries always run `code_search` first; INDEX is only useful when you already know what you're looking for
- `docs/ROADMAP.md` / `docs/STATUS.md` / `docs/CHANGELOG.md` — all indexed. Search via `code_search "..." {"lang": "markdown"}` for docs-scoped queries
- `crates/<your-crate>/src/` — code with tree-sitter AST chunking; search by symbol name or semantics directly
- `research/` (or wherever your notes live) — markdown with heading-aware indexing
