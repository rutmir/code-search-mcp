# Changelog

History of significant changes. Newest at the top. Dates are when work landed locally; this project doesn't tag releases yet.

## 2026-06-11 (docs fix)

### `examples/CLAUDE.md` template — push project-level instruction above auto-memory

Added `examples/CLAUDE.md` as a copy-and-edit template. Drop it at the project root and Claude Code reads it at session start, **above** MCP tool descriptions and user-level auto-memory hints in its instruction hierarchy. The template tells Claude to prefer `code_search` over Grep/Read/Bash for any project-content question and lists genuine fall-back cases (known file paths, exact-bytes ops, non-indexed paths).

Empirically observed problem this solves: even with the tool description rewritten to "PREFERRED first-line search", Claude Code was still reaching for `Read docs/ROADMAP.md` / `git log` / `sed` for navigation queries when the user's auto-memory had a "start with docs/INDEX.md" hint. A project-level `CLAUDE.md` sits above that hint in the precedence stack and lets the tool description win.

README's "Wire to Claude Code" section now includes the `cp examples/CLAUDE.md ...` step.

### MCP tool description: "preferred first-line search"

Rewrote the `code_search` tool description from a narrowly code-focused phrasing ("Best for: find code that does X") to an explicit broad-first-line framing: "PREFERRED first-line search over this project's indexed content. Use BEFORE Grep / Glob / iterative Read for any question about what's in the project — source code, markdown documentation, config files, READMEs, CHANGELOGs, ROADMAPs, design notes."

The earlier description led Claude Code to skip `code_search` for project-orientation queries (e.g. "where are we, what active tasks do we have") and reach for Grep / file reads instead. The new description names the doc/config use cases explicitly, frames token savings as a reason to prefer the tool, and lists genuine fall-back cases (known file paths, non-indexed content, exact-bytes ops).

Project-aware aliasing: filter examples now mention `lang = "markdown"` for docs-only and `path = "docs/"` for scoping, making it more obvious that the same tool works for documentation lookups.

### Project-scope MCP config path

Docs and example corrected: the project-scope MCP config goes at **`<project>/.mcp.json` (project root)**, not `<project>/.claude/mcp.json`. Project-scoped MCP servers also require explicit approval the first time via `claude` interactive (security against repo-supplied malicious configs). Verify with `claude mcp list`.

## 2026-06-11

### Config-change detection + tantivy-aware cache

**Two-level config fingerprint in marker.** The Qdrant marker point now carries `config_hard` (chunking + embedding fingerprint — anything that changes chunk identity) and `config_soft` (languages + exclude + gitignore — anything that changes the file set). At startup the indexer compares against the live config:
- `Fresh` → write marker
- `Match` → silent
- `SoftChanged` → warn + reindex (stale-detection cleans up)
- `HardChanged` → **auto-clear** Qdrant collection + tantivy directory, recreate, write new marker, reindex from scratch (loud WARN log)
- Project-identity mismatch → hard fail

Resolves the silent-corruption case where changing `chunking.max_chunk_chars` would coexist old chunks (5000-char boundaries) with new ones (8000-char boundaries) under different `chunk_uuid`s.

**Tantivy-aware cache.** `Bm25Index::list_indexed_files()` enumerates files present in the local tantivy. `indexer::run`'s file→sha cache (built from Qdrant scroll) is intersected with this set. Files in Qdrant but missing from tantivy get reprocessed → refills tantivy locally AND re-upserts to Qdrant idempotently.

Resolves the cross-machine sharing bug where a second machine pointed at the same Qdrant collection would see "everything indexed" via cache and skip every file, leaving its local tantivy empty.

### Walker: shell / systemd / env / text-config extensions

New languages (line-window chunker, no tree-sitter):
- `shell` (sh, bash, zsh)
- `systemd` (service, socket, timer, mount, target, path)
- `env` (env)
- `text` (txt, local, example, ini, cfg, conf) — catch-all for sample/override configs

## 2026-06-10

### Tree-sitter for C# + Java

`tree-sitter-c-sharp` and `tree-sitter-java`. C# extracts class / interface / struct / enum / record / delegate, with `method ClassName.method_name`. Namespaces descend without standalone emission. Java extracts class / interface / enum / record (Java 14+) / annotation, methods qualified the same way.

### Tree-sitter for TypeScript + JavaScript + Go

`tree-sitter-typescript` (TSX grammar — superset of plain TS), `tree-sitter-javascript`, `tree-sitter-go`. TS adds interface / type alias / enum support. JS gets the subset. Go normalizes pointer (`*Foo`) and value (`Foo`) receivers to the same `Receiver.method_name` qualification. `export_statement` is unwrapped so the `export` keyword stays in chunk text.

### MCP `notifications/cancelled`

`tools/call` requests now spawn as separate tasks tracked by request id; matching `notifications/cancelled` fires a oneshot that preempts the search via `tokio::select!`. Cancellation latency is sub-millisecond. In-flight HTTP requests to embedding/Qdrant/reranker get dropped as their futures die. The stdin reader was moved into its own task with mpsc bridging to the main loop, avoiding read-cancel-safety pitfalls.

### `[search]` config block

`dense_k`, `sparse_k`, `rerank_top_n`, `rrf_k` now configurable via TOML. Defaults stay quality-first (30/30/30/60). Drop K's to 15-20 for ~30-40 % faster searches at recall cost.

### Sub-directory `.gitignore` honored by PathFilter

`PathFilter::new` now walks the project tree via `ignore::WalkBuilder` and aggregates every `.gitignore` into the matcher. Nested ones (e.g., `crates/foo/.gitignore`) work alongside the root one. New `.gitignore` files added during watcher operation require restart.

### Python imports grouping

Pre-pass in the Python emitter collects the leading run of top-level imports (`import_statement` / `import_from_statement` / `future_import_statement`) into one chunk with `kind = "imports"`. Files that previously fell back to line-window because they had only imports + module-level statements now produce a proper imports chunk.

### Tree-sitter for Dart + C++

`tree-sitter-cpp` (full coverage: class / struct / union / enum / namespace / templates / outline `Type::method` definitions). `tree-sitter-dart` 0.0.4 (class with methods, mixin, enum, extension; top-level standalone functions fall back to line-window due to old grammar).

### Tree-sitter for Python

`tree-sitter-python`. Top-level fn / class with `@decorator` preserved in chunk text. Class methods qualified as `Class.method_name`. The class is also emitted as a whole.

### Tree-sitter for Rust + `kind` / `name` metadata

`tree-sitter-rust` for AST-aware chunks. New `Chunk { kind, name }` fields propagated through Qdrant payload (via `DenseHit`) and BM25 candidate fill-in (one Qdrant retrieve-by-IDs call patches kind/name for sparse-only candidates that BM25 schema doesn't store). MCP / CLI result formatter prints `[lang] fn Foo::bar` headers so the LLM sees syntactic anchors up front.

### `serve` + `watch` integration

`serve` mode now spawns a background watcher task if `[watcher].enabled = true` (default). Single process holds both: MCP request loop on the main task, watcher on `tokio::spawn`. Tantivy's N-readers-1-writer permits coexistence (search opens `Bm25Search` read-only per query; watcher owns `Bm25Index` writer). Clean abort + await on serve loop exit.

### Stage 2 — File watcher

`watch` subcommand: initial sync via `indexer::run` (cheap via sha cache), then debounced `notify` event loop. Per-event dispatch: file present + indexable → `process_one_file`; gone → `delete_file_from_indexes`. `notify-debouncer-full` collapses formatter/save-storm bursts. Ctrl-C clean shutdown.

**Fix**: project.root canonicalized at config load. notify gives absolute event paths but walker historically used `WalkBuilder::new(".")` with relative paths; the strip_prefix mismatch made the watcher write Qdrant payloads with absolute `file=` keys, orphaning them from the walker's relative-key cache. Canonicalize means both code paths agree on absolute paths.

### Markdown heading-aware chunker + per-language config

`[chunking.per_language.<lang>]` blocks. `HeadingsChunker` cuts on H1/H2 ATX boundaries, fenced-block-aware (so `# python comment` inside ``` ... ``` doesn't trigger). Result: markdown chunks aligned to section boundaries instead of arbitrary 60-line windows.

`ChunkerSet` dispatch picks per-language strategy at index time.

### Reranker: bge-reranker-v2-m3, throughput-aware timeout, two-stage rerank

Switched from `jina-reranker-v2-base-multilingual` (1024 native ctx, severe quality cap) to `BAAI/bge-reranker-v2-m3` (8192 native ctx). jina-v3 verified broken in mainline llama.cpp per [issue #17189](https://github.com/ggml-org/llama.cpp/issues/17189) — loads cleanly but classifier head outputs garbage scores.

**Throughput-aware per-request timeout** in `reranker::Client`: EWMA on observed chars/sec across successful calls, `timeout = base + safety × (total_chars / observed_chars_per_sec)`. No static `timeout_secs` to guess at every hardware change.

**Two-stage rerank**: only the top-N (default 30) by RRF go through the cross-encoder; the tail keeps RRF score. Cap on reranker tokens regardless of merged candidate count.

**Graceful fallback**: if reranker fails (timeout, ctx overflow, server crash), search falls back to RRF-sorted results with a warn log. Search never errors out to the LLM.

**Document truncation** (`max_document_chars`, default 8000): each candidate truncated client-side before reranker call. Sized to fit bge-v2-m3 at `ctx=8192` plus query+special tokens. Drop to 4000 for ~2× faster rerank on memory-bandwidth-bound CPUs.

### Stage 3 — MCP serve mode

`serve` subcommand exposing the `code_search` tool over line-delimited JSON-RPC stdio. `initialize` / `tools/list` / `tools/call` / `ping` / `notifications/initialized` handlers. `examples/mcp.json` skeleton for Claude Code integration.

### Stage 1.5 — Search CLI

`search` subcommand: hybrid retrieval pipeline. Query → embed (jina-code-embeddings) → parallel dense (Qdrant) + sparse (BM25/tantivy) → RRF merge → cross-encoder rerank. RRF chosen over normalized weighted-sum because dense (cosine) and BM25 (raw) score distributions are incompatible.

CLI flags: `--lang`, `--path`, `--no-rerank`, `--json`, `-n` / `--limit`.

## 2026-06-09

### Indexer adaptive batching (AIMD) + reliability

**Adaptive batching** in `src/adaptive_batcher.rs`. Replaces static `batch_size` / `max_input_chars` / `timeout_secs` with runtime self-tuning:
- Budget (chars per request) grows by 25 % after `INCREASE_THRESHOLD` consecutive successes (multiplicative-ish increase)
- Halves on any failure (multiplicative decrease)
- Per-request timeout derived from EWMA on observed chars/sec — no `timeout_secs` to guess

Modeled after TCP congestion control. Change `-ub` / `--parallel` on the embedding server, move to a different machine — the client reconverges over the next ~10 batches.

Drove the removal of the internal retry loop from `embedding::Client`: the adaptive layer owns the policy. Error classification (`ServerDown` / `WorkloadTooBig` / `PermanentBad` / `Ambiguous`) directs the response. Ambiguous → quick probe to disambiguate (workload too big vs. server down).

**Probe timeout** for `is_alive_quick` raised from 5 s to 30 s. Single-slot llama.cpp servers can't answer a probe immediately after cancelling the previous request — the slot needs time to drain. 5 s routinely fell inside that drain window and produced false `ServerDown` classifications.

**Qdrant `send_with_retry`**: `upsert_points` / `delete_points` / `delete_by_file` now retry on transient errors (connection closed, 5xx) with exponential backoff (500/1000/2000 ms × 3 attempts). The shared retry helper captures response body on 5xx so Qdrant's diagnostics land in the log.

### `clear` subcommand

`clear [--yes]` drops the Qdrant collection AND removes the tantivy directory. Interactive confirmation (literal "yes" required) unless `--yes`. Order: Qdrant first (atomic delete via API), then tantivy (`fs::remove_dir_all`). If Qdrant succeeds and tantivy fails, retrying `clear` is safe (Qdrant 404 = idempotent OK).

### Stage 1 — Indexer

`index` subcommand: walks the project (ignore/gitignore respected), line-chunks files, embeds via `jina-code-embeddings-0.5b`, upserts to Qdrant + tantivy. Per-file commit; tantivy commit BEFORE Qdrant upsert (Qdrant is the source of truth, written last; crash between the two leaves orphan tantivy chunks that get re-overwritten on next run via `delete_by_file`).

No `index_state.json` file ever existed — Qdrant payload's `file` + `file_sha256` fields are the source of truth. `vs.scroll_files()` rebuilds the file→sha cache at every startup.

**Embedding pooling**: `jina-code-embeddings-0.5b` requires `--pooling last` (Qwen3-0.6B base, decoder-only). `--pooling cls` or `mean` produces plausible-looking 896-dim vectors that are semantically degraded.

### Stage 0 — Config + check

`check` subcommand: validates TOML, pings embedding endpoint, verifies `dimensions` matches the live model output, pings Qdrant root endpoint.

### Project scaffold

CLI structure (clap subcommands), tokio runtime, tracing/EnvFilter for stderr logging (stdout reserved for MCP), reqwest HTTP client, tantivy 0.22, sha2 + uuid for chunk identity.
