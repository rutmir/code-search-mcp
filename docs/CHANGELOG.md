# Changelog

History of significant changes. Newest at the top. Dates are when work landed locally; this project doesn't tag releases yet.

## 2026-08-14 — v0.0.5

**⚠ Upgrade note: the first `index` / `serve` start after this upgrade auto-clears and fully rebuilds the index** (BM25 schema revision 2 → 3 for the new stored AST fields; re-embeds the whole corpus — plan for first-index duration).

### `lang` filter was only half-applied

`code_search(query, lang="rust")` pushed the language restriction into Qdrant but not into tantivy, and nothing filtered the merged pool afterwards — so any BM25 hit in any other language survived the merge with a full RRF vote and could outrank the Rust chunk the caller asked for. The restriction is now a hard `Must` clause on the sparse side too, with a post-merge retain as insurance against a payload/schema disagreement.

### `path` filter no longer starves the result set

`path` is a substring match neither store expresses cheaply (Qdrant needs a full-text payload index and then matches *tokens*, not substrings; tantivy's `file` field is raw-tokenized), so it stays a post-retrieval filter. But it used to be applied to a fixed top-30+30 pool: scoping to `docs/` in a repo whose global top-30 happened to be code returned nothing at all. The pool is now widened 10× (capped at 500) whenever a `path` filter is set. Reranking still only sees `rerank_top_n` candidates, so the extra depth costs retrieval bandwidth, not cross-encoder time.

### BM25 schema v3: AST metadata is stored and searched

`kind` and `name` are now stored in tantivy, and `name` is indexed and searched alongside `content` at a 3× boost. Two consequences: a chunk that came in via BM25 alone now carries its own syntactic anchor (`fn Foo::bar`) instead of needing a separate Qdrant round trip per query to fill it in, and a query naming a symbol gets a genuine BM25 signal from the name field rather than relying entirely on the post-merge symbol boost.

### Chunk-level vector reuse

The sha cache is per-file, so editing one function used to re-embed every chunk in the file — 40 embeddings to change one. Chunks now carry a `chunk_sha` payload, and a file being *re*indexed first pulls its stored vectors keyed by that hash; only chunks whose text actually changed go to the embedding server. A chunk that merely shifted down the file matches by content and is reused as-is. Safe because `config_hard` covers the embedding model and dimensions — a model change clears the collection before any vector could be reused under a different model.

### Per-query cost

- `Bm25Search` holds one `IndexReader` for its lifetime instead of building one per call — and per `lookup_content` call, which meant up to 31 reader constructions (each registering a directory watch) in a single search. Content hydration for the rerank head is now one `TermSetQuery`.
- New `SearchContext` owns the Qdrant, embedding and reranker clients plus the tantivy reader for the life of the process. `serve` builds one at startup, so a query no longer constructs a fresh `reqwest` client (new connection pool, new TLS handshake), re-pings the Qdrant root endpoint, or re-fetches the marker. It initializes lazily with retry, so the server still starts — and reports tool-level errors — when Qdrant is down or the index hasn't been built yet.
- Keyword payload indexes are created for `file`, `lang` and `kind`. Every search carries a `must_not kind = marker` clause and often a `lang` restriction, and every reindex issues a `file`-filtered delete.

### `code_read_chunk` MCP tool

Search previews are capped at 600 chars, after which the model had to `Read` the file — spending back much of what the search saved. The new tool returns the untruncated text of the chunks at a `file:start-end` from a previous result, straight from the index, capped at 20k chars per call.

### MCP: progress, version negotiation, cleaner shutdown

- `notifications/progress` is emitted (embedding → retrieving → reranking → finalizing) when the client supplies a `_meta.progressToken`. A quality-first search can run for minutes with no other sign of life.
- `initialize` now negotiates: the server echoes the client's `protocolVersion` when it speaks it (`2025-06-18`, `2025-03-26`, `2024-11-05`), otherwise offers its own. Advertised version moved from `2024-11-05` to `2025-06-18`.
- On stdin EOF the loop used to return immediately, dropping the responses of any `tools/call` already dispatched. Shutdown now drains those responses, bounded by a 5-second grace period.
- The in-flight cancellation registry recovers from a poisoned mutex instead of panicking on every subsequent `tools/call` for the life of the process.

### `watch`: renamed and deleted directories

notify reports `mv src/ old_src/` as a single event on the directory and never mentions its contents, so every chunk under it stayed in both stores until the next full `index` ran stale-detection. A path that no longer exists is now treated as a prefix over the file cache. The scan is gated on non-existence, so the noise path (build artifacts, ignored files) still costs one `stat`, not a walk of the cache.

### New `status` subcommand

`check` answers "are the services up". `status` answers "is my index current" — collection and point count, tantivy file count, per-file drift between the two stores in both directions, and every marker field compared against what the current config expects. Read-only, never fatal on a mismatch (reporting it is the point), and it reads tantivy through the read-only handle so it works while `serve` or `watch` holds the write lock.

### Kotlin / Swift buckets

`.kt` and `.swift` used to fall into the catch-all `text` bucket, so an Android or iOS project couldn't name its primary language in an `[index].languages` whitelist and `lang = "kotlin"` was not a usable search filter. Both now have their own bucket; `.kts` joins `gradle`, where build scripts belong. Still line-chunked — no tree-sitter grammar is wired up for them yet.

### Tests and CI

Test count 100 → 117. New: an integration test that drives the real binary over a pipe (handshake, both tool schemas, JSON-RPC error codes, and above all that stdout carries nothing but JSON-RPC — the one hard constraint no unit test can reach); collection-name derivation stability; a deterministic adversarial-UTF-8 sweep through the BM25 tokenizer, which slices by byte offset and would panic the indexer on a boundary mistake. CI gains a job pinned to the declared MSRV (1.88), previously an unverified assertion, and a `cargo audit` job.

The audit job's first run found four advisories; three were closed by semver-compatible lockfile bumps (`crossbeam-epoch` 0.9.18 → 0.9.20, `memmap2` 0.9.10 → 0.9.11, `quinn-proto` 0.11.14 → 0.11.16 — the last being an optional `reqwest` dependency that is never compiled here but is still recorded in `Cargo.lock`), plus `anyhow` 1.0.102 → 1.0.104 for an unsoundness advisory. The fourth, RUSTSEC-2025-0009 in `ring` <0.17.12, is not fixable in isolation: `ring` 0.17.12 needs a `cc` newer than 1.0.x while `tree-sitter-javascript 0.21.4` declares `cc = "~1.0.90"`, and 0.21.4 is the newest release compatible with the tree-sitter 0.22 ABI. It's accepted in `.cargo/audit.toml` with that reasoning recorded, and is unblocked by a tree-sitter 0.23 upgrade. MSRV is unchanged by the bumps (still driven by `time` 0.3).

The job runs `cargo audit` directly rather than through `rustsec/audit-check`, which publishes via the Checks API — that needs `checks: write` and is unavailable to fork pull requests, so it failed on permissions instead of on findings. A plain binary run reproduces exactly with `cargo audit` locally.

## 2026-07-08

### `[index].languages` is now optional — opt-out indexing by default

Flipped the walk model from **deny-unless-listed** to
**allow-unless-binary**. `[index].languages` used to be required, and any
extension not covered by a listed language was silently dropped — you only
found out a file type was missing (an Android manifest, a `.gradle`, a shader
config) by noticing search couldn't find it. That's a bad failure mode for a
quality-first tool.

Now:

- **Omit `languages` (or `[]`)** → the indexer walks *everything* except
  known-binary extensions and files over 2 MB (`walker::ExtPolicy::AllButBinary`).
  Zero-config: a new project just works, and no text type is silently dropped.
- **Set `languages`** → unchanged behavior: a strict whitelist to *narrow* a
  noisy repo (size cap doesn't apply — you named those types).

Binary safety is layered, not extension-guesswork:
- A built-in binary-extension deny-set (`BINARY_EXTENSIONS`: images, archives,
  native objects, `.spv`/`.jar`/`.class`/`.dex`, fonts, ML weights, …) is the
  fast pre-filter so large binaries aren't read just to be rejected.
- A 2 MB size cap skips generated/minified/bundled blobs without reading them.
- The indexer's read now sniffs content (`read_text_file`: NUL byte or invalid
  UTF-8 → `Ok(None)`) and skips binaries **quietly at debug**, catching
  anything the deny-set misses (extensionless executables, exotic types)
  without log spam.

Backward compatible: existing configs with an explicit `languages` list behave
exactly as before (same file set, same `config_soft` fingerprint). The new
default only affects configs that omit the field. Switching an existing config
to the default (removing `languages`) is a `config_soft` change → the marker
flow reindexes with stale cleanup on next `index` / `serve`.

### Android / JVM config buckets (`xml`, `gradle`, `properties`)

Also added three plain-text language buckets so an explicit whitelist can name
the mobile/AR config surface: `xml` (`AndroidManifest.xml`, `res/values/*.xml`,
layouts/menus), `gradle` (`build.gradle` / `settings.gradle`), `properties`
(`gradle.properties`). Line-chunked (no tree-sitter), like `toml`/`yaml`/`json`.
(With the opt-out default above these are indexed automatically anyway; the
buckets matter when you *narrow* via `languages`.)

Housekeeping: cleared two pre-existing clippy lints newer stable (1.96) began
flagging (`unnecessary_sort_by` in `vector_store.rs`, redundant `.into_iter()`
in `indexer.rs`) so `clippy -D warnings` is green again.

## 2026-06-12 — v0.0.3

**⚠ Upgrade note: the first `index` / `serve` start after this upgrade auto-clears and fully rebuilds the index** (BM25 schema revision bumped for the new tokenizer; re-embeds the whole corpus — plan for first-index duration).

### Code-aware BM25 tokenizer

The `content` field is now tokenized with identifier awareness: snake_case and camelCase split into sub-words (`build_si_portfolio` ↔ `buildSiPortfolio` ↔ "portfolio" all match each other), acronym boundaries respected (`HTTPServer` → `http`, `server`), digits stay attached (`bm25` whole). Only the parts are emitted — tantivy's QueryParser turns a multi-token word into a phrase query, so an exact identifier in the query still matches precisely as a consecutive phrase, including across naming styles.

BM25 queries are also sanitized (`:` → space; we only search `content`, so `field:value` syntax is never useful) and parsed leniently — previously `Watcher::run` parsed as a clause on a nonexistent field `Watcher` and could fail or silently drop the term.

`bm25::SCHEMA_VERSION` (new) feeds the marker's `config_hard` fingerprint — the auto-clear flow handles the migration.

### Exact-symbol boost

`[search].symbol_boost` (default 1.0): a chunk whose AST symbol is literally named in the query (`AdaptiveBatcher::note_failure`, tail segment `note_failure`, or verbatim `Indexer`) gets one extra #1 rank-vote in the RRF, applied before the rerank-head split so the match also earns a rerank slot. Ambiguity guard: plain lowercase English words (`run`, `new`) don't trigger it; identifier-likeness (underscores, `::`, camelCase, verbatim PascalCase) is required. This closes the main "legitimate grep fallback" — known-symbol lookups.

### Opt-in query log

`[serve].query_log_path`: one JSON line per `code_search` call (ts, query, filters, latency, result count, reranked flag, top-3 hits). MCP clients (Claude Code included) don't persist server stderr beyond connection start, so this is the only durable record of what the LLM asked and what came back. Off by default.

### `check` warns on chunking vs reranker-truncation mismatch

If any `chunking.max_chunk_chars` (default or per-language) exceeds `reranker.max_document_chars`, `check` warns that the cross-encoder will rank such chunks by their truncated head only.

### Housekeeping

- `rust-version = "1.88"` declared in Cargo.toml (computed from the dependency tree; README/CONTRIBUTING previously claimed 1.75+ which hasn't been true for a while)
- `examples/CLAUDE.md` tip: phrase queries in English — the reference embedding model is English-code-trained, the dense leg is weaker on non-English queries

## 2026-06-12 — v0.0.2

### Reranker resilience: halve-and-retry on "too large", visible RRF fallback

Two fixes for the silent-degradation mode found in production: a reranker server with a physical batch (`-ub`) smaller than `max_document_chars`-worth of tokens rejected every batch with HTTP 500 "input is too large to process", and every search quietly fell back to RRF-only ranking — visible only in server stderr, which Claude Code does not persist past connection start.

- **Halve-and-retry**: when the server rejects a batch as too large (physical batch / context overflow diagnostics matched in the error body), the truncation limit is halved and the call retried, up to 2 times with a 512-char floor. A degraded rerank over shortened documents beats losing the cross-encoder entirely. Token-per-char ratios vary ~2× between ASCII code and Cyrillic prose, so a static char limit can't be exactly right for every batch — the retry absorbs that.
- **Visible fallback**: when a reranker is configured but no returned hit carries a rerank score, the MCP tool response now leads with `WARNING: reranker unavailable — results are RRF-ranked only`. The degradation is visible to the calling LLM and in the transcript, not just in lost stderr. No warning when the reranker is intentionally disabled.

### `check` probes the reranker with a contrastive pair

`check` previously verified embeddings + Qdrant and only warn-logged that the reranker is not probed. It now sends a two-document contrastive probe (on-topic vs off-topic) and verifies the on-topic document scores higher — catching not just dead servers but the "loads cleanly, outputs garbage scores" failure mode of broken classifier heads (see jina-v3-under-llama.cpp history).

### Rerank fused into RRF as a rank-vote instead of replacing the final score

Previously, with rerank on, the final ranking was the cross-encoder's score alone — the dense+sparse RRF consensus was discarded. Observed failure mode on a real corpus (moex-trader): for "adaptive position sizing risk per trade", bge-reranker-v2-m3 ranked a data-pull bash script above `build_si_portfolio` (the actual fixed-fractional sizing code) that both retrieval modalities had at #1–2. A general-purpose multilingual cross-encoder is keyword-prose-biased on code; giving it veto power lets one model error sink a retrieval-consensus candidate.

Now the reranker contributes a third rank-vote on the same RRF scale: `final = rrf + rerank_weight / (rrf_k + rerank_rank + 1)`. Rank-based fusion, not score blending — reranker logits (−10..+10) and RRF scores (0.01–0.05) live on incomparable scales, the same argument that picked RRF over score-weighting for dense+sparse originally.

New `[search].rerank_weight` knob (default 2.0): the cross-encoder counts as two retrieval modalities — a strong vote, not a veto. `0.0` = rank purely by RRF while still reporting per-hit rerank scores. Head/tail ordering stays consistent: fused head scores ≥ their RRF ≥ any tail RRF.

Diagnostic change: final scores are now always on the RRF scale (~0.01–0.09). "Reranker fell back" is no longer detectable by score magnitude — look for missing `rerank=` component on hits instead (README troubleshooting updated).

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
