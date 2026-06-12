# Contributing to code-search-mcp

Thanks for considering a contribution! This document covers the practical bits: how to set up a dev environment, run the tests, and structure a PR so it's easy to review.

## Quick orientation

Single-binary Rust project, ~6 kLoC. The interesting bits:

- `src/indexer.rs` — walks the project, chunks files, embeds, upserts to Qdrant + tantivy
- `src/adaptive_batcher.rs` — AIMD batcher (the "no static tuning knobs" engine for embedding requests)
- `src/chunker.rs` — `LineChunker`, `HeadingsChunker`, `TreeSitterChunker` (one per language); `ChunkerSet` dispatches per-language
- `src/embedding.rs`, `src/reranker.rs`, `src/vector_store.rs` — HTTP clients with retries / throughput-aware timeouts / marker safety
- `src/bm25.rs` — tantivy index and read-only `Bm25Search`
- `src/search.rs` — hybrid retrieval pipeline: embed query → dense+sparse → RRF merge → optional rerank, fused into the RRF as a weighted rank-vote
- `src/serve.rs` — MCP stdio JSON-RPC loop, `tools/call` cancellation, watcher integration
- `src/watcher.rs` — `notify`-debounced incremental reindex
- `src/walker.rs` — file enumeration + `PathFilter` (gitignore/exclude/extensions)
- `src/main.rs` — clap subcommands, dispatch

If you're touching language support, the pattern is: add the crate to `Cargo.toml`, add the extensions to `walker.rs`, add a `LangId` variant + `new_*()` constructor + `emit_*_node()` in `chunker.rs`, and write unit tests in `chunker.rs::tests::ts_*`.

## Prerequisites

- **Rust toolchain** — stable, MSRV 1.88 (declared in `Cargo.toml`; rustup recommended)
- **docker / docker-compose** — for the three external services you'll need at integration-test time

## Dev stack (for E2E work)

Unit tests don't need anything beyond `cargo test`. For real `index` / `search` / `serve` smoke testing you need the same three services described in the [main README](README.md):

```yaml
# docker-compose.dev.yml (not committed; copy from README's reference compose)
services:
  qdrant:           # vector DB
  embedding:        # llama.cpp server with jina-code-embeddings-0.5b
  reranking:        # llama.cpp server with bge-reranker-v2-m3
```

Models can be downloaded with `huggingface-cli` or any HF mirror. The Q8_0 GGUFs are ~600 MB each.

If you don't want to run the reranker locally during dev, set `[reranker].enabled = false` in your test config — search falls back to RRF-only ranking, still useful for verifying the dense+sparse path.

## Build and test

```bash
# unit tests (no external deps)
cargo test

# release build
cargo build --release

# format + lint (do this before pushing)
cargo fmt
cargo clippy -- -D warnings

# E2E smoke with the dev stack running on localhost
./target/release/code-search-mcp --config dev/code-search.toml check
./target/release/code-search-mcp --config dev/code-search.toml index
./target/release/code-search-mcp --config dev/code-search.toml search "your query"
```

`cargo test` runs 75+ unit tests covering chunker behavior for every supported language, adaptive batcher AIMD logic, serve-mode response shapes, etc. If you add a feature, add a unit test or two — the chunker tests are good templates.

## Code style

- **`cargo fmt`** before every commit (CI will catch it but save the round-trip)
- **`cargo clippy -- -D warnings`** before pushing — we keep the codebase clippy-clean
- **Comments**: only when *why* is non-obvious. Don't restate what the code does. Don't reference past states ("used to be...", "removed for..."). Do explain hidden constraints, subtle invariants, or surprising decisions
- **Error handling**: use `anyhow::Result` and `.context("…")` at every API boundary; don't swallow errors with `let _ = …` unless intentional (and comment why)
- **Async**: tokio runtime, never block. Use `tokio::time::timeout` for per-operation timeouts; use `tokio::select!` for cancellation
- **Tracing**: log to `tracing` at the right level — `info` for state transitions, `debug` for per-batch detail, `warn` for recoverable issues, `error` only for things that abort the operation. `stdout` is reserved for MCP framing; everything goes to `stderr` via the global subscriber

## Design discussions before code

For anything bigger than a bug fix or a small additive feature (~100 LoC), please open a **GitHub Issue or Discussion** first describing the design. This avoids the worst-case of writing 2 kLoC that doesn't land because of a foundational disagreement. Reviewing a design sketch takes minutes; reviewing 2 kLoC takes hours.

Good candidates for issue-first:
- New language support that requires non-trivial AST handling (Rust impl methods, C++ outline definitions, etc. — patterns to follow)
- Changes to the marker fingerprint scheme (it's a forward-compat boundary)
- Anything in `serve.rs` touching the JSON-RPC concurrency model
- New MCP capabilities (resources, prompts, etc.)

Trivial / additive:
- New languages that follow an existing pattern
- New CLI flags
- Bug fixes with a test case
- Doc / example improvements

…just open a PR.

## PR process

1. **Branch from `main`**, one focused change per PR. If you discover a separate bug while working, fix it in a separate PR
2. **Write a clear PR description**: what changed, why, and how you verified it. Link to the issue if there is one
3. **Add tests** for new features, especially anything in `chunker.rs` (every language emitter has unit tests; new ones should too) or `adaptive_batcher.rs`
4. **Run** `cargo fmt && cargo clippy -- -D warnings && cargo test` locally before requesting review
5. **Keep commits clean** — squash WIP commits, but don't squash semantically distinct changes into one. We don't enforce a particular commit message style, but "fix X" beats "fixes" with no body
6. Expect a turnaround of a few days for review. Ping if it's been longer

## License acceptance

This project is dual-licensed under **MIT OR Apache-2.0**. By submitting a PR you agree that your contribution will be licensed under those same terms, without any additional terms or conditions. You don't need to add a CLA signature or copyright header — your `git author` is the attribution.

If your contribution incorporates third-party code, please flag it in the PR description and confirm the license is MIT/Apache-2.0 compatible (BSD-2/3, ISC, MPL-2.0 file-level are typically fine; GPL is not).

## Common gotchas

**Tantivy schema changes** — if you add a field to the tantivy schema, every existing index becomes incompatible. Bump the bm25 `schema_version` in the marker fingerprint so the auto-clear flow handles it (currently `config_hard`).

**Qdrant payload changes** — same principle: changing the chunk payload schema means existing chunks have stale data. The marker's `config_hard` fingerprint catches embedding model + dimensions changes, but if you add a new payload field that's required at search time, document the migration.

**Tree-sitter version compatibility** — `tree-sitter = "0.22"` is the parent crate; language-specific crates (`tree-sitter-rust`, `tree-sitter-python`, etc.) must use a compatible major. If `cargo update` pulls in an incompatible language crate version, pin the version in `Cargo.toml`.

**MCP stdio discipline** — anything written to `stdout` outside the JSON-RPC framing breaks the client silently. Don't add `println!` to anything reachable from `serve` mode. Use `tracing::info!` etc. (which goes to stderr via the global subscriber).

**Per-language emitter pitfalls** — when adding a new tree-sitter language, watch for: (a) field name differences (`name` vs `identifier` vs direct child traversal — check `node-types.json`), (b) wrapping nodes (TypeScript `export_statement`, C++ `template_declaration`, Python `decorated_definition` — unwrap and recurse), (c) methods inside classes need qualification (`Class.method` for dot-call languages, `Class::method` for `::`-call languages).

## Questions

Open a GitHub Discussion. Don't email maintainers privately for general questions — keep it in the open so others benefit from the answer.
