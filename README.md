# code-search-mcp

A **token-saving, quality-first replacement** for Claude Code's built-in `Explore` / Grep+Read iterative search. Exposed to Claude Code as an MCP server, it returns a ranked list of code chunks for any natural-language query in a single tool call.

[![CI](https://github.com/<owner>/code-search-mcp/actions/workflows/ci.yml/badge.svg)](https://github.com/<owner>/code-search-mcp/actions/workflows/ci.yml)
[![License: MIT OR Apache-2.0](https://img.shields.io/badge/license-MIT%20OR%20Apache--2.0-blue.svg)](#license)

> Replace `<owner>` in the CI badge URL with your GitHub username/org after the first push.

## Why

When Claude Code explores a codebase to answer a question, it currently does it the hard way: many grep / read iterations, each one feeding back into the model. A real exploration query consumes tens of thousands of tokens (and a lot of wall time) before the model converges on an answer.

`code-search-mcp` returns the final ranked list (typically ~1500 tokens of structured `file:line + symbol + preview`) from a single tool call. The hit rate matches or exceeds the iterative approach because:

- **Hybrid retrieval** — dense semantic embeddings + sparse BM25 + cross-encoder reranking together outperform any single signal
- **AST-aware chunking** — chunks aligned to function / class / method boundaries via tree-sitter for 9 languages; the LLM sees `fn AdaptiveBatcher::note_failure` headers, not raw line ranges
- **Quality-first defaults** — recall is the lever, latency is not. A 2-minute search that finds the right answer beats a 10-second search that misses

**Typical savings: 30–100×** on tokens for code exploration, with equal or better recall.

## What it gives Claude Code

Once wired up, your Claude Code session gets a single tool:

```jsonc
// tools/call → code_search
{
  "query": "AIMD adaptive batching halving on failure",
  "limit": 5,        // optional, default 10
  "lang": "rust",    // optional, restrict to one language
  "path": "crates/"  // optional, restrict to a path prefix
}
```

Output (each hit carries `file:start-end`, language, syntactic kind+name, score, preview):

```
#1 crates/bot/src/execution/cross_pressure.rs:69-91  [rust]  struct CrossPressureDetector  score=1.6960
    pub struct CrossPressureDetector { /// Diagnostic only ... future_feed: Arc<L2Feed>, ...

#2 crates/bot/src/execution/cross_pressure.rs:390-399  [rust]  fn observe_tick_returns_verdicts  score=0.8847
    fn observe_tick_returns_verdicts() { let fut = L2Feed::new(); ...
```

## How it works

```
┌──────────────────────────────────────────────────────────────────────┐
│  Claude Code (MCP client)                                            │
└─────────────────────────┬────────────────────────────────────────────┘
                          │ JSON-RPC over stdio
┌─────────────────────────▼────────────────────────────────────────────┐
│  code-search-mcp serve  (one process per project)                    │
│  ┌──────────────────┐  ┌──────────────────────────────────────────┐ │
│  │  search loop     │  │  background watcher (incremental reindex)│ │
│  └────┬─────────────┘  └────────┬─────────────────────────────────┘ │
│       │                         │                                    │
└───────┼─────────────────────────┼────────────────────────────────────┘
        │                         │
        ▼                         ▼
   embed query                 walk + chunk + embed (changed files)
        │                         │
        ▼                         ▼
   ┌─────────────────────┐   ┌─────────────────────┐
   │  Qdrant             │   │  tantivy (local)    │
   │  dense vectors      │   │  BM25 sparse index  │
   └──────────┬──────────┘   └──────────┬──────────┘
              │                          │
              └────────────┬─────────────┘
                           ▼
                ┌──────────────────────┐
                │  RRF merge           │
                │  + top-N rerank      │
                │  (jina / bge-v2-m3)  │
                └──────────────────────┘
                           │
                           ▼
                    ranked results
```

The indexer is incremental (sha-cached), the watcher tracks live edits, and the search is **two-stage** (RRF over a 30+30 candidate pool, cross-encoder over the top 30) so reranker cost is bounded regardless of corpus size.

## Requirements

You'll need three running services. The recommended setup is `docker-compose` on a single LAN host (or `localhost` on a workstation):

- **[Qdrant](https://qdrant.tech/)** — vector DB. Any recent version. Default port `6333`.
- **[llama.cpp server](https://github.com/ggml-org/llama.cpp)** running an embedding model. The reference stack uses [`jina-code-embeddings-0.5b`](https://huggingface.co/jinaai/jina-code-embeddings-0.5b) (896-dim, code-specialized). Default port `7788`.
- **[llama.cpp server](https://github.com/ggml-org/llama.cpp)** running a cross-encoder reranker. The recommended model is [`BAAI/bge-reranker-v2-m3`](https://huggingface.co/BAAI/bge-reranker-v2-m3) (8192 ctx, multilingual). Default port `7799`.

A Rust toolchain is needed to build the binary. Tested with stable 1.75+.

### Reference docker-compose

```yaml
services:
  qdrant:
    image: qdrant/qdrant:latest
    ports: ["6333:6333"]
    volumes: ["./qdrant-storage:/qdrant/storage"]
    restart: always

  embedding:
    image: ghcr.io/ggml-org/llama.cpp:server
    ports: ["7788:7788"]
    volumes: ["./llm-models:/models"]
    command:
      ["-m", "/models/jina/jina-code-embeddings-0.5-Q8_0.gguf",
       "--embedding", "--pooling", "last",
       "--port", "7788", "--host", "0.0.0.0",
       "-c", "8192", "-ub", "2048", "--parallel", "1",
       "--no-webui"]
    restart: always

  reranking:
    image: ghcr.io/ggml-org/llama.cpp:server
    ports: ["7799:7799"]
    volumes: ["./llm-models:/models"]
    command:
      ["-m", "/models/bge/bge-reranker-v2-m3-q8_0.gguf",
       "--reranking",
       "--port", "7799", "--host", "0.0.0.0",
       "-c", "8192", "-ub", "2048",
       "--parallel", "1", "--threads", "4",
       "--no-webui"]
    restart: always
    mem_limit: 6g
```

The `--pooling last` on the embedding server is **non-optional** for jina-code-embeddings-0.5b (it's a Qwen3-decoder model; `cls`/`mean` pooling produces semantically broken vectors). `--parallel 1` on the reranker is the right choice for memory-bandwidth-bound CPUs.

## Install

```bash
git clone <repo-url> code-search-mcp
cd code-search-mcp
cargo build --release
# binary lands at ./target/release/code-search-mcp
```

## Configure

Create `.claude/code-search.toml` at the **root of the project you want to make searchable**. The minimal config:

```toml
[project]
id = "myproject"
root = "."

[index]
languages = ["rust", "toml", "markdown"]   # add any of: python, cpp, dart, typescript, javascript, go, csharp, java, shell, systemd, env, text, yaml, json
respect_gitignore = true
exclude = ["target/**", "build/**", "**/*.lock"]

[embedding]
provider = "openai-compatible"
url = "http://localhost:7788/v1/embeddings"
model = "jina-code-embeddings-0.5b"
dimensions = 896

[vector_store]
provider = "qdrant"
url = "http://localhost:6333"
# `collection` intentionally unset → auto-derives a collision-safe name.

[bm25]
provider = "tantivy"
index_path = "~/.local/state/code-search-mcp/myproject/tantivy"

[reranker]
provider = "openai-compatible"
url = "http://localhost:7799/v1/rerank"
model = "bge-reranker-v2-m3"
enabled = true

[chunking]
strategy = "lines"
max_chunk_lines = 60
overlap_lines = 10
max_chunk_chars = 20000

[chunking.per_language.rust]
strategy = "tree-sitter"
max_chunk_chars = 8000

[chunking.per_language.markdown]
strategy = "headings"
max_chunk_chars = 15000

[chunking.per_language.python]
strategy = "tree-sitter"
max_chunk_chars = 8000

[watcher]
enabled = true        # serve auto-spawns the watcher
debounce_ms = 500
```

A complete annotated example with every knob explained lives at `examples/full-config.toml`.

### Languages

| Language | Strategy | Method qualifier | Notable |
|---|---|---|---|
| `rust` | tree-sitter | `Type::method` | `impl_method` for impl blocks |
| `python` | tree-sitter | `Class.method` | `@decorator` preserved; imports grouped |
| `cpp` | tree-sitter | `Type::method` | inline + outline methods; templates unwrapped |
| `dart` | tree-sitter | `Class.method` | class methods reliable; top-level fns fall back |
| `typescript` | tree-sitter | `Class.method` | export keyword preserved, interface/type/enum |
| `javascript` | tree-sitter | `Class.method` | subset of TS emitter |
| `go` | tree-sitter | `Receiver.method` | pointer & value receivers normalized |
| `csharp` | tree-sitter | `Class.method` | namespaces descend without standalone emission |
| `java` | tree-sitter | `Class.method` | classes / interfaces / enums / records / annotations |
| `markdown` | headings | n/a | H1/H2 cuts, fenced-block aware |
| `toml`, `yaml`, `json` | lines (default) | n/a | structured config |
| `shell`, `systemd`, `env`, `text` | lines | n/a | catch-all for `*.sh`, `*.service`, `*.env`, `*.ini`, `*.cfg`, `*.conf`, `*.example`, `*.local` |

Pick the ones you actually have and add them to `[index].languages`.

## Verify the stack

```bash
./target/release/code-search-mcp --config .claude/code-search.toml check
```

This pings the embedding endpoint, verifies the model's dimensions match the config, pings Qdrant, and (if there's an existing index) verifies the project-identity marker.

Expected output:

```
INFO embedding endpoint OK model=jina-code-embeddings-0.5b dimensions=896
INFO qdrant collection does not exist yet — will be created on first `index`
INFO all checks passed
```

## Build the first index

```bash
./target/release/code-search-mcp --config .claude/code-search.toml index
```

The first run does a full scan + embed. Subsequent runs only process changed files (via sha-cache). On modest CPU hardware expect ~5–30 minutes for the first run of a typical mid-sized repo, then seconds for incremental updates.

## Try a search from the CLI

```bash
./target/release/code-search-mcp --config .claude/code-search.toml \
    search "adaptive batching halving on failure" -n 5
```

```
query: adaptive batching halving on failure
results: 5 (18207 ms)

#1  src/adaptive_batcher.rs:42-89  [rust]  fn AdaptiveBatcher::note_failure  score=1.5821
    pub fn note_failure(&mut self) { self.consecutive_ok = 0; ...

#2  src/indexer.rs:445-510  [rust]  fn embed_with_adaptive_batching  score=0.6112
    ...
```

If you see RRF-scored results in the 0.01–0.05 range instead, the reranker failed and search fell back to RRF — check the reranker server.

## Wire to Claude Code

Copy `examples/mcp.json` to your project's **`.mcp.json` at the project root** (NOT inside `.claude/` — Claude Code looks for it at the repo root). Edit the absolute path to the binary inside:

```json
{
  "mcpServers": {
    "code-search": {
      "command": "/abs/path/to/target/release/code-search-mcp",
      "args": ["--config", ".claude/code-search.toml", "serve"]
    }
  }
}
```

Project-scoped MCP servers require **explicit approval the first time** (security against malicious repo-supplied configs):

1. Run `claude` interactively in the project root
2. Accept the workspace trust prompt and approve the `code-search` server
3. Verify with `claude mcp list` — your server should appear without the `⏸ Pending approval` marker

After approval the `code_search` tool appears in Claude Code's tool list. The MCP server also runs the file watcher in the background (if `[watcher].enabled = true`), so the index stays live as you edit.

### Make Claude Code prefer `code_search` over Grep/Read

Tool descriptions alone don't always override Claude Code's habitual patterns (or your own auto-memory hints like "start by reading docs/INDEX.md"). To make `code_search` the default for any project-content query, drop a project-level `CLAUDE.md` at the project root — instructions there sit above MCP tool descriptions in Claude Code's instruction hierarchy:

```bash
cp examples/CLAUDE.md /path/to/your/project/CLAUDE.md
# edit the "Project-specific notes" section at the bottom to point at your own canonical files
```

The example tells Claude Code to use `code_search` BEFORE Grep / Glob / Read / Bash for any question about project content (code, docs, ROADMAPs, configs), and lists the genuine fall-back cases (known file paths, exact-bytes ops, non-indexed paths). After restart, Claude Code will reach for `code_search` first for exploration queries — visible as `tools/call code_search` entries in the MCP server's stderr logs at `~/.cache/claude-cli-nodejs/<project-slug>/mcp-logs-code-search/`.

## Configuration reference

The minimal config above is enough for most setups. Optional knobs you may want to touch:

**`[embedding]`**
- `startup_wait_secs` (default 300) — how long to wait for the server to finish loading the model at first `index`/`watch` start
- `max_input_chars` — warm-start hint for the adaptive batcher's initial budget. Defaults to 10 000 if unset; the batcher self-tunes from there

**`[vector_store]`**
- `collection` — leave unset (auto-derive). Only set if you intentionally want to share a collection across multiple installs

**`[reranker]`**
- `enabled` (default true) — set false if your environment doesn't have a reranker available; search falls back to RRF-only ranking
- `max_document_chars` (default 8000) — per-doc truncation. Drop to ~4000 for ~2× faster reranks at some quality cost
- `timeout_secs` (default 120) — outer reqwest cap; the throughput-aware dynamic timeout almost always fires first

**`[search]`** (all fields optional)
- `dense_k`, `sparse_k`, `rerank_top_n` (all default 30) — top-K from each modality + rerank ceiling. Drop to 15–20 for ~30–40% faster searches at recall cost
- `rrf_k` (default 60) — Reciprocal Rank Fusion constant per Cormack et al.

**`[chunking]`** and **`[chunking.per_language.<lang>]`**
- Changes to chunking parameters (strategy, max_chunk_lines, overlap_lines, max_chunk_chars, per-language overrides) trigger an **auto-clear + rebuild** on the next `index` because chunk identity changes. This is the right behavior, just be aware of it
- `max_chunk_chars` should be chosen by retrieval-quality preference, NOT by server capacity (the adaptive batcher handles server limits regardless)

**`[watcher]`**
- `enabled` (default true) — `serve` spawns the watcher in the background; standalone `watch` subcommand also available
- `debounce_ms` (default 500) — quiet window before processing a batch of filesystem events

See `examples/full-config.toml` for the fully-annotated reference.

## Operational notes

### Qdrant collection identity + config-change detection

A project-identity marker stored as a special point inside the collection prevents:
- Two projects accidentally picking the same `vector_store.collection` name from silently merging data (project-identity fingerprint mismatch → hard fail with actionable error)
- Changing sensitive chunking / embedding parameters from coexisting old chunks under different `chunk_uuid`s (config-hard mismatch → auto-clear + rebuild from scratch with a loud WARN log)
- Adding/removing languages or exclude patterns from leaving stale chunks (config-soft mismatch → reindex; stale-detection cleans up)

The marker is filtered out of search results by `kind = "marker"` payload tag, so you'll never see it in output.

### Tantivy is per-machine; Qdrant can be shared

If the same project tree is checked out at the same canonical path on two machines pointing at the same Qdrant, the marker's project-identity fingerprint matches and Qdrant chunks are shared — but each machine needs its own tantivy. The indexer's file→sha cache is intersected with the local tantivy file set, so files in Qdrant but missing from local tantivy get reprocessed (refills tantivy + idempotent re-upsert to Qdrant).

### MCP cancellation

`tools/call` requests run in their own tasks. A matching `notifications/cancelled` (with `params.requestId` matching the original request's id) preempts the search within microseconds via `tokio::select!`. Useful when Claude Code decides to abort a query.

### Multi-project

One MCP server process per Claude Code session. Each project's `.mcp.json` (at the project root) spawns its own server with its own `--config`. Per-project tantivy state lives under `~/.local/state/code-search-mcp/<project_id>/`. Qdrant collection names are namespaced per project (auto-derived) so multiple projects share one Qdrant instance without conflict.

## Troubleshooting

**`embedding endpoint returned HTTP 500: ... input too large to process`**
The adaptive batcher will halve the budget automatically and retry. If you see many of these in a row, your `--ubatch-size` on the embedding server may be too small for typical chunks; either raise it or lower `chunking.max_chunk_chars`.

**`reranker failed; falling back to RRF-only ranking`**
The reranker server is unhealthy, slow, or has a smaller ctx than your `max_document_chars`. Drop `[reranker].max_document_chars` to 4000–6000, or check the reranker container logs. Search still returns useful results via RRF.

**`Qdrant collection '...' fingerprint mismatch`**
Your `vector_store.collection` points at someone else's collection (different project_id or root path). Either fix the config, rename the collection, or `clear --yes` to wipe and start fresh.

**`config-hard change detected ... AUTO-CLEARING the collection`**
You changed `chunking` or `embedding` parameters since the last index. This is expected behavior — the indexer detected the change and is rebuilding cleanly. If it was an accidental edit, ctrl-C immediately and revert the config.

**Search returns results from an unrelated project**
Auto-derive prevents this for default setups, but if you copied a config from one project to another without changing `project.id` (and explicitly set the same `vector_store.collection` in both), the marker check should still catch it on the next `index`. If somehow it doesn't, `clear --yes` and re-index.

## Contributing

PRs welcome — see [CONTRIBUTING.md](CONTRIBUTING.md) for dev-stack setup, build/test workflow, and PR process.

## License

Dual-licensed under either:

- [Apache License 2.0](LICENSE-APACHE)
- [MIT license](LICENSE-MIT)

at your option.

Unless you explicitly state otherwise, any contribution intentionally submitted for inclusion in this work by you, as defined in the Apache 2.0 license, shall be dual-licensed as above, without any additional terms or conditions.

## Development history

See [docs/CHANGELOG.md](docs/CHANGELOG.md) for a date-ordered history of feature work and design decisions.
