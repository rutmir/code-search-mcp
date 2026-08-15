# code-search-mcp

A **token-saving, quality-first replacement** for Claude Code's built-in `Explore` / Grep+Read iterative search. Exposed to Claude Code as an MCP server, it returns a ranked list of code chunks for any natural-language query in a single tool call.

[![CI](https://github.com/rutmir/code-search-mcp/actions/workflows/ci.yml/badge.svg)](https://github.com/rutmir/code-search-mcp/actions/workflows/ci.yml)
[![License: MIT OR Apache-2.0](https://img.shields.io/badge/license-MIT%20OR%20Apache--2.0-blue.svg)](#license)

## Why

When Claude Code explores a codebase to answer a question, it currently does it the hard way: many grep / read iterations, each one feeding back into the model. A real exploration query consumes tens of thousands of tokens (and a lot of wall time) before the model converges on an answer.

`code-search-mcp` returns the final ranked list (typically ~1500 tokens of structured `file:line + symbol + preview`) from a single tool call. The hit rate matches or exceeds the iterative approach because:

- **Hybrid retrieval** — dense semantic embeddings + sparse BM25 + cross-encoder reranking together outperform any single signal
- **AST-aware chunking** — chunks aligned to function / class / method boundaries via tree-sitter for 9 languages; the LLM sees `fn AdaptiveBatcher::note_failure` headers, not raw line ranges
- **Code-aware matching** — the BM25 tokenizer splits snake_case / camelCase identifiers (`buildSiPortfolio` finds `build_si_portfolio`), and a query that literally names a symbol boosts that chunk to the top — exact-symbol lookups don't need grep
- **Quality-first defaults** — recall is the lever, latency is not. A 2-minute search that finds the right answer beats a 10-second search that misses

**Typical savings: 30–100×** on tokens for code exploration, with equal or better recall.

## What it gives Claude Code

Once wired up, your Claude Code session gets two tools. The first is the one you'll see used:

```jsonc
// tools/call → code_search
{
  "query": "AIMD adaptive batching halving on failure",
  "limit": 5,        // optional, default 10
  "lang": "rust",    // optional, restrict to one language
  "path": "crates/"  // optional, only hits whose path contains this substring
}
```

Output (each hit carries `file:start-end`, language, syntactic kind+name, score, preview):

```
#1 crates/bot/src/execution/cross_pressure.rs:69-91  [rust]  struct CrossPressureDetector  score=0.0655
    pub struct CrossPressureDetector { /// Diagnostic only ... future_feed: Arc<L2Feed>, ...

#2 crates/bot/src/execution/cross_pressure.rs:390-399  [rust]  fn observe_tick_returns_verdicts  score=0.0512
    fn observe_tick_returns_verdicts() { let fut = L2Feed::new(); ...
```

(The CLI `search` command additionally prints per-component `dense=` / `bm25=` / `rerank=` scores for debugging; the MCP response keeps them out to save tokens.)

The second closes the loop when a preview isn't enough:

```jsonc
// tools/call → code_read_chunk
{
  "file": "crates/bot/src/execution/cross_pressure.rs",
  "start_line": 69,  // optional — omit both for every chunk of the file
  "end_line": 91
}
```

Previews are capped at 600 chars, and without this the model's next move is `Read` on the whole file — spending back much of what the search saved. `code_read_chunk` returns the chunk's untruncated text straight from the index (no file I/O), capped at 20k chars per call.

Both filters are exact about what they promise: `lang` is a hard restriction pushed into *both* retrieval sides, and a `path`-scoped query retrieves a deeper candidate pool so the filter can't starve the result set.

Searches that take a while report `notifications/progress` (embedding → retrieving → reranking → finalizing) to clients that ask for it with a `_meta.progressToken`.

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
                │  (bge-reranker-v2-m3)│
                │  fused as rank-vote  │
                └──────────────────────┘
                           │
                           ▼
                    ranked results
```

The indexer is incremental (sha-cached), the watcher tracks live edits, and the search is **two-stage** (RRF over a 30+30 candidate pool, cross-encoder over the top 30) so reranker cost is bounded regardless of corpus size. The cross-encoder's opinion is fused into the dense+sparse RRF as a weighted rank-vote rather than replacing the score — one reranker miss can't sink a candidate both retrieval modalities agree on (see `[search].rerank_weight`).

## Requirements

You'll need three running services. The recommended setup is `docker-compose` on a single LAN host (or `localhost` on a workstation):

- **[Qdrant](https://qdrant.tech/)** — vector DB. Any recent version. Default port `6333`.
- **[llama.cpp server](https://github.com/ggml-org/llama.cpp)** running an embedding model. The reference stack uses [`jina-code-embeddings-0.5b`](https://huggingface.co/jinaai/jina-code-embeddings-0.5b) (896-dim, code-specialized). Default port `7788`.
- **[llama.cpp server](https://github.com/ggml-org/llama.cpp)** running a cross-encoder reranker. The recommended model is [`BAAI/bge-reranker-v2-m3`](https://huggingface.co/BAAI/bge-reranker-v2-m3) (8192 ctx, multilingual). Default port `7799`.

A Rust toolchain is needed to build the binary. MSRV is 1.88 (declared in `Cargo.toml`; driven by the dependency tree).

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
       "-c", "8192", "-b", "8192", "-ub", "4096",
       "--parallel", "1", "--threads", "4",
       "--no-webui"]
    restart: always
    mem_limit: 6g
```

The `--pooling last` on the embedding server is **non-optional** for jina-code-embeddings-0.5b (it's a Qwen3-decoder model; `cls`/`mean` pooling produces semantically broken vectors). `--parallel 1` on the reranker is the right choice for memory-bandwidth-bound CPUs.

On the reranker, a cross-encoder processes each query+document pair in one pass, so the physical batch (`-ub`) must fit your longest truncated document — and llama.cpp silently clamps `-ub` to `-b`, so **set both**. `-b 8192 -ub 4096` comfortably fits the default `max_document_chars = 8000` (≈2000–4000 tokens depending on language). If a document still overflows, the client halves its truncation and retries automatically, at some quality cost.

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
# `languages` is OPTIONAL. Omit it to index everything that isn't a known
# binary or >2 MB (zero-config, nothing silently dropped). Set it only to
# NARROW a noisy repo to a whitelist — any of: rust, python, cpp, dart,
# typescript, javascript, go, csharp, java, xml, gradle, properties, shell,
# systemd, env, text, yaml, json, markdown, toml.
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
| `kotlin`, `swift` | lines | n/a | own bucket so they're whitelistable and filterable; no grammar wired up yet |
| `toml`, `yaml`, `json` | lines (default) | n/a | structured config |
| `shell`, `systemd`, `env`, `text` | lines | n/a | catch-all for `*.sh`, `*.service`, `*.env`, `*.ini`, `*.cfg`, `*.conf`, `*.example`, `*.local` |

Pick the ones you actually have and add them to `[index].languages`.

## Verify the stack

```bash
./target/release/code-search-mcp --config .claude/code-search.toml check
```

This pings the embedding endpoint, verifies the model's dimensions match the config, pings Qdrant, (if there's an existing index) verifies the project-identity marker, and probes the reranker with a contrastive document pair — a healthy cross-encoder must score an on-topic document above an off-topic one, which catches the "server is up but the model outputs garbage scores" failure mode.

Expected output:

```
INFO embedding endpoint OK model=jina-code-embeddings-0.5b dimensions=896
INFO qdrant collection does not exist yet — will be created on first `index`
INFO reranker endpoint OK model=bge-reranker-v2-m3 on_topic=2.31 off_topic=-8.7
INFO all checks passed
```

## Build the first index

```bash
./target/release/code-search-mcp --config .claude/code-search.toml index
```

The first run does a full scan + embed. Subsequent runs only process changed files (via sha-cache), and within a changed file only the chunks whose text actually changed — editing one function in a large file re-embeds that function, not all forty of its chunks. On modest CPU hardware expect ~5–30 minutes for the first run of a typical mid-sized repo, then seconds for incremental updates.

To see what's actually indexed at any point:

```bash
./target/release/code-search-mcp --config .claude/code-search.toml status
```

```
project      myproject
root         /home/u/proj
collection   myproject_a1b2c3d4
qdrant       48213 points
marker       written by v0.0.4 on 1754870400 (u@workstation)
  identity   match
  config     hard: match   soft: match
qdrant files 842
tantivy      /home/u/.cache/code-search/myproject — 842 files
drift        none — both stores agree
```

Unlike `check` (which pings the services), `status` is about the index itself: it's how you catch a config change you haven't reindexed for, or the two stores drifting apart. Read-only, and safe to run while `serve` or `watch` is going.

## Try a search from the CLI

```bash
./target/release/code-search-mcp --config .claude/code-search.toml \
    search "adaptive batching halving on failure" -n 5
```

```
query: adaptive batching halving on failure
results: 5 (18207 ms)

#1  src/adaptive_batcher.rs:42-89  [rust]  fn AdaptiveBatcher::note_failure  score=0.0648  dense=0.531  bm25=14.207  rerank=1.582
    pub fn note_failure(&mut self) { self.consecutive_ok = 0; ...

#2  src/indexer.rs:445-510  [rust]  fn embed_with_adaptive_batching  score=0.0489  dense=0.502  rerank=0.611
    ...
```

Each hit carries its component scores: `dense=` / `bm25=` / `rerank=`. If `rerank=` is missing on every hit, the reranker failed and search fell back to RRF-only ranking — check the reranker server.

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

After approval the `code_search` and `code_read_chunk` tools appear in Claude Code's tool list. The MCP server also runs the file watcher in the background (if `[watcher].enabled = true`), so the index stays live as you edit.

### Make Claude Code prefer `code_search` over Grep/Read

Tool descriptions alone don't always override Claude Code's habitual patterns (or your own auto-memory hints like "start by reading docs/INDEX.md"). To make `code_search` the default for any project-content query, drop a project-level `CLAUDE.md` at the project root — instructions there sit above MCP tool descriptions in Claude Code's instruction hierarchy:

```bash
cp examples/CLAUDE.md /path/to/your/project/CLAUDE.md
# edit the "Project-specific notes" section at the bottom to point at your own canonical files
```

**If your documentation isn't in English**, the template's advice on query language matters: ask in the language the document is written in, not in English. Identifiers are English whatever you do, so code lookups are unaffected — but on a project with Russian docs, the same five questions scored MRR 0.82 asked in Russian against 0.32 asked in English, and found the right document every time instead of four times in five (`eval/`, 6702-chunk corpus). Translating the question yourself only adds a gap for retrieval to bridge.

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
- `rerank_weight` (default 2.0) — the reranker's rank-vote weight in the final fusion, relative to one retrieval modality. The cross-encoder's opinion is fused into the dense+sparse RRF as a third rank-vote rather than replacing the score outright, so a candidate both retrieval sides agree on survives a single reranker miss. `0.0` ranks purely by RRF (rerank scores still reported per-hit)
- `symbol_boost` (default 1.0) — extra #1 rank-vote for chunks whose AST symbol is literally named in the query (`AdaptiveBatcher::note_failure`, `build_si_portfolio`, `Indexer`). Makes `code_search` competitive with grep for exact-symbol lookups. Ambiguous plain English words (`run`, `new`) don't trigger it. `0.0` disables

**`[serve]`** (all fields optional)
- `query_log_path` — when set, `serve` appends one JSON line per `code_search` call (timestamp, query, filters, latency, top hits). MCP clients don't persist the server's stderr, so this file is the only durable record of what was asked and what came back — useful when tuning retrieval quality. Off by default

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

**`reranker rejected batch as too large; halving truncation and retrying`**
A document exceeded the reranker server's physical batch (`-ub`) or context. The client automatically halves its truncation limit and retries (up to 2 times), so the search still gets reranked — over shortened documents. If this happens on most queries, raise `-b`/`-ub` on the reranker server (note: llama.cpp clamps `-ub` to `-b`, so set both) or lower `[reranker].max_document_chars`.

**`reranker failed; falling back to RRF-only ranking`** (and `WARNING: reranker unavailable` in tool output)
The reranker server is down, slow, or persistently rejecting requests. Check the reranker container logs; `check` now probes the reranker and reports whether it scores sanely. Search still returns useful results via RRF, and the MCP tool response is marked with a degradation warning so the calling LLM knows the ordering is approximate.

**`Qdrant collection '...' fingerprint mismatch`**
Your `vector_store.collection` points at someone else's collection (different project_id or root path). Either fix the config, rename the collection, or `clear --yes` to wipe and start fresh.

**`config-hard change detected ... AUTO-CLEARING the collection`**
You changed `chunking` or `embedding` parameters since the last index — or upgraded to a binary with a newer BM25 schema/tokenizer revision (noted in the [changelog](docs/CHANGELOG.md)). This is expected behavior: the indexer detected the change and is rebuilding cleanly, which re-embeds the whole corpus (plan for the same duration as the first index). If it was an accidental config edit, ctrl-C immediately and revert.

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
