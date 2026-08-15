# Retrieval evaluation

`cargo test` pins behaviour. It cannot tell you whether a ranking change made
results *better* — every test passes with `rerank_weight` at 0.0 or 4.0, and
with the `name` field boost at 1× or 10×. This harness closes that gap.

It exists because this project targets servers of very different speeds. A
retrieval knob is rarely correct in the abstract; it is correct for a given
host, at a given latency budget. So the report separates two kinds of number:

- **quality** — recall@k, MRR, symbol hit rate. Properties of the *ranking*.
  Comparable across machines: a number measured on a laptop still means
  something on a GPU host.
- **latency** — p50/p95. A property of *this* host's embedding and rerank
  servers. Never compare it across machines; use it to choose where on the
  quality/latency curve you want to sit.

## Running it

Needs a built binary and an **already-indexed** project — the harness only
queries, it never indexes.

Rebuild right before a long run. `cargo clippy --release` writes check-only
metadata into the same profile directory and takes the executable with it, so
a lint pass between the build and the run leaves `run.py` spawning a binary
that no longer exists — and it surfaces on the first query of a twenty-minute
sweep, not at startup.

```bash
cargo build --release

# baseline, saved for later comparison
eval/run.py --config /path/to/project/.claude/code-search.toml --out base.json

# after changing a default or a fusion weight
eval/run.py --config /path/to/project/.claude/code-search.toml --baseline base.json
```

The comparison prints per-metric deltas, so "did this help" has an answer
instead of an opinion.

### Answering a tuning question

`--sweep` runs the whole set once per value and prints a table:

```bash
# what is the cross-encoder's vote actually worth here?
eval/run.py --config … --sweep search.rerank_weight=0,1,2,4

# is the candidate pool deep enough on this corpus?
eval/run.py --config … --sweep search.dense_k=10,30,60
```

`--set section.key=value` applies overrides without touching the project's
real config (a temporary copy is patched instead), and repeats.

### Faster iteration

`--no-rerank` skips the cross-encoder. On a CPU host that is the difference
between minutes and tens of minutes per run. Use it while working on the
retrieval legs; use a full run before concluding anything about ranking.

Queries run sequentially on purpose: the reference llama.cpp servers use
`--parallel 1`, so concurrent queries would queue and make the latency
numbers meaningless.

## Writing a query set for your project

`queries/code-search-mcp.toml` describes *this* repository. For another
project, copy its shape:

```toml
[[query]]
q = "deciding a file has not changed since the last index"
category = "semantic"
files = ["src/indexer.rs"]      # where a person would actually look
symbol = "process_one_file"     # optional, adds a stricter symbol check
lang = "rust"                   # optional, passed through as a filter
path = "src/"                   # optional, likewise
```

A query counts as answered when **any** expected file appears in the results;
the rank of that first correct hit drives MRR and recall@k.

What makes a set worth having:

- **List where someone would actually look, not everything containing the
  words.** A set that accepts near-misses cannot detect a regression.
- **Keep the categories meaningful.** They isolate which leg moved: `symbol`
  leans on BM25's `name` field and the exact-symbol boost, `semantic` on the
  dense leg and the cross-encoder, `docs` on the heading chunker,
  `cross_cutting` on plain term matching. A change that helps one and hurts
  another is invisible in a single aggregate number.
- **Avoid anything the walker never indexes** — hidden directories
  (`.github/`, `.cargo/`), gitignored paths, files above the size cap. Those
  become permanent misses that drag every number down and teach you nothing.
- **Thirty-ish queries is enough to see a real move** and small enough to
  keep honest. Prefer adding a query when a real search disappoints you: a
  set grown from actual failures is worth more than one written in the
  abstract.

## Reading the output

```
MRR              0.7421
recall@1         0.6111
recall@10        0.9167
found via        {'both': 22, 'dense': 6, 'sparse': 5}
```

`found via` attributes the first correct hit to the retrieval leg that
surfaced it. It is the actionable one when deciding whether `dense_k` or
`sparse_k` is starving: if almost nothing arrives via `sparse`, the BM25 side
is not contributing and its budget is wasted.

The `misses` section lists every unanswered query with what came back
instead. That list, not the aggregate, is where the next improvement usually
comes from.

## A worked example

The first thing this harness was pointed at was `[search].rerank_weight`,
whose default of 2.0 had been chosen by argument rather than measurement.
Run against this repository (648 chunks) on a CPU-only llama.cpp host,
2026-08-15:

| rerank_weight | MRR | recall@1 | recall@5 | recall@10 | p50 |
|---|---|---|---|---|---|
| *(no rerank call)* | 0.8097 | 0.722 | 0.944 | 0.972 | **0.22 s** |
| 0 | 0.8097 | 0.722 | 0.944 | 0.972 | 32.5 s |
| 1 | 0.8111 | 0.722 | 0.944 | 0.972 | 33.3 s |
| **2** *(default)* | **0.8444** | **0.778** | **0.972** | 0.972 | 33.1 s |
| 4 | 0.8481 | 0.778 | 0.944 | 0.972 | 33.7 s |

Four things fall out of that table, and none were knowable before:

1. **The default sits at the knee.** Below 2 the vote barely registers;
   above it the curve is flat. 2.0 turns out to be right — now for a reason.
2. **`recall@10` never moves.** Reranking reorders the head, it never
   retrieves anything new. If a chunk isn't in the candidate pool, no weight
   rescues it — that's a `dense_k`/`sparse_k` question, not a rerank one.
3. **Weight 0 exactly matches skipping the call** (0.8097 in both, and
   identical per-category numbers). That's the fusion behaving as designed,
   and a useful self-check on the harness.
4. **The gain is concentrated.** Per category, going from 0 to 2:
   `cross_cutting` 0.675 → 0.875 and `docs` 0.383 → 0.490, while `symbol`
   stays at 0.967 and `semantic` slips slightly, 0.836 → 0.829. The
   cross-encoder earns its keep on prose and cross-cutting terms and does
   nothing at all for exact-identifier lookups.

Point 4 is the kind of thing worth acting on: a query that literally names a
symbol already ranks near-perfectly from retrieval alone, so on a host where
the cross-encoder costs 33 seconds it is 33 seconds spent for no measured
gain. Deciding that per query beats setting it per project.

And note what the table does *not* say. Latency here is a CPU host's; on a
GPU the same quality gain costs almost nothing and the trade is obvious.
That is the reason quality and latency are reported separately.
