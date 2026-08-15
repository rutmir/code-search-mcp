# Backlog

Work that is understood but not done. Ordered by value, not by effort. Each
entry states what is wrong, how to approach it, and — importantly — how you
would know it worked, since several of these are the kind of change that
looks right and isn't.

Everything here came out of the review and measurement work of August 2026;
the numbers quoted are from `eval/` and from real reindexing runs, not from
argument. See `docs/CHANGELOG.md` for what has already landed.

---

## 1. Let the batcher derive its limits instead of guessing them

**Status:** ready to implement.

Two constants in `adaptive_batcher.rs` claim to know a server they have never
met. The project's whole premise is that it adapts to hosts of very different
speeds, and these are the places where it stops doing that.

**`MAX_BUDGET = 256_000`** is a hard ceiling on the char budget. On a
CPU-bound llama.cpp host it is unreachable, so the AIMD loop climbs to it,
times out, halves, and climbs again — paying a full timeout per cycle,
forever. Observed repeatedly during the v0.0.5 rebuild of a 6702-chunk
project.

The fix is *not* to lower the number: that would break a GPU host, and
trading one hard-coded guess for another misses the point. `chars_per_sec` is
already tracked as an EWMA — derive the ceiling from it, as the largest batch
that still completes in a sane time at the observed throughput.

**`TIMEOUT_CEILING_SECS = 300`** equals the default `embedding.timeout_secs`,
so a derived timeout that exceeds the cap is silently truncated and the
request fails looking exactly like a genuine "batch too large". The batcher
then shrinks its budget when what was actually wrong was its model of how
long the server takes. It can never learn that, because the two failures are
indistinguishable at the point of decision.

Reconcile the two: the batcher's own timeout should fire first and be
attributable, and a timeout that came from the client's own impatience should
correct the throughput estimate rather than the budget.

**How you'd know it worked:** reindex a project and grep the log for repeated
`batch too large` at the same batch size. They should be gone, and total wall
time should drop against the 3.5 hours that run took.

**Risk:** low. A miscalculated ceiling degrades to current behaviour. But it
can only be judged on a real multi-hour run, so budget for that.

---

## 2. Find out why `docs` is the weakest category

**Status:** investigation, not a task. No known fix yet.

`eval/` consistently ranks prose worst: MRR 0.469 with the cross-encoder,
0.320 without, against 0.967 for exact-symbol lookups. Concretely, "how to
add support for a new language" returns three chunks of `walker.rs` instead
of `CONTRIBUTING.md`, which answers it in words. Code out-competes
documentation on the same vocabulary.

Three hypotheses, cheapest first:

1. **Heading chunks are too big for the cross-encoder.** The reference config
   sets `max_chunk_chars = 15000` for markdown while `[reranker]
   .max_document_chars` defaults to 8000 — so document chunks are truncated
   on the way in and the reranker judges a fragment. `check` already warns
   about this mismatch. Lower the markdown chunk cap, re-measure. One run.
2. **BM25 over-rewards identifier density.** Code chunks carry many rare
   tokens; prose carries common ones. Worth looking at the raw `bm25=` scores
   on a `docs` query before doing anything.
3. **The dense leg is simply weaker on prose** than on code, this being a
   code-specialised embedding model.

**How you'd know it worked:** the `docs` and `docs_ru` categories move up
while `symbol` and `semantic` do not move down. The last part matters — it is
easy to help prose by handicapping code.

---

## 3. Measure the retrieval constants that are still guesses

**Status:** mechanical; may well conclude "current values are correct".

`dense_k`, `sparse_k`, `rerank_top_n`, `PATH_FILTER_OVERSAMPLE` and
`MAX_RETRIEVAL_K` were all chosen by argument. `eval/run.py --sweep` exists
precisely for this.

Start with `dense_k` / `sparse_k`: they decide what enters the candidate pool
at all, and reranking has been measured to reorder that pool without ever
adding to it. A constant that starves the pool cannot be rescued downstream.

Note this is the one entry that may produce no code. That is a legitimate
outcome — "the defaults are right, and here is the evidence" is worth having
written down.

**Cost:** roughly 20 minutes per sweep point on a 6702-chunk corpus, less on
a small one but correspondingly less informative.

---

## 4. Move the tree-sitter family to ABI 0.23

**Status:** large, and gated on a correctness fix that must ship with it.

One upgrade closes three separate things: Kotlin and Swift grammars (the
language buckets are already in place and line-chunked), `ring` >= 0.17.12
— currently an accepted advisory in `.cargo/audit.toml`, blocked because
`tree-sitter-javascript 0.21.4` pins `cc = "~1.0.90"` — and two unsound
advisories in `lru`, blocked by `tantivy 0.22`.

**Do not start this without also fixing the fingerprint.** Grammar versions
are *not* part of `config_hard` (`vector_store.rs::config_hard_fingerprint`
covers chunking parameters, the embedding model and `bm25::SCHEMA_VERSION`).
If a new grammar cuts AST boundaries even slightly differently, chunk
identities change with no auto-clear, and old and new chunks coexist as
orphans — precisely the failure the marker exists to prevent. Add the grammar
revision to the hard fingerprint in the same change, and expect a full
reindex (~3.5 h on a 6702-chunk project).

**Suggested order:** upgrade the existing nine grammars first and confirm the
`chunker.rs::tests::ts_*` suite passes untouched — that tells you whether the
emitters need reworking for renamed grammar fields. Only then add Kotlin and
Swift, which are new emitters rather than migrations.

---

## Smaller items

- **`semantic` ordering gets slightly worse with reranking** (MRR 0.570 →
  0.518) while coverage improves (recall@10 0.700 → 0.900). Ten queries is
  too small a sample to act on; revisit if it survives a larger set.
- **`eval/run.py --sweep search.rerank_weight` re-runs the cross-encoder per
  point** even though rerank scores don't depend on the weight — only the
  fusion does. One pass could compute every point. Worth it only if weight
  sweeps become routine.
