# Backlog

Work that is understood but not done. Ordered by value, not by effort. Each
entry states what is wrong, how to approach it, and — importantly — how you
would know it worked, since several of these are the kind of change that
looks right and isn't.

Everything here came out of the review and measurement work of August 2026;
the numbers quoted are from `eval/` and from real reindexing runs, not from
argument. See `docs/CHANGELOG.md` for what has already landed.

---

## 1. ~~Let the batcher derive its limits instead of guessing them~~

**Status: done, v0.0.8.** The budget ceiling and the per-request timeout are
now derived from measured throughput, and a timeout corrects that
measurement instead of only shrinking the budget. Verified on the reference
CPU host: the budget converged to ~23 KB against the measured ~750 chars/s
and stopped, with zero timeouts across a full reindex where the oscillation
used to produce them routinely. See the v0.0.8 changelog entry.

---

## 2. Find out why `docs` is the weakest category

**Status:** cause identified, one fix tried and measured worse, reverted.
Still open.

### What the cause turned out to be

Not chunk-size truncation — hypothesis 1 below is dead. Only 4% of markdown
sections exceed the reranker's 8000-char limit, and the section that a
failing query should have matched is 6676 chars, comfortably under it.

The real mechanism is BM25's IDF meeting a code corpus. The code-aware
tokenizer splits `upsert_and_save` into [`upsert`, `and`, `save`], so
English function words exist as index terms — and in a corpus of source
code they are *rare*, which BM25 reads as *informative*. A prose question
therefore scores a match on "and" as though it were a match on a technical
term.

Measured on a 6702-chunk project: the query "workspace architecture and
engineering decisions" returned `audit/atr_state.rs` first at bm25=15.37,
and the one-word query "and" returned that same chunk at exactly the same
score. The entire ranking of the leader came from the conjunction. The
document that answered the question was in the pool at rank 11 — so this
is a ranking defect, not a retrieval one, and widening `dense_k`/`sparse_k`
will not touch it.

### The fix that didn't work

Dropping a narrow list of English function words from the BM25 query
(whole words only, so `upsert_and_save` stays intact; query-side only, so
no reindex). Measured against a baseline on the same index:

| | before | after |
|---|---|---|
| MRR | 0.7853 | 0.7777 |
| recall@10 | 0.9714 | 0.9429 |
| `symbol` MRR | 0.967 | 1.000 |
| `semantic` MRR | 0.568 | 0.508 |
| `docs` MRR | 0.629 | 0.595 |

Worse overall, and the query that motivated the whole investigation moved
from rank 2 to rank 3. One query fell out of the top ten entirely: "FIFO
queue fill and VWAP blending of passive and market legs", whose target
module opens with *"FIFO queue fill (A5-sim), **and** VWAP blending **of**
passive + market legs"*.

That is the flaw in the idea, and it is worth remembering: a stopword is
noise when it matches an identifier fragment in unrelated code, and signal
when it matches prose in the right document. A query-side filter cannot
tell those apart, so it throws away both.

Anything that works will have to distinguish them — for instance by
where the term occurs rather than by what the term is. Note also that
category MRR here moves by ±0.16 between identical runs (five queries per
category, and the cross-encoder is not deterministic), so only the
35-query aggregate and the deterministic `symbol` category are worth
reading closely.

### Where it stands now

`eval/` consistently ranks prose worst: MRR 0.469 with the cross-encoder,
0.320 without, against 0.967 for exact-symbol lookups. Concretely, "how to
add support for a new language" returns three chunks of `walker.rs` instead
of `CONTRIBUTING.md`, which answers it in words. Code out-competes
documentation on the same vocabulary.

The hypotheses, and what is left of them:

1. ~~**Heading chunks are too big for the cross-encoder.**~~ **Ruled out.**
   Only 4% of markdown sections exceed the 8000-char limit, the median is
   679 chars, and the section behind a failing query is 6676 — under the
   limit and still missing. Costs nothing to re-check on another corpus,
   but it is not the explanation here.
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

**Status:** `dense_k` and `sparse_k` measured — no effect. Others still open.

### dense_k / sparse_k: depth is not a lever here

Swept 10 / 30 / 60 / 120 on a 6702-chunk project, cross-encoder off, three
runs per point:

| value | recall@10 (dense_k) | recall@10 (sparse_k) |
|---|---|---|
| 10 | 0.886 | 0.886 |
| 30 *(default)* | 0.886 | 0.886 |
| 60 | 0.886 | 0.886 |
| 120 | 0.886 | 0.886 |

Identical at every depth on both legs. `recall@5` oscillated by one query
with no trend, and the MRR bands overlap — see the run-to-run spread note
in `eval/README.md` before reading anything into those.

So the defaults are fine, and more usefully: **the four queries that miss
are missing at any depth.** They are not absent from the candidate pool,
they are ranked below it — the same conclusion the `docs` investigation
reached from the other direction. Widening retrieval cannot fix them; only
scoring can.

### Still unmeasured

- `rerank_top_n` — only meaningful with the cross-encoder on, so each point
  costs a full slow run and inherits the reranker's own noise. Needs
  `--repeat` and patience.
- `PATH_FILTER_OVERSAMPLE` and `MAX_RETRIEVAL_K` — not reachable from
  config (they are `const` in `search.rs`), and **no query in either set
  uses a `path` filter**, so there is nothing to measure them against yet.
  Add path-scoped queries first, then decide whether to make the constants
  configurable or to sweep them by rebuilding.

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
