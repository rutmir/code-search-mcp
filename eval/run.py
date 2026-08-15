#!/usr/bin/env python3
"""Retrieval-quality harness for code-search-mcp.

Answers the question the test suite structurally cannot: *did a ranking
change make results better or worse?* Unit tests pin behaviour, not
quality — a fusion weight can be moved in either direction and every test
still passes.

Two kinds of number come out of a run, and they are reported separately
because they travel differently:

  quality  (recall@k, MRR, symbol hit rate)
      A property of the ranking. Comparable across machines, so a result
      measured on a laptop still means something on a GPU host.

  latency  (p50/p95)
      A property of *this* host's embedding and rerank servers. Never
      compare it across machines; use it to pick a quality/latency
      trade-off for the host you are on.

That split is the point. The project targets servers of very different
speeds, so a knob is rarely "correct" in the abstract — it is correct for
a given host, and this tells you what each setting costs there.

Usage
-----
    # baseline
    eval/run.py --config .claude/code-search.toml --out base.json

    # after a change, compare
    eval/run.py --config .claude/code-search.toml --baseline base.json

    # what does the reranker's vote actually buy?
    eval/run.py --config .claude/code-search.toml \\
        --sweep search.rerank_weight=0,1,2,4

    # retrieval legs only — much faster, no cross-encoder
    eval/run.py --config .claude/code-search.toml --no-rerank

Requires only the stdlib. Queries are executed sequentially on purpose:
the reference llama.cpp servers run with --parallel 1, so concurrent
queries would queue and make the latency numbers meaningless.
"""

import argparse
import json
import os
import statistics
import subprocess
import sys
import tempfile
import time
import tomllib
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_BINARY = REPO_ROOT / "target" / "release" / "code-search-mcp"
DEFAULT_QUERIES = Path(__file__).resolve().parent / "queries" / "code-search-mcp.toml"
RECALL_AT = (1, 3, 5, 10)


# --------------------------------------------------------------------------
# query set


def load_queries(path):
    with open(path, "rb") as f:
        doc = tomllib.load(f)
    queries = doc.get("query", [])
    if not queries:
        sys.exit(f"no [[query]] entries in {path}")
    for i, q in enumerate(queries):
        if "q" not in q:
            sys.exit(f"query #{i + 1} in {path} has no `q`")
        if not q.get("files"):
            sys.exit(f"query {q['q']!r} has no expected `files`")
        q.setdefault("category", "uncategorized")
    return queries


# --------------------------------------------------------------------------
# config overrides


def patch_config(base_path, overrides, tmpdir):
    """Write a copy of the config with dotted `section.key=value` overrides.

    Values are parsed as TOML scalars, so `2.0`, `30` and `"rust"` all keep
    their intended type. Sweeping a knob without hand-editing the project's
    real config is the whole point.
    """
    if not overrides:
        return base_path
    with open(base_path, "rb") as f:
        doc = tomllib.load(f)
    for item in overrides:
        key, _, raw = item.partition("=")
        if not raw:
            sys.exit(f"--set expects section.key=value, got {item!r}")
        try:
            value = tomllib.loads(f"v = {raw}")["v"]
        except tomllib.TOMLDecodeError:
            value = raw  # bare string
        node = doc
        parts = key.split(".")
        for part in parts[:-1]:
            node = node.setdefault(part, {})
        node[parts[-1]] = value

    out = Path(tmpdir) / "config.toml"
    out.write_text(dumps_toml(doc))
    return out


def dumps_toml(doc):
    """Minimal TOML writer. Python ships `tomllib` for reading but no
    writer, and this is a debugging convenience — not worth a dependency.

    Recursive over tables at any depth: real configs here nest three levels
    (`[chunking.per_language.markdown]`), and a writer that stops at two
    silently emits the third as a quoted string, which the binary then
    rejects as a type error far from the cause.
    """

    def scalar(v):
        if isinstance(v, bool):
            return "true" if v else "false"
        if isinstance(v, (int, float)):
            return repr(v)
        if isinstance(v, list):
            return "[" + ", ".join(scalar(x) for x in v) + "]"
        return json.dumps(str(v))

    lines = []

    def emit(node, path):
        # Scalars first: TOML assigns bare keys to the most recent table
        # header, so anything emitted after a child header would land in
        # the wrong table.
        for key, value in node.items():
            if not isinstance(value, dict):
                lines.append(f"{key} = {scalar(value)}")
        for key, value in node.items():
            if isinstance(value, dict):
                lines.append(f"\n[{'.'.join(path + [key])}]")
                emit(value, path + [key])

    emit(doc, [])
    return "\n".join(lines).lstrip("\n") + "\n"


# --------------------------------------------------------------------------
# execution


def run_query(binary, config, query, limit, no_rerank, cwd):
    cmd = [str(binary), "-c", str(config), "search", query["q"], "-n", str(limit), "--json"]
    if query.get("lang"):
        cmd += ["--lang", query["lang"]]
    if query.get("path"):
        cmd += ["--path", query["path"]]
    if no_rerank:
        cmd.append("--no-rerank")

    started = time.monotonic()
    proc = subprocess.run(cmd, capture_output=True, text=True, cwd=cwd)
    elapsed_ms = (time.monotonic() - started) * 1000
    if proc.returncode != 0:
        tail = proc.stderr.strip().splitlines()[-3:]
        sys.exit(f"search failed for {query['q']!r}:\n  " + "\n  ".join(tail))
    try:
        hits = json.loads(proc.stdout)
    except json.JSONDecodeError:
        sys.exit(f"non-JSON output for {query['q']!r}:\n{proc.stdout[:500]}")
    return hits, elapsed_ms


def first_correct(hits, expected_files):
    """1-based rank of the first hit in an expected file, plus that hit."""
    for i, hit in enumerate(hits, start=1):
        if hit.get("file") in expected_files:
            return i, hit
    return None, None


def modality(hit):
    """Which retrieval leg surfaced this hit — the actionable signal when
    deciding whether dense_k or sparse_k is the one starving."""
    dense = hit.get("dense_score") is not None
    sparse = hit.get("sparse_score") is not None
    if dense and sparse:
        return "both"
    if dense:
        return "dense"
    if sparse:
        return "sparse"
    return "unknown"


def evaluate(binary, config, queries, limit, no_rerank, cwd, verbose):
    rows = []
    for n, query in enumerate(queries, start=1):
        hits, elapsed_ms = run_query(binary, config, query, limit, no_rerank, cwd)
        rank, hit = first_correct(hits, query["files"])
        symbol_ok = None
        if hit is not None and query.get("symbol"):
            name = hit.get("name") or ""
            symbol_ok = name == query["symbol"] or name.endswith("::" + query["symbol"])
        rows.append(
            {
                "q": query["q"],
                "category": query["category"],
                "expected": query["files"],
                "rank": rank,
                "modality": modality(hit) if hit else None,
                "symbol_ok": symbol_ok,
                "elapsed_ms": round(elapsed_ms, 1),
                "top": [h.get("file") for h in hits[:3]],
                "reranked": any(h.get("rerank_score") is not None for h in hits),
            }
        )
        mark = "." if rank else "X"
        print(f"  [{n}/{len(queries)}] {mark} {query['q'][:60]}", file=sys.stderr)
        if verbose and not rank:
            print(f"        expected {query['files']}, got {rows[-1]['top']}", file=sys.stderr)
    return rows


# --------------------------------------------------------------------------
# metrics


def metrics(rows):
    total = len(rows)
    found = [r for r in rows if r["rank"]]
    out = {
        "queries": total,
        "mrr": round(sum(1.0 / r["rank"] for r in found) / total, 4) if total else 0.0,
    }
    for k in RECALL_AT:
        out[f"recall@{k}"] = round(sum(1 for r in found if r["rank"] <= k) / total, 4)

    symbol_rows = [r for r in rows if r["symbol_ok"] is not None]
    if symbol_rows:
        out["symbol_hit_rate"] = round(
            sum(1 for r in symbol_rows if r["symbol_ok"]) / len(symbol_rows), 4
        )

    lat = sorted(r["elapsed_ms"] for r in rows)
    if lat:
        out["latency_p50_ms"] = round(statistics.median(lat), 1)
        out["latency_p95_ms"] = round(lat[min(len(lat) - 1, int(len(lat) * 0.95))], 1)

    by_cat = {}
    for r in rows:
        c = by_cat.setdefault(r["category"], {"n": 0, "hit": 0, "rr": 0.0})
        c["n"] += 1
        if r["rank"]:
            c["hit"] += 1
            c["rr"] += 1.0 / r["rank"]
    out["by_category"] = {
        cat: {
            "queries": c["n"],
            "recall@10": round(c["hit"] / c["n"], 4),
            "mrr": round(c["rr"] / c["n"], 4),
        }
        for cat, c in sorted(by_cat.items())
    }

    mods = {}
    for r in rows:
        if r["modality"]:
            mods[r["modality"]] = mods.get(r["modality"], 0) + 1
    out["found_via"] = dict(sorted(mods.items()))
    return out


def print_report(m, rows, title):
    print(f"\n=== {title} ===")
    print(f"queries          {m['queries']}")
    print(f"MRR              {m['mrr']:.4f}")
    for k in RECALL_AT:
        print(f"recall@{k:<10} {m[f'recall@{k}']:.4f}")
    if "symbol_hit_rate" in m:
        print(f"symbol hit rate  {m['symbol_hit_rate']:.4f}")
    print(f"found via        {m['found_via']}")
    print("\nby category:")
    for cat, c in m["by_category"].items():
        print(f"  {cat:<16} n={c['queries']:<3} recall@10={c['recall@10']:.3f}  mrr={c['mrr']:.3f}")
    print("\nlatency (this host only — never compare across machines):")
    print(f"  p50 {m.get('latency_p50_ms', 0):>8.1f} ms")
    print(f"  p95 {m.get('latency_p95_ms', 0):>8.1f} ms")

    misses = [r for r in rows if not r["rank"]]
    if misses:
        print(f"\nmisses ({len(misses)}):")
        for r in misses:
            print(f"  {r['q']}")
            print(f"      expected {r['expected']}")
            print(f"      got      {r['top']}")


def print_delta(base, new):
    print("\n=== vs baseline ===")
    keys = ["mrr"] + [f"recall@{k}" for k in RECALL_AT]
    if "symbol_hit_rate" in base and "symbol_hit_rate" in new:
        keys.append("symbol_hit_rate")
    for key in keys:
        b, n = base.get(key, 0.0), new.get(key, 0.0)
        d = n - b
        arrow = "→" if abs(d) < 1e-9 else ("↑" if d > 0 else "↓")
        print(f"  {key:<16} {b:.4f} → {n:.4f}  {arrow} {d:+.4f}")
    for key in ("latency_p50_ms", "latency_p95_ms"):
        if key in base and key in new:
            b, n = base[key], new[key]
            print(f"  {key:<16} {b:.1f} → {n:.1f} ms  ({n - b:+.1f})")


# --------------------------------------------------------------------------


def selftest(config):
    """Check the harness itself against a real config, without touching any
    service. Exists because the config rewriter shipped a bug that only
    surfaced as a type error from the binary three levels down: a table at
    depth three was emitted as a quoted string, and a check that merely
    *printed* the value could not tell the difference.
    """
    import copy

    failures = []

    def check(name, cond, detail=""):
        print(f"{'PASS' if cond else 'FAIL'}  {name}{(' — ' + detail) if detail else ''}")
        if not cond:
            failures.append(name)

    original = tomllib.loads(Path(config).read_text())
    with tempfile.TemporaryDirectory() as tmp:
        patched_path = patch_config(config, ["search.rerank_weight=4.0", "search.dense_k=60"], tmp)
        patched = tomllib.loads(Path(patched_path).read_text())

    expected = copy.deepcopy(original)
    expected.setdefault("search", {}).update({"rerank_weight": 4.0, "dense_k": 60})
    check("config round-trips with only the overrides changed", patched == expected)
    check("float override keeps its type", isinstance(patched["search"]["rerank_weight"], float))
    check("int override keeps its type", isinstance(patched["search"]["dense_k"], int))
    for path in (("chunking", "per_language"), ("chunking", "per_language", "markdown")):
        node = patched
        for part in path:
            node = node.get(part, {}) if isinstance(node, dict) else {}
        if node:
            check(f"{'.'.join(path)} survives as a table", isinstance(node, dict),
                  f"got {type(node).__name__}")

    rows = [
        {"rank": 1, "category": "a", "symbol_ok": True, "elapsed_ms": 100, "modality": "both"},
        {"rank": 3, "category": "a", "symbol_ok": False, "elapsed_ms": 200, "modality": "sparse"},
        {"rank": None, "category": "b", "symbol_ok": None, "elapsed_ms": 300, "modality": None},
        {"rank": 2, "category": "b", "symbol_ok": None, "elapsed_ms": 400, "modality": "dense"},
    ]
    m = metrics(rows)
    check("MRR", abs(m["mrr"] - (1 + 1 / 3 + 0 + 0.5) / 4) < 1e-4, str(m["mrr"]))
    check("recall@1", m["recall@1"] == 0.25, str(m["recall@1"]))
    check("recall@5", m["recall@5"] == 0.75, str(m["recall@5"]))
    check("symbol hit rate ignores unlabelled queries", m["symbol_hit_rate"] == 0.5)
    check("a miss contributes no modality", m["found_via"] == {"both": 1, "dense": 1, "sparse": 1})

    print("\nALL PASS" if not failures else f"\nFAILURES: {', '.join(failures)}")
    return 1 if failures else 0


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--config", required=True, help="project config passed to the binary")
    ap.add_argument("--selftest", action="store_true", help="check the harness; no services used")
    ap.add_argument("--queries", default=str(DEFAULT_QUERIES))
    ap.add_argument("--binary", default=str(DEFAULT_BINARY))
    ap.add_argument("--limit", type=int, default=10, help="hits per query (recall ceiling)")
    ap.add_argument("--no-rerank", action="store_true", help="retrieval legs only; much faster")
    ap.add_argument("--set", action="append", default=[], metavar="section.key=value")
    ap.add_argument("--sweep", metavar="section.key=v1,v2,...", help="one run per value")
    ap.add_argument("--out", help="write results JSON here")
    ap.add_argument("--baseline", help="compare against a results JSON from an earlier run")
    ap.add_argument("--verbose", action="store_true", help="print expected/got on each miss")
    args = ap.parse_args()

    if args.selftest:
        sys.exit(selftest(args.config))

    binary, config = Path(args.binary), Path(args.config)
    if not binary.exists():
        sys.exit(f"binary not found: {binary}  (cargo build --release)")
    if not config.exists():
        sys.exit(f"config not found: {config}")
    # `project.root = "."` is common, so the binary must run from the
    # project directory or it resolves a different collection entirely.
    cwd = config.resolve().parent.parent if config.parent.name == ".claude" else Path.cwd()

    queries = load_queries(args.queries)
    print(f"{len(queries)} queries, limit {args.limit}, cwd {cwd}", file=sys.stderr)

    with tempfile.TemporaryDirectory() as tmp:
        if args.sweep:
            key, _, values = args.sweep.partition("=")
            table = []
            for value in values.split(","):
                cfg = patch_config(config, args.set + [f"{key}={value}"], tmp)
                print(f"\n--- {key}={value} ---", file=sys.stderr)
                rows = evaluate(binary, cfg, queries, args.limit, args.no_rerank, cwd, args.verbose)
                m = metrics(rows)
                table.append((value, m))
                print_report(m, rows, f"{key}={value}")
            print(f"\n=== sweep: {key} ===")
            head = f"{'value':<12}{'MRR':>8}{'r@1':>8}{'r@5':>8}{'r@10':>8}{'p50 ms':>10}"
            print(head)
            print("-" * len(head))
            for value, m in table:
                print(
                    f"{value:<12}{m['mrr']:>8.4f}{m['recall@1']:>8.3f}"
                    f"{m['recall@5']:>8.3f}{m['recall@10']:>8.3f}"
                    f"{m.get('latency_p50_ms', 0):>10.1f}"
                )
            return

        cfg = patch_config(config, args.set, tmp)
        rows = evaluate(binary, cfg, queries, args.limit, args.no_rerank, cwd, args.verbose)
        m = metrics(rows)
        print_report(m, rows, "results")

        if args.baseline:
            with open(args.baseline) as f:
                print_delta(json.load(f)["metrics"], m)

        if args.out:
            payload = {
                "metrics": m,
                "rows": rows,
                "settings": {
                    "config": str(config),
                    "queries": args.queries,
                    "limit": args.limit,
                    "no_rerank": args.no_rerank,
                    "overrides": args.set,
                    "host": os.uname().nodename,
                },
            }
            Path(args.out).write_text(json.dumps(payload, indent=2))
            print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
