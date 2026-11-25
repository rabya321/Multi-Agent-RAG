"""Build the orchestrator routing dev and test sets.

Samples retrievable queries from each agent's existing evaluation set,
merges hand-written queries from synthetic_queries.json (if present),
and writes two non-overlapping files:

    routing_dev.json   used while building the router (error analysis,
                       few-shot examples, choosing between variants)
    routing_test.json  run once at the end for the reported numbers

Read-only on the source files. Design: see routing_testset.md.

Each output query has: id, query, expected_route, target_kb, origin.
    expected_route: retrieve | fallback_general | clarify | out_of_scope
    target_kb:      wiki | papers | both   (only for retrieve, otherwise null)

Usage:
    python build_routing_testset.py            # dry run: print counts only
    python build_routing_testset.py --write    # also write both files
"""
import argparse
import hashlib
import json
import random
from collections import Counter, defaultdict
from pathlib import Path

SEED = 42
N_TEST_TOTAL = 120
N_TEST_PER_AGENT = 25
N_DEV_PER_AGENT = 10
N_DEV_PER_SYNTHETIC_CLASS = 4

EVAL_DIR = Path(__file__).resolve().parent
REPO_ROOT = EVAL_DIR.parents[1]
WIKI_STANDARD = REPO_ROOT / "wiki_agent/evaluation/output/standard_questions.json"
WIKI_CROSS = REPO_ROOT / "wiki_agent/evaluation/output/cross_track_questions.json"
PAPERS_SET = REPO_ROOT / "papers_agent/Golden_query_set/test_set_120.json"
SYNTHETIC = EVAL_DIR / "synthetic_queries.json"
DEV_OUTPUT = EVAL_DIR / "routing_dev.json"
TEST_OUTPUT = EVAL_DIR / "routing_test.json"

# Classes are used only to draw a balanced sample; they are not written to the output.
SAMPLED_CLASSES = ["retrieve_wiki", "retrieve_papers"]
SYNTHETIC_CLASSES = [
    "retrieve_both",
    "retrieve_short_definitional",
    "retrieve_not_in_kb",
    "clarify",
    "out_of_scope",
]
CLASSES = SAMPLED_CLASSES + SYNTHETIC_CLASSES


def _preflight_sources() -> None:
    missing = [p for p in (WIKI_STANDARD, WIKI_CROSS, PAPERS_SET, SYNTHETIC) if not p.exists()]
    if missing:
        lines = "\n".join(f"- {p}" for p in missing)
        raise FileNotFoundError(
            "Missing required input file(s) for routing set build:\n"
            f"{lines}\n"
            "Place/generate these files and rerun."
        )


def _rel(path: Path) -> str:
    return path.relative_to(REPO_ROOT).as_posix()


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()[:16]


def _allocate(counts: dict[str, int], n: int) -> dict[str, int]:
    """Proportional allocation of n samples across strata (largest remainder)."""
    total = sum(counts.values())
    quotas = {k: n * c / total for k, c in counts.items()}
    alloc = {k: min(int(q), counts[k]) for k, q in quotas.items()}
    leftover = n - sum(alloc.values())
    for k in sorted(quotas, key=lambda k: quotas[k] - alloc[k], reverse=True):
        if leftover == 0:
            break
        if alloc[k] < counts[k]:
            alloc[k] += 1
            leftover -= 1
    return alloc


def _stratified_dev_test(pool: list[dict], strata_key: str, rng: random.Random) -> tuple[list, list]:
    """Draw a test sample, then a dev sample from what remains. Both proportional by stratum."""
    by_stratum = defaultdict(list)
    for item in pool:
        by_stratum[item[strata_key]].append(item)
    for items in by_stratum.values():
        rng.shuffle(items)

    test_alloc = _allocate({s: len(v) for s, v in by_stratum.items()}, N_TEST_PER_AGENT)
    test = []
    for s in sorted(by_stratum):
        test += by_stratum[s][: test_alloc[s]]
        by_stratum[s] = by_stratum[s][test_alloc[s]:]

    dev_alloc = _allocate({s: len(v) for s, v in by_stratum.items()}, N_DEV_PER_AGENT)
    dev = []
    for s in sorted(by_stratum):
        dev += by_stratum[s][: dev_alloc[s]]
    return dev, test


def sample_wiki(rng: random.Random) -> tuple[list[dict], list[dict]]:
    pool = []
    for src in (WIKI_STANDARD, WIKI_CROSS):
        for q in json.loads(src.read_text(encoding="utf-8"))["questions"]:
            pool.append({
                "class": "retrieve_wiki",
                "category": q["type"],
                "query": q["question"],
                "expected_route": "retrieve",
                "target_kb": "wiki",
                "origin": {"file": _rel(src), "id": q["id"]},
            })
    return _stratified_dev_test(pool, "category", rng)


def sample_papers(rng: random.Random) -> tuple[list[dict], list[dict]]:
    pool = []
    for q in json.loads(PAPERS_SET.read_text(encoding="utf-8")):
        if q.get("ground_truth_retrievable") is not True or q.get("question_type") == "out_of_scope":
            continue
        pool.append({
            "class": "retrieve_papers",
            "category": q["question_type"],
            "query": q["question"],
            "expected_route": "retrieve",
            "target_kb": "papers",
            "origin": {"file": _rel(PAPERS_SET), "id": q.get("question_id")},
        })
    return _stratified_dev_test(pool, "category", rng)


def _label(r: dict) -> tuple[str, str | None]:
    """(expected_route, target_kb) for a hand-written query."""
    if r.get("expected_outcome") == "fallback_general":
        return "fallback_general", None
    if r["expected_route"] != "retrieve":
        return r["expected_route"], None
    targets = r["expected_targets"]
    return "retrieve", "both" if len(targets) == 2 else targets[0]


def load_synthetic(rng: random.Random, n_test_target: int) -> tuple[list[dict], list[dict]]:
    records = []
    for r in json.loads(SYNTHETIC.read_text(encoding="utf-8")):
        route, target_kb = _label(r)
        records.append({
            "class": r["class"],
            "category": r["category"],
            "query": r["query"],
            "expected_route": route,
            "target_kb": target_kb,
            "origin": {"file": SYNTHETIC.name},
        })
    by_class = defaultdict(list)
    for r in records:
        by_class[r["class"]].append(r)

    # Dev items are drawn proportionally within each category so the test
    # split keeps the intended category mix (e.g. off_domain vs live_or_private).
    dev, test = [], []
    remaining_by_class_and_cat = defaultdict(lambda: defaultdict(list))
    for cls in SYNTHETIC_CLASSES:
        by_cat = defaultdict(list)
        for r in by_class[cls]:
            by_cat[r["category"]].append(r)
        dev_alloc = _allocate({c: len(v) for c, v in by_cat.items()}, N_DEV_PER_SYNTHETIC_CLASS)
        for c in sorted(by_cat):
            rng.shuffle(by_cat[c])
            dev += by_cat[c][: dev_alloc[c]]
            remaining_by_class_and_cat[cls][c] = by_cat[c][dev_alloc[c]:]

    remaining_total = sum(len(v) for by_cat in remaining_by_class_and_cat.values() for v in by_cat.values())
    if remaining_total < n_test_target:
        raise ValueError(
            "Not enough synthetic queries to reach target test size after synthetic dev draw. "
            f"Need {n_test_target}, have {remaining_total}."
        )

    class_alloc = _allocate(
        {
            cls: sum(len(v) for v in remaining_by_class_and_cat[cls].values())
            for cls in SYNTHETIC_CLASSES
        },
        n_test_target,
    )
    for cls in SYNTHETIC_CLASSES:
        by_cat = remaining_by_class_and_cat[cls]
        cat_alloc = _allocate({c: len(v) for c, v in by_cat.items()}, class_alloc[cls])
        for c in sorted(by_cat):
            rng.shuffle(by_cat[c])
            test += by_cat[c][: cat_alloc[c]]

    return dev, test


def _finalise(records: list[dict], split: str) -> list[dict]:
    """Order by class, number the queries and keep only the output fields."""
    ordered = sorted(records, key=lambda r: CLASSES.index(r["class"]))
    return [
        {"id": f"{split}_{i:03d}", **{k: r[k] for k in ("query", "expected_route", "target_kb", "origin")}}
        for i, r in enumerate(ordered, 1)
    ]


def _write(path: Path, split: str, records: list[dict]) -> None:
    sources = [WIKI_STANDARD, WIKI_CROSS, PAPERS_SET, SYNTHETIC]
    payload = {
        "metadata": {
            "split": split,
            "seed": SEED,
            "sources": {_rel(p): _sha256(p) for p in sources},
            "total": len(records),
        },
        "queries": records,
    }
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Wrote {path}")


def _counts(records: list[dict]) -> Counter:
    return Counter(f"{r['expected_route']} {r['target_kb'] or ''}".strip() for r in records)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--write", action="store_true", help=f"Write {DEV_OUTPUT.name} and {TEST_OUTPUT.name}.")
    args = parser.parse_args()

    _preflight_sources()

    rng = random.Random(SEED)
    wiki_dev, wiki_test = sample_wiki(rng)
    papers_dev, papers_test = sample_papers(rng)

    if len(wiki_test) != N_TEST_PER_AGENT:
        raise ValueError(
            f"Wiki sampled test size is {len(wiki_test)}; expected {N_TEST_PER_AGENT}. "
            "Check source pool size and filters."
        )
    if len(papers_test) != N_TEST_PER_AGENT:
        raise ValueError(
            f"Papers sampled test size is {len(papers_test)}; expected {N_TEST_PER_AGENT}. "
            "Check source pool size and filters."
        )

    n_synthetic_test_target = N_TEST_TOTAL - len(wiki_test) - len(papers_test)
    if n_synthetic_test_target < 0:
        raise ValueError(
            f"Invalid targets: fixed wiki/papers test size ({len(wiki_test) + len(papers_test)}) "
            f"exceeds N_TEST_TOTAL ({N_TEST_TOTAL})."
        )

    syn_dev, syn_test = load_synthetic(rng, n_synthetic_test_target)

    dev = _finalise(wiki_dev + papers_dev + syn_dev, "dev")
    test = _finalise(wiki_test + papers_test + syn_test, "test")

    if len(test) != N_TEST_TOTAL:
        raise AssertionError(f"Expected exactly {N_TEST_TOTAL} test queries, got {len(test)}")

    overlap = {r["query"] for r in dev} & {r["query"] for r in test}
    assert not overlap, f"dev/test overlap: {overlap}"

    dev_counts, test_counts = _counts(dev), _counts(test)
    print(f"{'label':24} {'dev':>4} {'test':>5}")
    for label in sorted(dev_counts | test_counts):
        print(f"{label:24} {dev_counts[label]:>4} {test_counts[label]:>5}")
    print(f"{'TOTAL':24} {len(dev):>4} {len(test):>5}")

    if args.write:
        _write(DEV_OUTPUT, "dev", dev)
        _write(TEST_OUTPUT, "test", test)
    else:
        print("\n[dry run] Nothing written. Use --write to save.")


if __name__ == "__main__":
    main()
