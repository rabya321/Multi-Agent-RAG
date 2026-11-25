"""Primary overall tests plus exploratory question-type tests.

Bonferroni correction covers six planned tests separately within each group,
NOT all groups together. Question-type tests require more than 10 valid pairs.
"""

import csv
import json
from pathlib import Path
from statistics import mean

import scipy
from scipy.stats import wilcoxon


# STEP 1: Choose input files and the six planned comparisons.
ROOT = Path(__file__).resolve().parent
AGENT_FILE = ROOT / "evaluation_results_agent_instrumented/per_question_full_20261123_002436.json"
BASELINES = {
    "k=5": ROOT / "evaluation_results_naive_k5/per_question_full_20261123_013608.json",
    "k=15": ROOT / "evaluation_results_naive_k15/per_question_full_20261122_203528.json",
}
METRICS = ["answer_correctness", "context_recall", "faithfulness"]
ALPHA = 0.05
NUMBER_OF_TESTS = len(BASELINES) * len(METRICS)
MIN_TYPE_PAIRS = 10  # Strictly more than 10 valid pairs, checked per metric.


def load_scores(path):
    """STEP 2: Read records and check duplicate IDs."""
    with path.open(encoding="utf-8") as file:
        records = json.load(file)
    scores = {}
    seen = set()
    for record in records:
        qid = record["question_id"]
        if qid in seen:
            raise ValueError(f"Duplicate question ID {qid} in {path}")
        seen.add(qid)
        scores[qid] = record
    return scores


def _require_fields(record: dict, fields: tuple[str, ...], qid: str, label: str) -> None:
    """Raise a clear error when an expected comparison field is missing."""
    missing = [field for field in fields if field not in record]
    if missing:
        raise ValueError(
            f"Missing required fields for {qid} in {label}: {', '.join(missing)}"
        )


def compare(agent, baseline, baseline_name, metric, question_ids, group):
    # STEP 4: Use all aligned pairs.
    key = "ragas_" + metric
    pairs = [
        (agent[qid][key], baseline[qid][key]) for qid in question_ids
    ]

    # STEP 5: Positive = agent wins; zero = tie; negative = baseline wins.
    # Round tiny floating-point errors away before ranking differences.
    differences = [round(a - b, 12) for a, b in pairs]
    nonzero = [d for d in differences if d != 0]
    result = {
        "group": group,
        "analysis": "primary" if group == "Overall" else "exploratory",
        "baseline": baseline_name,
        "metric": metric,
        "n_questions": len(question_ids),
        "n_valid": len(pairs),
        "n_effective": len(nonzero),
        "agent_mean": mean(a for a, b in pairs) if pairs else None,
        "baseline_mean": mean(b for a, b in pairs) if pairs else None,
        "mean_difference": mean(a - b for a, b in pairs) if pairs else None,
        "wins": sum(d > 0 for d in differences),
        "ties": sum(d == 0 for d in differences),
        "losses": sum(d < 0 for d in differences),
        "W": None,
        "p_raw": None,
        "p_adjusted": None,
        "significant": False,
        "note": "",
    }
    if group != "Overall" and len(pairs) <= MIN_TYPE_PAIRS:
        result["note"] = "Descriptive only: 10 or fewer valid pairs"
        return result
    if not nonzero:
        result["note"] = "All pairs tied" if pairs else "No valid pairs"
        return result

    # STEP 6: Test signed ranks, NOT the arithmetic mean directly.
    # Two-sided allows either system to outperform the other.
    # zero_method='wilcox' excludes tied pairs from ranking.
    test = wilcoxon(
        differences, alternative="two-sided", zero_method="wilcox", method="auto"
    )
    result["W"] = float(test.statistic)
    result["p_raw"] = float(test.pvalue)

    # STEP 7: Six planned tests within THIS group (not across all groups).
    result["p_adjusted"] = min(result["p_raw"] * NUMBER_OF_TESTS, 1.0)
    result["significant"] = result["p_adjusted"] < ALPHA
    return result


def main():
    agent = load_scores(AGENT_FILE)
    groups = {"Overall": sorted(agent)}
    for qid in sorted(agent):
        question_type = agent[qid]["question_type"]
        if question_type in {"section_specific", "unanswerable_from_summary"}:
            question_type = "section-specific"
        groups.setdefault(question_type, []).append(qid)
    # Smaller groups cannot have >10 valid pairs, so omit them altogether.
    groups = {name: ids for name, ids in groups.items()
              if name == "Overall" or len(ids) > MIN_TYPE_PAIRS}
    results = []
    for baseline_name, path in BASELINES.items():
        baseline = load_scores(path)

        # STEP 3: Confirm these really are the same questions and references.
        if not agent or agent.keys() != baseline.keys():
            raise ValueError(f"Empty or mismatched question set for {baseline_name}")
        fields_to_match = ("question", "question_type", "expected_answer")
        for qid in agent:
            _require_fields(agent[qid], fields_to_match, qid, AGENT_FILE.name)
            _require_fields(baseline[qid], fields_to_match, qid, path.name)
            for field in fields_to_match:
                if agent[qid][field] != baseline[qid][field]:
                    raise ValueError(f"Mismatched {field} for {qid}: {baseline_name}")

        for group, question_ids in groups.items():
            for metric in METRICS:
                results.append(compare(agent, baseline, baseline_name, metric,
                                       question_ids, group))

    # STEP 8: Print readable blocks rather than a very wide terminal table.
    print(f"SciPy {scipy.__version__}; two-sided Wilcoxon, method=auto, zeros=wilcox")
    print(f"Bonferroni: {NUMBER_OF_TESTS} tests PER GROUP; adjusted p < {ALPHA}")
    print("Overall = primary; question types = exploratory (>10 valid pairs).")
    print("Correction does not control false positives across all groups together.")
    for row in results:
        print(f"\n{row['group']} ({row['analysis']}): Agent vs {row['baseline']} - {row['metric']}")
        print(f"  Valid pairs: {row['n_valid']}; non-tied pairs: {row['n_effective']}")
        print(f"  Wins / ties / losses: {row['wins']} / {row['ties']} / {row['losses']}")
        for field in ("agent_mean", "baseline_mean", "mean_difference", "W",
                      "p_raw", "p_adjusted"):
            value = row[field]
            text = "N/A" if value is None else f"{value:.6g}"
            print(f"  {field}: {text}")
        print(f"  Significant: {row['significant'] if row['p_raw'] is not None else 'Not tested'}")
        if row["note"]:
            print(f"  Note: {row['note']}")

    # STEP 9: Keep primary and exploratory results in separate output files.
    save_results([row for row in results if row["analysis"] == "primary"], "primary")
    save_results([row for row in results if row["analysis"] == "exploratory"], "question_type")


def save_results(results, suffix):
    if not results:
        return
    csv_path = ROOT / f"wilcoxon_{suffix}_results.csv"
    json_path = ROOT / f"wilcoxon_{suffix}_results.json"
    with csv_path.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(results[0]))
        writer.writeheader()
        writer.writerows(results)
    with json_path.open("w", encoding="utf-8") as file:
        json.dump(results, file, indent=2, allow_nan=False)
    print(f"\nSaved: {csv_path}\nSaved: {json_path}")


if __name__ == "__main__":
    main()


