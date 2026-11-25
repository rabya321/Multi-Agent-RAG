"""Wiki agent vs k=8/k=30: primary and exploratory paired Wilcoxon tests.

Same design as the papers experiment: three metrics x two baselines, with
Bonferroni correction separately within each group. No LLM/API calls.
"""

import csv
import hashlib
import json
import math
from datetime import datetime
from pathlib import Path
from statistics import mean

import scipy
from scipy.stats import wilcoxon


# STEP 1: Inputs and planned tests. Baseline k labels follow metrics_run.md.
ROOT = Path(__file__).resolve().parent
AGENT_FILE = ROOT / "agent_all_eval.json"
BASELINES = {
    "k=8": ROOT / "archive_pre_fix/naive_all__eval.json",
    "k=30": ROOT / "archive_pre_fix/naive_all_k30_eval.json",
}
METRICS = ["answer_correctness", "context_recall", "faithfulness"]
ALPHA = 0.05
NUMBER_OF_TESTS = len(BASELINES) * len(METRICS)
MIN_TYPE_PAIRS = 10  # Require strictly MORE than 10 valid pairs per metric.


def load_scores(path):
    """STEP 2: Index the wiki's evaluation records by their unique 'id'."""
    with path.open(encoding="utf-8") as file:
        records = json.load(file)
    scores = {}
    for record in records:
        qid = record["id"]
        if qid in scores:
            raise ValueError(f"Duplicate question ID {qid} in {path}")
        scores[qid] = record
    return scores


def valid_score(value):
    """Retain real zero scores; exclude missing, NaN, and infinite values."""
    return isinstance(value, (int, float)) and math.isfinite(value)


def _require_fields(record: dict, fields: tuple[str, ...], qid: str, label: str) -> None:
    """Raise a clear error when an expected comparison field is missing."""
    missing = [field for field in fields if field not in record]
    if missing:
        raise ValueError(
            f"Missing required fields for {qid} in {label}: {', '.join(missing)}"
        )


def compare(agent, baseline, baseline_name, metric, question_ids, group):
    # STEP 4: Match valid scores for the same questions. Wiki keys have no prefix.
    pairs = [
        (agent[qid][metric], baseline[qid][metric])
        for qid in question_ids
        if valid_score(agent[qid].get(metric)) and valid_score(baseline[qid].get(metric))
    ]

    # STEP 5: Positive differences favor the agent. Round numerical noise away.
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

    # STEP 6: Two-sided signed-rank test, excluding zero differences from ranking.
    # This is NOT a direct test of the arithmetic mean.
    test = wilcoxon(
        differences, alternative="two-sided", zero_method="wilcox", method="auto"
    )
    result["W"] = float(test.statistic)
    result["p_raw"] = float(test.pvalue)

    # STEP 7: Correct six planned tests in THIS group, not across all groups.
    result["p_adjusted"] = min(result["p_raw"] * NUMBER_OF_TESTS, 1.0)
    result["significant"] = result["p_adjusted"] < ALPHA
    return result


def save_results(rows, suffix):
    if not rows:
        return
    path = ROOT / f"wilcoxon_{suffix}_results"
    with path.with_suffix(".csv").open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    with path.with_suffix(".json").open("w", encoding="utf-8") as file:
        json.dump(rows, file, indent=2, allow_nan=False)


def format_value(value):
    return "N/A" if value is None else f"{value:.4g}"


def save_report(results):
    """Readable primary and question-type tables, with interpretation limits."""
    text = ["# Wiki-agent paired Wilcoxon results", "",
            "Primary population: all 111 saved questions, including within-track and cross-track questions. No separate cross-track test is performed.", "",
            "Agent: agent_all_eval.json. Baselines: archived k=8 and k=30 evaluations (paths and hashes in wilcoxon_run_manifest.json). The k=8 label follows metrics_run.md.", "",
            "Each row uses valid paired scores. Non-tied pairs exclude equal scores. Positive mean differences favor the agent. Bold adjusted p-values are below 0.05.", ""]
    groups = list(dict.fromkeys(row["group"] for row in results))
    for group in groups:
        title = "Overall (primary)" if group == "Overall" else group + " (exploratory)"
        text += [f"## {title}", "",
                 "| Baseline | Metric | Valid pairs | Non-tied pairs | Agent mean | Baseline mean | Mean difference | Adjusted p | Significant? |",
                 "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |"]
        for row in results:
            if row["group"] != group:
                continue
            p = format_value(row["p_adjusted"])
            if row["significant"]:
                p = f"**{p}**"
            decision = "Not tested" if row["p_raw"] is None else "Yes" if row["significant"] else "No"
            values = [row["baseline"], row["metric"], row["n_valid"], row["n_effective"],
                      format_value(row["agent_mean"]), format_value(row["baseline_mean"]),
                      format_value(row["mean_difference"]), p, decision]
            text.append("| " + " | ".join(map(str, values)) + " |")
        text.append("")
    text += ["## Limitations", "",
             "- Question-type tests are exploratory and require more than ten valid pairs per metric. Many tied pairs can still leave little information.",
             "- Bonferroni correction covers six planned comparisons separately within each group. It does not control false positives across all groups together.",
             "- Nonsignificance does not establish equality. Significance in one type but not another does not establish a difference between the types' system advantages.",
             "- The signed-rank test assumes independent question pairs and symmetry of the difference distribution for its usual location interpretation. Shared source modules and bounded scores may challenge these assumptions.",
             "- These are single saved evaluations. The test does not measure repeated-run variability or isolate which agent feature caused a difference.",
             "- Baselines are archived evaluations, while the agent is a later rerun. Matching IDs, questions, references, types and track labels is verified; this alone does not establish identical evaluator versions, prompts or runtime conditions across runs.", ""]
    (ROOT / "wilcoxon_report.md").write_text("\n".join(text), encoding="utf-8")


def main():
    agent = load_scores(AGENT_FILE)
    groups = {"Overall": sorted(agent)}
    for qid in sorted(agent):
        groups.setdefault(agent[qid]["type"], []).append(qid)
    groups = {group: ids for group, ids in groups.items()
              if group == "Overall" or len(ids) > MIN_TYPE_PAIRS}
    results = []
    for name, path in BASELINES.items():
        baseline = load_scores(path)
        # STEP 3: Verify complete matching; never silently intersect question IDs.
        if not agent or agent.keys() != baseline.keys():
            raise ValueError(f"Empty or mismatched question set for {name}")
        fields_to_match = ("user_input", "reference", "type", "evaluation_mode", "track")
        for qid in agent:
            _require_fields(agent[qid], fields_to_match, qid, AGENT_FILE.name)
            _require_fields(baseline[qid], fields_to_match, qid, path.name)
            for field in fields_to_match:
                if agent[qid][field] != baseline[qid][field]:
                    raise ValueError(f"Mismatched {field} for {qid}: {name}")
        for group, ids in groups.items():
            for metric in METRICS:
                results.append(compare(agent, baseline, name, metric, ids, group))

    # STEP 8: Print and save primary results separately from exploratory results.
    print(f"SciPy {scipy.__version__}; two-sided Wilcoxon, auto method, wilcox zeros")
    print(f"Bonferroni: {NUMBER_OF_TESTS} tests per group; adjusted p < {ALPHA}")
    for row in results:
        print(f"{row['group']:24} {row['baseline']:5} {row['metric']:20} "
              f"n={row['n_valid']:3} nonzero={row['n_effective']:3} "
              f"delta={format_value(row['mean_difference']):>9} "
              f"p_adj={format_value(row['p_adjusted']):>10} significant={row['significant']}")
    save_results([row for row in results if row["analysis"] == "primary"], "primary")
    save_results([row for row in results if row["analysis"] == "exploratory"], "question_type")
    save_report(results)
    manifest = {
        "created_at": datetime.now().isoformat(), "scipy_version": scipy.__version__,
        "alternative": "two-sided", "method": "auto", "zero_method": "wilcox",
        "difference_decimals": 12, "alpha": ALPHA, "tests_per_group": NUMBER_OF_TESTS,
        "primary_population": "All saved questions, including within_track and cross_track",
        "inputs": {path.relative_to(ROOT).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
                   for path in [AGENT_FILE, *BASELINES.values()]},
    }
    (ROOT / "wilcoxon_run_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Saved CSV/JSON results and report: {ROOT / 'wilcoxon_report.md'}")


if __name__ == "__main__":
    main()
