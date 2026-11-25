"""Routing evaluation: every question goes once through the full orchestrator (router + agents).

What the orchestrator returns:
    route   retrieve | clarify | out_of_scope (from the router),
            or fallback_general when every agent found no answer
    target  wiki | papers | both (the sources the router chose), null if not retrieve

Metrics (percent):
    router accuracy    router_route == expected_route (expected fallback_general counts as retrieve)
                       over all questions
    target accuracy    target == target_kb
                       over questions with expected_route = retrieve
    target coverage    the expected sources are among the chosen ones (target == target_kb or target == both)
                       over questions with expected_route = retrieve; secondary to target accuracy
    fallback accuracy  route == fallback_general
                       over questions with expected_route = fallback_general that the router sent to retrieve
    router confusion   rows: expected router route, columns: the router's route (counts)

Usage (from orchestrator/). Questions run one at a time (papers store allows one query at a time):
    uv run python evaluation/eval_router.py                         # dev set, topics scope
    uv run python evaluation/eval_router.py --scope-mode full
    uv run python evaluation/eval_router.py --split test --final    # once, on the frozen router
"""
import argparse
import json
import sys
import uuid
from datetime import datetime
from pathlib import Path

from langchain_core.messages import HumanMessage

from orchestrator.adapters import close as close_adapters
from orchestrator.config import ROUTER_MODEL
from orchestrator.graph import build_orchestrator

EVAL_DIR = Path(__file__).resolve().parent
RESULTS_DIR = EVAL_DIR / "results"
SPLITS = {"dev": EVAL_DIR / "routing_dev.json", "test": EVAL_DIR / "routing_test.json"}
ROUTES = ["retrieve", "clarify", "out_of_scope"]


def run_query(graph, query: str) -> dict:
    """Send one question through the orchestrator.

    route is the scored outcome; router_route is the router's own decision, saved to tell
    router misses from agent misses. sufficiency, agent_score, error and answer are saved
    for inspection only.
    """
    config = {"configurable": {"thread_id": uuid.uuid4().hex}}   # new conversation for every question
    state = graph.invoke({"messages": [HumanMessage(query)]}, config=config)
    decision = state["decision"]
    router_route = decision["route"]
    if any(r["error"] for r in state.get("results", [])):
        route = "error"                # an agent crashed: never count this as a correct fallback
    elif state["answer_type"] == "general_knowledge":
        route = "fallback_general"
    else:
        route = router_route
    sources = [t["source"] for t in decision["targets"]]
    target = "both" if len(sources) == 2 else (sources[0] if sources else None)
    return {"route": route, "router_route": router_route, "target": target,
            "reasoning": decision["reasoning"],
            "sufficiency": {r["source"]: r["sufficiency"] for r in state.get("results", [])},
            "agent_score": {r["source"]: r["agent_score"] for r in state.get("results", [])},
            "error": {r["source"]: r["error"] for r in state.get("results", []) if r["error"]},
            "answer": state["messages"][-1].content}


def accuracy(hits: list[bool]) -> str:
    # e.g. "75.0% (3/4)": the count shows how much one miss weighs on a small set.
    if not hits:
        return "n/a (0/0)"
    return f"{round(100 * sum(hits) / len(hits), 1)}% ({sum(hits)}/{len(hits)})"


def router_expected(expected_route: str) -> str:
    # For a fallback question the right router decision is retrieve: search first, then fall back.
    return "retrieve" if expected_route == "fallback_general" else expected_route


def confusion(results: list[dict]) -> dict:
    # table[expected][actual] = number of questions; shows where the router's misses go.
    table = {e: {a: 0 for a in ROUTES + ["error"]} for e in ROUTES}
    for r in results:
        actual = r["router_route"] if r["router_route"] in ROUTES else "error"
        table[router_expected(r["expected_route"])][actual] += 1
    return table


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description="Routing evaluation of the orchestrator.")
    parser.add_argument("--split", choices=list(SPLITS), default="dev")
    parser.add_argument("--scope-mode", choices=["topics", "full"], default="topics")
    parser.add_argument("--final", action="store_true", help="Required for --split test (run once).")
    args = parser.parse_args()
    if args.split == "test" and not args.final:
        parser.error("The test set is run once on the frozen router. Add --final to confirm.")

    queries = json.loads(SPLITS[args.split].read_text(encoding="utf-8"))["queries"]
    RESULTS_DIR.mkdir(exist_ok=True)
    out_path = RESULTS_DIR / f"routing_{args.split}_{args.scope_mode}_{datetime.now():%Y%m%d_%H%M%S}.json"
    run = {"split": args.split, "scope_mode": args.scope_mode, "model": ROUTER_MODEL,
           "date": f"{datetime.now():%Y-%m-%d %H:%M}", "results": []}

    graph = build_orchestrator(args.scope_mode)
    try:
        for i, q in enumerate(queries, 1):
            try:
                out = run_query(graph, q["query"])
            except Exception as e:
                out = {"route": "error", "router_route": "error", "target": None,
                       "reasoning": f"{type(e).__name__}: {e}", "sufficiency": {}, "agent_score": {},
                       "error": {"orchestrator": f"{type(e).__name__}: {e}"}, "answer": ""}
            route, router_route, target = out["route"], out["router_route"], out["target"]

            router_correct = router_route == router_expected(q["expected_route"])
            target_correct = target == q["target_kb"] if q["expected_route"] == "retrieve" else None
            fallback_correct = (route == "fallback_general"
                                if q["expected_route"] == "fallback_general" and router_route == "retrieve"
                                else None)
            ok = router_correct and target_correct is not False and fallback_correct is not False
            print(f"[{i:>3}/{len(queries)}] {'OK  ' if ok else 'MISS'} {q['id']}  "
                  f"route={route} (expected {q['expected_route']}, router {router_route})  "
                  f"target={target} (expected {q['target_kb']})  sufficiency={out['sufficiency']}")
            if not ok:
                print(f"      query:     {q['query']}\n      reasoning: {out['reasoning']}")

            run["results"].append({
                "id": q["id"], "query": q["query"],
                "expected_route": q["expected_route"], "route": route, "router_route": router_route,
                "target_kb": q["target_kb"], "target": target,
                "sufficiency": out["sufficiency"], "agent_score": out["agent_score"],
                "error": out["error"], "reasoning": out["reasoning"], "answer": out["answer"],
            })
            # Saved after every question, so an interrupted run keeps what it has done.
            out_path.write_text(json.dumps(run, indent=2, ensure_ascii=False), encoding="utf-8")
    finally:
        close_adapters()

    results = run["results"]
    run["router_accuracy"] = accuracy([r["router_route"] == router_expected(r["expected_route"])
                                       for r in results])
    run["target_accuracy"] = accuracy([r["target"] == r["target_kb"]
                                       for r in results if r["expected_route"] == "retrieve"])
    # "both" covers wiki and papers: the needed source was asked, only extra time was spent.
    run["target_coverage"] = accuracy([r["target"] in (r["target_kb"], "both")
                                       for r in results if r["expected_route"] == "retrieve"])
    run["fallback_accuracy"] = accuracy([r["route"] == "fallback_general" for r in results
                                         if r["expected_route"] == "fallback_general"
                                         and r["router_route"] == "retrieve"])
    run["router_confusion"] = confusion(results)
    out_path.write_text(json.dumps(run, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"\nsplit={args.split}  scope={args.scope_mode}  model={ROUTER_MODEL}  n={len(results)}")
    print(f"Router accuracy   {run['router_accuracy']}")
    print(f"Target accuracy   {run['target_accuracy']}")
    print(f"Target coverage   {run['target_coverage']}")
    print(f"Fallback accuracy {run['fallback_accuracy']}")

    columns = ROUTES + ["error"]
    print("\nRouter confusion (rows: expected, columns: router)")
    print(f"{'':14}" + "".join(f"{c:>14}" for c in columns))
    for expected, row in run["router_confusion"].items():
        print(f"{expected:14}" + "".join(f"{row[c]:>14}" for c in columns))
    print(f"\n[saved] {out_path}")


if __name__ == "__main__":
    main()
