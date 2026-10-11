"""Compare two search reports using one pooled, frozen relevance judgment."""
from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path

import asyncpg
import httpx

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import evaluate_search_quality as evaluator
from api import main as search_api


def _mean(rows: list[dict], key: str) -> float:
    return round(sum(row[key] for row in rows) / len(rows), 4) if rows else 0.0


async def _run(args: argparse.Namespace) -> None:
    baseline = json.loads(args.baseline.read_text())
    candidate = json.loads(args.candidate.read_text())
    cases = {
        case["id"]: case
        for case in json.loads(evaluator.CASES_PATH.read_text())
    }
    baseline_cases = {case["id"]: case for case in baseline["cases"]}
    candidate_cases = {case["id"]: case for case in candidate["cases"]}
    common_ids = [case_id for case_id in baseline_cases if case_id in candidate_cases]
    if not common_ids:
        raise ValueError("Reports have no cases in common")

    frozen_qrels = json.loads(args.qrels_input.read_text()) if args.qrels_input else None
    api_key = search_api.OPENROUTER_API_KEY
    if frozen_qrels is None or args.judge_missing:
        if not api_key:
            raise RuntimeError("OPENROUTER_API_KEY is not configured")
        search_api.http_client = httpx.AsyncClient(timeout=120)
        search_api.db_pool = await asyncpg.create_pool(
            host=search_api.DB_HOST,
            port=search_api.DB_PORT,
            user=search_api.DB_USER,
            password=search_api.DB_PASSWORD,
            database=search_api.DB_NAME,
            min_size=1,
            max_size=5,
            statement_cache_size=0,
            command_timeout=60,
            init=search_api._init_connection,
        )
    checkpoint = (
        json.loads(args.output.read_text())
        if args.resume and args.output.exists()
        else {}
    )
    comparisons = checkpoint.get("cases", [])
    completed_ids = {row["id"] for row in comparisons}
    qrels = (
        json.loads(args.qrels_output.read_text())
        if args.resume and args.qrels_output.exists()
        else dict(frozen_qrels or {})
    )
    total_cost = float(checkpoint.get("judge_cost_usd") or 0)
    try:
        pending_ids = [case_id for case_id in common_ids if case_id not in completed_ids]
        for number, case_id in enumerate(pending_ids, start=1):
            case = cases[case_id]
            before = baseline_cases[case_id]
            after = candidate_cases[case_id]
            doc_ids = sorted({
                result["doc_id"]
                for report_case in (before, after)
                for result in report_case["results"]
            })
            if frozen_qrels is not None:
                case_qrels = dict(frozen_qrels.get(case_id, {}))
                missing = [doc_id for doc_id in doc_ids if doc_id not in case_qrels]
                if missing and args.judge_missing:
                    # Grade only documents the frozen labels have not seen, so
                    # every variant is scored against one consistent label set.
                    docs_by_id = await search_api._fetch_docs(missing)
                    judged_ids = [doc_id for doc_id in missing if doc_id in docs_by_id]
                    grades, usage = await evaluator._judge(
                        search_api.http_client,
                        api_key,
                        args.judge_model,
                        case,
                        [docs_by_id[doc_id] for doc_id in judged_ids],
                    )
                    total_cost += float(usage.get("cost") or 0)
                    for idx, doc_id in enumerate(judged_ids):
                        case_qrels[doc_id] = {
                            "grade": grades[idx]["relevance"],
                            "reason": grades[idx]["reason"],
                        }
                grades_by_id = {
                    doc_id: case_qrels.get(doc_id, {}).get("grade", 0)
                    for doc_id in doc_ids
                }
                qrels[case_id] = case_qrels
            else:
                docs_by_id = await search_api._fetch_docs(doc_ids)
                judged_ids = [doc_id for doc_id in doc_ids if doc_id in docs_by_id]
                judged_docs = [docs_by_id[doc_id] for doc_id in judged_ids]
                grades, usage = await evaluator._judge(
                    search_api.http_client,
                    api_key,
                    args.judge_model,
                    case,
                    judged_docs,
                )
                total_cost += float(usage.get("cost") or 0)
                grades_by_id = {
                    doc_id: grades[idx]["relevance"]
                    for idx, doc_id in enumerate(judged_ids)
                }
                qrels[case_id] = {
                    doc_id: {
                        "grade": grades[idx]["relevance"],
                        "reason": grades[idx]["reason"],
                    }
                    for idx, doc_id in enumerate(judged_ids)
                }
            before_grades = [grades_by_id.get(row["doc_id"], 0) for row in before["results"]]
            after_grades = [grades_by_id.get(row["doc_id"], 0) for row in after["results"]]
            before_metrics = evaluator._metrics(before_grades)
            after_metrics = evaluator._metrics(after_grades)
            delta = {
                key: round(after_metrics[key] - before_metrics[key], 4)
                for key in before_metrics
            }
            comparisons.append({
                "id": case_id,
                "baseline": before_metrics,
                "candidate": after_metrics,
                "delta": delta,
            })
            args.output.write_text(json.dumps({
                "incomplete": True,
                "judge_cost_usd": round(total_cost, 6),
                "cases": comparisons,
            }, ensure_ascii=False, indent=2) + "\n")
            args.qrels_output.write_text(json.dumps(qrels, ensure_ascii=False, indent=2) + "\n")
            print(
                f"[{number:02d}/{len(pending_ids):02d}] {case_id}: "
                f"ΔP@5={delta['precision_at_5']:+.2f} "
                f"ΔnDCG={delta['ndcg_at_10']:+.2f}"
            )
    finally:
        if search_api.db_pool is not None:
            await search_api.db_pool.close()
            search_api.db_pool = None
        if search_api.http_client is not None:
            await search_api.http_client.aclose()
            search_api.http_client = None

    metric_names = ["ndcg_at_10", "precision_at_5", "success_at_5", "mean_relevance_at_5"]
    baseline_aggregate = {
        key: _mean([row["baseline"] for row in comparisons], key) for key in metric_names
    }
    candidate_aggregate = {
        key: _mean([row["candidate"] for row in comparisons], key) for key in metric_names
    }
    regressions = [
        row for row in comparisons
        if row["delta"]["success_at_5"] < 0
        or row["delta"]["precision_at_5"] < 0
        or row["delta"]["mean_relevance_at_5"] < -0.2
    ]
    ranking_warnings = [
        row for row in comparisons
        if row["delta"]["ndcg_at_10"] < -0.1
    ]
    output = {
        "configuration": {
            "cases": len(comparisons),
            "judge_model": args.judge_model,
            "baseline": str(args.baseline),
            "candidate": str(args.candidate),
        },
        "baseline": baseline_aggregate,
        "candidate": candidate_aggregate,
        "delta": {
            key: round(candidate_aggregate[key] - baseline_aggregate[key], 4)
            for key in metric_names
        },
        "judge_free": {
            "baseline": baseline.get("judge_free_by_mode"),
            "candidate": candidate.get("judge_free_by_mode"),
        },
        "passes_regression_guard": not regressions,
        "regressions": regressions,
        "ranking_warnings": ranking_warnings,
        "judge_cost_usd": round(total_cost, 6),
        "cases": comparisons,
    }
    args.output.write_text(json.dumps(output, ensure_ascii=False, indent=2) + "\n")
    args.qrels_output.write_text(json.dumps(qrels, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({
        "baseline": baseline_aggregate,
        "candidate": candidate_aggregate,
        "delta": output["delta"],
        "judge_free": output["judge_free"],
        "passes_regression_guard": not regressions,
        "regressions": len(regressions),
        "ranking_warnings": len(ranking_warnings),
    }, ensure_ascii=False, indent=2))
    if args.fail_on_regression and regressions:
        raise SystemExit(1)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--qrels-output", type=Path, required=True)
    parser.add_argument("--qrels-input", type=Path)
    parser.add_argument(
        "--judge-missing",
        action="store_true",
        help="With --qrels-input, judge documents the frozen labels do not cover",
    )
    parser.add_argument("--fail-on-regression", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--judge-model", default=evaluator.DEFAULT_JUDGE_MODEL)
    return parser.parse_args()


if __name__ == "__main__":
    asyncio.run(_run(_parse_args()))
