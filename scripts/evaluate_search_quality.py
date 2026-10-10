"""Evaluate Portuguese caselaw search against a fixed relevance suite.

The judge is intentionally different from the production reranker. Reports
contain only IDs, grades, and metrics, making before/after runs comparable.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import math
import os
import sys
import time
from pathlib import Path
from typing import Any, Optional

import asyncpg
import httpx
from fastapi import HTTPException

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from api import main as search_api

CASES_PATH = ROOT / "tests" / "search_quality_cases.json"
DEFAULT_JUDGE_MODEL = "google/gemini-3.1-pro-preview"


def _candidate_text(result: dict[str, Any]) -> str:
    metadata = result.get("metadata") or {}
    descriptors = metadata.get("legal_descriptors") or []
    citations = metadata.get("legislation_cited") or []
    citation_text = ", ".join(
        " ".join(
            str(value) for value in (item.get("law"), item.get("article")) if value
        )
        for item in citations
        if isinstance(item, dict)
    )
    fields = [
        f"Tribunal: {result.get('court_short') or ''}",
        f"Data: {result.get('decision_date') or ''}",
        f"Domínio: {result.get('legal_domain') or ''}",
        f"Resumo: {result.get('summary') or ''}",
        f"Questão jurídica: {metadata.get('legal_question') or ''}",
        f"Decisão: {metadata.get('decision_outcome') or ''}",
        f"Fundamentação determinante: {metadata.get('ratio_decidendi') or ''}",
        f"Descritores: {', '.join(str(x) for x in descriptors)}",
        f"Legislação citada: {citation_text}",
    ]
    return "\n".join(fields)[:12000]


def _case_query(case: dict[str, Any]) -> str:
    mode = case.get("mode", "hybrid")
    if mode == "legislation":
        refs = [
            ref if isinstance(ref, str) else f"{ref.get('law', '')} {ref.get('article', '')}".strip()
            for ref in case["articles"]
        ]
        operator = " E " if case.get("match") == "all" else " OU "
        return f"Acórdãos que citem {operator.join(refs)}"
    return case.get("q_semantic") or case.get("q") or ""


async def _judge(
    client: httpx.AsyncClient,
    api_key: str,
    model: str,
    case: dict[str, Any],
    results: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    candidates = "\n\n".join(
        f"RESULTADO {idx}\n{_candidate_text(result)}"
        for idx, result in enumerate(results)
    )
    prompt = f"""Avalie resultados de pesquisa de jurisprudência portuguesa.

Consulta: {_case_query(case)}
Critério de relevância: {case['relevance']}

Classifique cada resultado independentemente:
0 = irrelevante
1 = relacionado, mas não responde ao critério
2 = útil e responde parcialmente
3 = diretamente relevante

Não premie um resultado apenas por conter palavras da consulta. Avalie o tema
efetivamente decidido no acórdão. Devolva uma classificação para cada índice.

{candidates}"""
    schema = {
        "name": "caselaw_relevance_grades",
        "strict": True,
        "schema": {
            "type": "object",
            "properties": {
                "grades": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "index": {"type": "integer"},
                            "relevance": {"type": "integer", "minimum": 0, "maximum": 3},
                            "reason": {"type": "string"},
                        },
                        "required": ["index", "relevance", "reason"],
                        "additionalProperties": False,
                    },
                }
            },
            "required": ["grades"],
            "additionalProperties": False,
        },
    }
    body = None
    content = None
    for attempt in range(3):
        try:
            response = await client.post(
                f"{search_api.OPENROUTER_BASE}/chat/completions",
                headers={
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type": "application/json",
                    "HTTP-Referer": "https://github.com/pt-caselaw-dgsi",
                    "X-Title": "pt-caselaw-dgsi search evaluation",
                },
                json={
                    "model": model,
                    "messages": [{"role": "user", "content": prompt}],
                    "temperature": 0,
                    "response_format": {"type": "json_schema", "json_schema": schema},
                },
                timeout=120,
            )
        except httpx.RequestError:
            if attempt == 2:
                raise
            await asyncio.sleep(1.0 * (2**attempt))
            continue
        if response.status_code in {429, 500, 502, 503, 504} and attempt < 2:
            await asyncio.sleep(0.5 * (2**attempt))
            continue
        response.raise_for_status()
        body = response.json()
        content = ((body.get("choices") or [{}])[0].get("message") or {}).get("content")
        if content:
            break
        if attempt < 2:
            await asyncio.sleep(0.5 * (2**attempt))
    if not content or body is None:
        raise RuntimeError("Judge returned no structured content after 3 attempts")
    parsed = json.loads(content)
    by_index = {
        grade["index"]: grade
        for grade in parsed["grades"]
        if 0 <= grade["index"] < len(results)
    }
    grades = [
        by_index.get(i, {"index": i, "relevance": 0, "reason": "Sem classificação"})
        for i in range(len(results))
    ]
    return grades, body.get("usage") or {}


def _dcg(grades: list[int], k: int) -> float:
    return sum((2**grade - 1) / math.log2(rank + 2) for rank, grade in enumerate(grades[:k]))


def _duplicate_key(result: dict[str, Any]) -> Optional[tuple]:
    """Same court + process number + decision date = the same decision
    published more than once on DGSI."""
    if not result.get("process_number"):
        return None
    return (result.get("court_short"), result["process_number"], result.get("decision_date"))


def _judge_free_metrics(results: list[dict[str, Any]]) -> dict[str, float]:
    """Metrics that need no relevance labels, comparable across runs."""
    top = results[:10]
    seen: set[tuple] = set()
    duplicates = 0
    for result in top:
        key = _duplicate_key(result)
        if key is None:
            continue
        if key in seen:
            duplicates += 1
        seen.add(key)
    return {
        "duplicate_rate_at_10": round(duplicates / max(1, len(top)), 4),
        "result_count": len(results),
    }


def _percentile(values: list[float], pct: float) -> float:
    ordered = sorted(values)
    return round(ordered[min(len(ordered) - 1, int(round(pct * (len(ordered) - 1))))], 3)


def _metrics(grades: list[int]) -> dict[str, float]:
    ideal = sorted(grades, reverse=True)
    dcg10 = _dcg(grades, 10)
    ideal10 = _dcg(ideal, 10)
    top5 = grades[:5]
    return {
        "ndcg_at_10": round(dcg10 / ideal10, 4) if ideal10 else 0.0,
        "precision_at_5": round(sum(g >= 2 for g in top5) / max(1, len(top5)), 4),
        "success_at_5": float(any(g >= 2 for g in top5)),
        "mean_relevance_at_5": round(sum(top5) / max(1, len(top5)), 4),
    }


async def _search_case(
    case: dict[str, Any],
    results: int,
    overfetch: int,
    weights: Optional[dict[str, float]] = None,
) -> list[dict[str, Any]]:
    mode = case.get("mode", "hybrid")
    common = {
        "limit": results,
        "filters": case.get("filters"),
        "courts": case.get("courts"),
        "from_date": case.get("from_date"),
        "to_date": case.get("to_date"),
        "is_auj": case.get("is_auj"),
    }
    if mode == "hybrid":
        extra = {"weights": search_api.SearchWeights(**weights)} if weights else {}
        response = await search_api.search_hybrid(search_api.SearchRequest(
            q_semantic=case["q_semantic"],
            q_keywords=case["q_keywords"],
            overfetch=overfetch,
            **common,
            **extra,
        ))
    elif mode == "legislation":
        response = await search_api.search_legislation(search_api.LegislationSearchRequest(
            articles=case["articles"],
            match=case.get("match", "any"),
            **common,
        ))
    elif mode == "legislation_semantic":
        response = await search_api.search_legislation_semantic(
            search_api.LegislationSemanticSearchRequest(
                q=case["q"],
                top_k=case.get("top_k", 20),
                **common,
            )
        )
    else:
        raise ValueError(f"Unsupported mode {mode!r} in {case['id']}")
    return [result.model_dump(mode="json") for result in response.results]


async def _run(args: argparse.Namespace) -> None:
    cases = json.loads(CASES_PATH.read_text())
    if args.mode:
        cases = [case for case in cases if case.get("mode", "hybrid") in args.mode]
    if args.id_prefix:
        cases = [case for case in cases if case["id"].startswith(args.id_prefix)]
    if args.case:
        requested = set(args.case)
        cases = [case for case in cases if case["id"] in requested]
        missing = requested - {case["id"] for case in cases}
        if missing:
            raise ValueError(f"Unknown cases: {', '.join(sorted(missing))}")
    if args.limit_cases:
        cases = cases[: args.limit_cases]
    api_key = search_api.OPENROUTER_API_KEY
    if not api_key:
        raise RuntimeError("OPENROUTER_API_KEY is not configured")

    if args.disable_reranker:
        search_api.RERANK_WEIGHT = 0
        search_api.LEGISLATION_RERANK_WEIGHT = 0
    frozen_qrels = json.loads(args.qrels.read_text()) if args.qrels else None
    search_api.http_client = httpx.AsyncClient(timeout=60)
    search_api.db_pool = await asyncpg.create_pool(
        host=search_api.DB_HOST,
        port=search_api.DB_PORT,
        user=search_api.DB_USER,
        password=search_api.DB_PASSWORD,
        database=search_api.DB_NAME,
        min_size=1,
        max_size=4,
        statement_cache_size=0,
        command_timeout=60,
        init=search_api._init_connection,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    checkpoint = (
        json.loads(args.output.read_text())
        if args.resume and args.output.exists()
        else {}
    )
    reports = checkpoint.get("cases", [])
    completed_ids = {report["id"] for report in reports}
    total_cost = float(checkpoint.get("judge_cost_usd") or 0)
    started = time.perf_counter()
    try:
        pending_cases = [case for case in cases if case["id"] not in completed_ids]
        for number, case in enumerate(pending_cases, start=1):
            before = time.perf_counter()
            for search_attempt in range(3):
                try:
                    result_dicts = await _search_case(
                        case, args.results, args.overfetch, args.weights
                    )
                    break
                except HTTPException as exc:
                    if "429" not in str(exc.detail) or search_attempt == 2:
                        raise
                    await asyncio.sleep(30)
                except (asyncpg.PostgresError, ConnectionError, OSError) as exc:
                    transient = any(
                        marker in str(exc).lower()
                        for marker in ("connection", "closed", "max clients", "emaxconn")
                    )
                    if not transient or search_attempt == 2:
                        raise
                    await asyncio.sleep(15)
            if args.skip_judge:
                grades = [
                    {"index": idx, "relevance": 0, "reason": "Julgamento adiado para comparação pooled"}
                    for idx, _ in enumerate(result_dicts)
                ]
            elif frozen_qrels is not None:
                case_qrels = frozen_qrels.get(case["id"], {})
                grades = [
                    {
                        "index": idx,
                        "relevance": case_qrels.get(result["doc_id"], {}).get("grade", 0),
                        "reason": case_qrels.get(result["doc_id"], {}).get(
                            "reason", "Documento ainda não julgado"
                        ),
                    }
                    for idx, result in enumerate(result_dicts)
                ]
            else:
                grades, usage = await _judge(
                    search_api.http_client, api_key, args.judge_model, case, result_dicts
                )
                total_cost += float(usage.get("cost") or 0)
            numeric_grades = [grade["relevance"] for grade in grades]
            metrics = {**_metrics(numeric_grades), **_judge_free_metrics(result_dicts)}
            expected_process = case.get("expected_process_number")
            if expected_process:
                expected_rank = next(
                    (
                        idx for idx, result in enumerate(result_dicts, start=1)
                        if result.get("process_number") == expected_process
                    ),
                    None,
                )
                metrics["reciprocal_rank"] = (
                    round(1 / expected_rank, 4) if expected_rank is not None else 0.0
                )
            reports.append({
                "id": case["id"],
                "mode": case.get("mode", "hybrid"),
                "query": _case_query(case),
                "metrics": metrics,
                "latency_seconds": round(time.perf_counter() - before, 3),
                "results": [
                    {
                        "rank": idx + 1,
                        "doc_id": result["doc_id"],
                        "court": result.get("court_short"),
                        "process_number": result.get("process_number"),
                        "decision_date": result.get("decision_date"),
                        "legal_domain": result.get("legal_domain"),
                        "grade": grades[idx]["relevance"],
                        "reason": grades[idx]["reason"],
                    }
                    for idx, result in enumerate(result_dicts)
                ],
            })
            args.output.write_text(json.dumps({
                "incomplete": True,
                "judge_cost_usd": round(total_cost, 6),
                "cases": reports,
            }, ensure_ascii=False, indent=2) + "\n")
            print(
                f"[{number:02d}/{len(pending_cases):02d}] {case['id']}: "
                f"P@5={metrics['precision_at_5']:.2f} "
                f"nDCG@10={metrics['ndcg_at_10']:.2f}"
            )
            if args.delay:
                await asyncio.sleep(args.delay)
    finally:
        try:
            await search_api.db_pool.close()
        except OSError as exc:
            print(f"Warning: database pool close failed: {exc}")
        try:
            await search_api.http_client.aclose()
        except OSError as exc:
            print(f"Warning: HTTP client close failed: {exc}")
        search_api.db_pool = None
        search_api.http_client = None

    metric_names = ["ndcg_at_10", "precision_at_5", "success_at_5", "mean_relevance_at_5"]
    aggregate = {
        name: round(sum(report["metrics"][name] for report in reports) / len(reports), 4)
        for name in metric_names
    }
    modes = sorted({report["mode"] for report in reports})
    aggregate_by_mode = {
        mode: {
            name: round(
                sum(report["metrics"][name] for report in reports if report["mode"] == mode)
                / sum(report["mode"] == mode for report in reports),
                4,
            )
            for name in metric_names
        }
        for mode in modes
    }
    judge_free = {
        mode: {
            "duplicate_rate_at_10": round(
                sum(r["metrics"]["duplicate_rate_at_10"] for r in reports if r["mode"] == mode)
                / sum(r["mode"] == mode for r in reports), 4,
            ),
            "mean_result_count": round(
                sum(r["metrics"]["result_count"] for r in reports if r["mode"] == mode)
                / sum(r["mode"] == mode for r in reports), 2,
            ),
            "latency_p50_seconds": _percentile(
                [r["latency_seconds"] for r in reports if r["mode"] == mode], 0.5),
            "latency_p95_seconds": _percentile(
                [r["latency_seconds"] for r in reports if r["mode"] == mode], 0.95),
        }
        for mode in modes
    }
    output = {
        "configuration": {
            "cases": len(cases),
            "results_per_case": args.results,
            "overfetch": args.overfetch,
            "weights": args.weights,
            "judge_model": args.judge_model,
            "embedding_model": search_api.EMBEDDING_MODEL,
            "reranker_enabled": not args.disable_reranker,
            "reranker_model": search_api.RERANK_MODEL,
            "reranker_weight": search_api.RERANK_WEIGHT,
            "legislation_reranker_weight": search_api.LEGISLATION_RERANK_WEIGHT,
            "qrels": str(args.qrels) if args.qrels else None,
            "judge_skipped": args.skip_judge,
        },
        "aggregate": aggregate,
        "aggregate_by_mode": aggregate_by_mode,
        "judge_free_by_mode": judge_free,
        "elapsed_seconds": round(time.perf_counter() - started, 3),
        "judge_cost_usd": round(total_cost, 6),
        "cases": reports,
    }
    args.output.write_text(json.dumps(output, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(aggregate, ensure_ascii=False, indent=2))
    print(json.dumps(judge_free, ensure_ascii=False, indent=2))
    print(f"Report: {args.output}")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--judge-model", default=os.getenv("SEARCH_EVAL_JUDGE_MODEL", DEFAULT_JUDGE_MODEL))
    parser.add_argument("--results", type=int, default=10)
    parser.add_argument("--overfetch", type=int, default=8)
    parser.add_argument("--limit-cases", type=int)
    parser.add_argument("--case", action="append", help="Run only this case ID; repeatable")
    parser.add_argument("--id-prefix", help="Run only case IDs starting with this prefix")
    parser.add_argument("--delay", type=float, default=0, help="Delay between cases in seconds")
    parser.add_argument("--resume", action="store_true", help="Resume an incomplete output report")
    parser.add_argument(
        "--mode",
        action="append",
        choices=["hybrid", "legislation", "legislation_semantic"],
        help="Run only this search mode; repeatable",
    )
    parser.add_argument(
        "--weights",
        type=json.loads,
        help='Hybrid source weights as JSON, e.g. \'{"embedding":0,"embedding_context":0,'
             '"embedding_ratio":0,"fts":1}\' to benchmark FTS alone',
    )
    parser.add_argument(
        "--disable-reranker",
        action="store_true",
        help="Measure the retrieval-only baseline for hybrid cases",
    )
    parser.add_argument(
        "--qrels",
        type=Path,
        help="Use frozen relevance labels instead of calling the judge",
    )
    parser.add_argument(
        "--skip-judge",
        action="store_true",
        help="Collect rankings only; judge their pooled union during comparison",
    )
    return parser.parse_args()


if __name__ == "__main__":
    asyncio.run(_run(_parse_args()))
