"""
Evaluation & Benchmarking API Router for SupportIQ AI.
Provides baseline comparisons, confusion matrices, human agreement statistics, and golden set inspection.
"""
from fastapi import APIRouter, BackgroundTasks, Query
from typing import Dict, Any, List, Optional
import os
import json
import pandas as pd
import subprocess
import sys

router = APIRouter(prefix="/api/evaluation", tags=["Evaluation & Benchmarking"])


@router.get("")
async def get_evaluation_results() -> Dict[str, Any]:
    """Retrieve multi-system benchmark results, confusion matrices, and judge scores."""
    benchmark_path = "data/evaluation/benchmark_results.json"
    if not os.path.exists(benchmark_path):
        return {
            "status": "NOT_EVALUATED",
            "message": "No evaluation results available. Run evaluation pipeline to benchmark models.",
            "benchmark_table": [],
            "supportiq_metrics": {},
        }

    with open(benchmark_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    return {
        "status": "EVALUATED",
        "benchmark_table": data.get("benchmark_table", []),
        "supportiq_metrics": data.get("supportiq_full_eval", {}),
        "majority_eval": data.get("majority_eval", {}),
        "tfidf_eval": data.get("tfidf_eval", {}),
    }


@router.get("/golden")
async def get_golden_set(
    difficulty: Optional[str] = Query(None, description="Filter by difficulty (easy, medium, hard, adversarial)"),
    intent: Optional[str] = Query(None, description="Filter by gold intent"),
    action: Optional[str] = Query(None, description="Filter by gold action (AUTO, HUMAN)"),
    limit: int = Query(50, ge=1, le=200),
    offset: int = Query(0, ge=0),
) -> Dict[str, Any]:
    """Inspect the 200-example Golden Evaluation Set."""
    golden_path = "data/golden/golden_set.csv"
    if not os.path.exists(golden_path):
        return {"total_count": 0, "items": []}

    df = pd.read_csv(golden_path)
    records = df.to_dict(orient="records")

    filtered = records
    if difficulty and difficulty != "all":
        filtered = [r for r in filtered if str(r.get("difficulty", "")).lower() == difficulty.lower()]
    if intent and intent != "all":
        filtered = [r for r in filtered if str(r.get("gold_intent", "")).lower() == intent.lower()]
    if action and action != "all":
        filtered = [r for r in filtered if str(r.get("gold_action", "")).upper() == action.upper()]

    total_count = len(filtered)
    paginated = filtered[offset : offset + limit]

    return {
        "total_count": total_count,
        "items": paginated,
        "offset": offset,
        "limit": limit,
    }


def _run_eval_job():
    py_bin = r"C:\Users\aryan\AppData\Local\Python\bin\python.exe"
    if not os.path.exists(py_bin):
        py_bin = sys.executable
    subprocess.run([py_bin, "scripts/evaluate.py"], check=True)


@router.post("/run")
async def trigger_evaluation_pipeline(bg_tasks: BackgroundTasks) -> Dict[str, Any]:
    """Trigger background evaluation job."""
    bg_tasks.add_task(_run_eval_job)
    return {
        "status": "RUNNING",
        "message": "Evaluation pipeline triggered across all 5 benchmark systems and 200 Golden cases.",
    }
