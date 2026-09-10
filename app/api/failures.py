"""
Failure Analysis & Error Diagnosis API Router for SupportIQ AI.
Provides diagnostic drill-downs into the 5 core failure modes with root-cause explanations, proposed fixes, and manual tagging.
"""
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Dict, Any, List, Optional
import os
import json
from app.database import SessionLocal, FailureTag

router = APIRouter(prefix="/api/failures", tags=["Failure Analysis"])


class TagRequest(BaseModel):
    user_tag: str
    user_notes: Optional[str] = None


@router.get("")
async def get_failure_analysis() -> Dict[str, Any]:
    """Retrieve top failure categories and case records."""
    benchmark_path = "data/evaluation/benchmark_results.json"
    if not os.path.exists(benchmark_path):
        return {
            "status": "NOT_EVALUATED",
            "total_failures": 0,
            "category_counts": {},
            "failure_cases": [],
        }

    with open(benchmark_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    supportiq_eval = data.get("supportiq_full_eval", {})
    failures_rep = supportiq_eval.get("failure_analysis", {})

    # Load manual user tags from database
    session = SessionLocal()
    try:
        tags = session.query(FailureTag).all()
        tag_map = {t.id: {"user_tag": t.user_tag, "user_notes": t.user_notes} for t in tags}
    finally:
        session.close()

    cases = failures_rep.get("failure_cases", [])
    for c in cases:
        c_id = str(c.get("id"))
        if c_id in tag_map:
            c["user_tag"] = tag_map[c_id]["user_tag"]
            c["user_notes"] = tag_map[c_id]["user_notes"]

    return {
        "status": "EVALUATED",
        "total_evaluated": failures_rep.get("total_evaluated", 0),
        "total_failures": failures_rep.get("total_failures", 0),
        "failure_rate_pct": round(failures_rep.get("failure_rate", 0.0) * 100, 2),
        "category_counts": failures_rep.get("category_counts", {}),
        "failure_cases": cases,
    }


@router.post("/{case_id}/tag")
async def tag_failure_case(case_id: str, req: TagRequest) -> Dict[str, Any]:
    """Save manual supervisor tag and diagnostic notes for a failure case."""
    session = SessionLocal()
    try:
        existing = session.query(FailureTag).filter(FailureTag.id == case_id).first()
        if existing:
            existing.user_tag = req.user_tag
            existing.user_notes = req.user_notes
        else:
            new_tag = FailureTag(id=case_id, user_tag=req.user_tag, user_notes=req.user_notes)
            session.add(new_tag)
        session.commit()
        return {"status": "SUCCESS", "case_id": case_id, "user_tag": req.user_tag}
    finally:
        session.close()
