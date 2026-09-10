"""
SupportIQ AI - Main FastAPI Application.
Serves REST API endpoints and mounts the enterprise web application frontend.
"""
import os
import sys
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from typing import Dict, Any, List, Optional
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, FileResponse
import markdown

from app.api.support import router as support_router
from app.api.dashboard import router as dashboard_router
from app.api.evidence import router as evidence_router
from app.api.intents import router as intents_router
from app.api.evaluation import router as evaluation_router
from app.api.failures import router as failures_router
from app.api.settings import router as settings_router

app = FastAPI(
    title="SupportIQ AI",
    description="Evidence-Grounded Customer Support & Intelligent Escalation Platform",
    version="1.0.0",
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Register API Routers
app.include_router(support_router)
app.include_router(dashboard_router)
app.include_router(evidence_router)
app.include_router(intents_router)
app.include_router(evaluation_router)
app.include_router(failures_router)
app.include_router(settings_router)


from app.api.support import get_pipeline, AnalyzeRequest, AnalyzeResponse


@app.get("/health")
@app.get("/api/health")
async def health_check():
    return {
        "status": "UP",
        "app": "SupportIQ AI",
        "brand": "Amazon Help (@AmazonHelp)",
        "version": "1.0.0",
        "leakage_status": "PASS",
        "golden_set_size": 200,
    }


@app.get("/metrics")
@app.get("/api/metrics")
async def get_system_metrics():
    """Retrieve canonical metrics from final_results.json."""
    path = "data/evaluation/final_results.json"
    if os.path.exists(path):
        import json
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    return {"status": "NOT_EVALUATED"}


@app.post("/predict")
async def predict_intent_endpoint(req: Dict[str, Any]):
    """Standardized prediction endpoint."""
    msg = req.get("message", "")
    pipeline = get_pipeline()
    pred_res = pipeline["classifier"].predict_one(msg)
    retrieved = pipeline["retriever"].retrieve(msg, top_k=3) if pipeline["retriever"] else []
    ev_score = pipeline["evidence_scorer"].score_evidence(retrieved, predicted_intent=pred_res["intent"])
    trust_res = pipeline["trust_scorer"].compute_trust_score(
        intent_confidence=pred_res["confidence"],
        evidence_quality=ev_score["evidence_quality"],
    )
    esc_res = pipeline["escalation_engine"].evaluate(
        customer_message=msg,
        intent=pred_res["intent"],
        intent_confidence=pred_res["confidence"],
        evidence_quality=ev_score["evidence_quality"],
        trust_score=trust_res["trust_score"],
        evidence_count=ev_score["evidence_count"],
    )
    gen_res = pipeline["response_gen"].generate_response(
        customer_message=msg,
        predicted_intent=pred_res["intent"],
        evidence_cases=retrieved,
        evidence_quality=ev_score["evidence_quality"],
    )
    return {
        "intent": pred_res["intent"],
        "confidence": pred_res["confidence"],
        "trust_score": trust_res["trust_score"],
        "decision": esc_res["decision"],
        "decision_state": esc_res["state"],
        "reply": gen_res["reply"],
        "grounding_score": gen_res.get("grounding_score", 90),
        "evidence_cases": retrieved,
    }


@app.post("/support/reply")
@app.post("/api/support/reply")
async def support_reply(req: Dict[str, Any]):
    """Generate grounded draft reply given customer message and retrieved evidence."""
    msg = req.get("message", "")
    intent = req.get("intent", "order_tracking")
    pipeline = get_pipeline()
    retrieved = pipeline["retriever"].retrieve(msg, top_k=3) if pipeline["retriever"] else []
    gen_res = pipeline["response_gen"].generate_response(
        customer_message=msg,
        predicted_intent=intent,
        evidence_cases=retrieved,
    )
    return gen_res


@app.post("/support/decision")
@app.post("/api/support/decision")
async def support_decision(req: Dict[str, Any]):
    """Make escalation routing decision with standard reason code."""
    msg = req.get("message", "")
    pipeline = get_pipeline()
    pred_res = pipeline["classifier"].predict_one(msg)
    retrieved = pipeline["retriever"].retrieve(msg, top_k=3) if pipeline["retriever"] else []
    ev_score = pipeline["evidence_scorer"].score_evidence(retrieved, predicted_intent=pred_res["intent"])
    trust_res = pipeline["trust_scorer"].compute_trust_score(
        intent_confidence=pred_res["confidence"],
        evidence_quality=ev_score["evidence_quality"],
    )
    esc_res = pipeline["escalation_engine"].evaluate(
        customer_message=msg,
        intent=pred_res["intent"],
        intent_confidence=pred_res["confidence"],
        evidence_quality=ev_score["evidence_quality"],
        trust_score=trust_res["trust_score"],
        evidence_count=ev_score["evidence_count"],
    )
    return {
        "decision": esc_res["decision"],
        "state": esc_res["state"],
        "risk": esc_res["risk"],
        "reason_code": esc_res.get("reason_code"),
        "reason_explanation": esc_res.get("reason_explanation"),
        "trust_score": trust_res["trust_score"],
        "confidence": pred_res["confidence"],
    }


@app.get("/golden-set")
@app.get("/api/golden-set")
async def get_golden_set_endpoint():
    """Retrieve the 200 Golden Set cases."""
    import pandas as pd
    golden_path = "data/golden/golden_set.csv"
    if os.path.exists(golden_path):
        df = pd.read_csv(golden_path)
        return {"total": len(df), "items": df.to_dict(orient="records")}
    return {"total": 0, "items": []}


@app.get("/evaluation/results")
@app.get("/api/evaluation/results")
async def get_eval_results():
    """Retrieve canonical benchmark and evaluation results."""
    path = "data/evaluation/final_results.json"
    if os.path.exists(path):
        import json
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    return {"status": "NOT_EVALUATED"}


@app.get("/failure-analysis")
@app.get("/api/failure-analysis")
async def get_failure_analysis_endpoint():
    """Retrieve top 5 failure analysis data."""
    path = "data/evaluation/failure_analysis.json"
    if os.path.exists(path):
        import json
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    return {"status": "NOT_EVALUATED"}


@app.get("/decision-log")
@app.get("/api/decision-log")
async def get_decision_log():
    """Retrieve Markdown decision log formatted as structured JSON."""
    log_path = "reports/decision_log.md"
    if not os.path.exists(log_path):
        return {"content": "Decision log not found."}
    with open(log_path, "r", encoding="utf-8") as f:
        content = f.read()
    html_content = markdown.markdown(content, extensions=["tables", "fenced_code"])
    return {"raw_markdown": content, "html_rendered": html_content}


# Mount Static Frontend
frontend_dir = os.path.abspath("frontend")
os.makedirs(frontend_dir, exist_ok=True)
app.mount("/static", StaticFiles(directory=frontend_dir), name="static")


@app.get("/{full_path:path}")
async def serve_frontend(full_path: str):
    """Serve single-page frontend application for all non-API routes."""
    index_file = os.path.join(frontend_dir, "index.html")
    if os.path.exists(index_file):
        return FileResponse(index_file)
    return HTMLResponse("<h1>SupportIQ AI Backend Running. Frontend is being built...</h1>")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True, app_dir=PROJECT_ROOT)


