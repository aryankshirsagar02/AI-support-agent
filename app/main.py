"""
SupportIQ AI - Main FastAPI Application.
Serves REST API endpoints and mounts the enterprise web application frontend.
"""
import os
import sys
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


@app.get("/api/health")
async def health_check():
    return {
        "status": "UP",
        "app": "SupportIQ AI",
        "brand": "Amazon Help (@AmazonHelp)",
        "version": "1.0.0",
    }


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
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)
