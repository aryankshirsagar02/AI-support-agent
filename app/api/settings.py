"""
Settings & System Configuration API Router for SupportIQ AI.
Allows dynamic adjustment of Brand, Retrieval K, Trust Score weights, and Escalation thresholds.
"""
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Dict, Any, List, Optional
import yaml
import os

router = APIRouter(prefix="/api/settings", tags=["Settings"])


class SettingsUpdateRequest(BaseModel):
    brand_name: Optional[str] = None
    brand_handle: Optional[str] = None
    top_k: Optional[int] = None
    min_similarity_threshold: Optional[float] = None
    auto_handle_min_trust: Optional[float] = None
    review_recommended_min_trust: Optional[float] = None
    min_intent_confidence_for_auto: Optional[float] = None
    min_evidence_quality_for_auto: Optional[float] = None
    w_intent: Optional[float] = None
    w_evidence: Optional[float] = None
    w_consistency: Optional[float] = None
    w_safety: Optional[float] = None
    llm_provider: Optional[str] = None


@router.get("")
async def get_settings() -> Dict[str, Any]:
    """Retrieve current system configuration."""
    config_path = "config/brand_config.yaml"
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    return {
        "brand": config.get("brand", {}),
        "retrieval": config.get("retrieval", {}),
        "trust_engine": config.get("trust_engine", {}),
        "escalation_policy": config.get("escalation_policy", {}),
        "llm_provider": os.getenv("LLM_PROVIDER", "builtin"),
        "openai_model": os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
        "gemini_model": os.getenv("GEMINI_MODEL", "gemini-1.5-flash"),
    }


@router.post("")
async def update_settings(req: SettingsUpdateRequest) -> Dict[str, Any]:
    """Update system settings and save to brand config YAML."""
    config_path = "config/brand_config.yaml"
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    if req.brand_name:
        config["brand"]["name"] = req.brand_name
    if req.brand_handle:
        config["brand"]["twitter_handle"] = req.brand_handle
    if req.top_k is not None:
        config["retrieval"]["top_k"] = req.top_k
    if req.min_similarity_threshold is not None:
        config["retrieval"]["min_similarity_threshold"] = req.min_similarity_threshold

    # Thresholds
    t_cfg = config["trust_engine"].setdefault("thresholds", {})
    if req.auto_handle_min_trust is not None:
        t_cfg["auto_handle_min_trust"] = req.auto_handle_min_trust
    if req.review_recommended_min_trust is not None:
        t_cfg["review_recommended_min_trust"] = req.review_recommended_min_trust
    if req.min_intent_confidence_for_auto is not None:
        t_cfg["min_intent_confidence_for_auto"] = req.min_intent_confidence_for_auto
    if req.min_evidence_quality_for_auto is not None:
        t_cfg["min_evidence_quality_for_auto"] = req.min_evidence_quality_for_auto

    # Weights
    w_cfg = config["trust_engine"].setdefault("weights", {})
    if req.w_intent is not None:
        w_cfg["intent_confidence"] = req.w_intent
    if req.w_evidence is not None:
        w_cfg["evidence_quality"] = req.w_evidence
    if req.w_consistency is not None:
        w_cfg["historical_consistency"] = req.w_consistency
    if req.w_safety is not None:
        w_cfg["safety_score"] = req.w_safety

    with open(config_path, "w", encoding="utf-8") as f:
        yaml.dump(config, f, sort_keys=False)

    return {"status": "SUCCESS", "message": "Settings updated successfully", "config": config}


@router.post("/test-connection")
async def test_connection() -> Dict[str, Any]:
    """Test connection to retriever, classifier, and generator."""
    retriever_ok = os.path.exists("models/retriever.pkl")
    classifier_ok = os.path.exists("models/semantic_classifier.pkl")
    golden_ok = os.path.exists("data/golden/golden_set.csv")

    return {
        "status": "HEALTHY" if (retriever_ok and classifier_ok and golden_ok) else "DEGRADED",
        "retriever_index": "READY" if retriever_ok else "MISSING",
        "semantic_classifier": "READY" if classifier_ok else "MISSING",
        "golden_set": "READY" if golden_ok else "MISSING",
        "llm_provider": os.getenv("LLM_PROVIDER", "builtin"),
    }
