"""
Live Support Workspace API endpoint.
Handles real-time customer message analysis, retrieval, generation, trust scoring, and escalation.
"""
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional

from src.intent_engine import ClassifierFactory, IntentTaxonomy
from src.retrieval_engine import HistoricalRetriever, EvidenceScorer
from src.response_engine import ResponseGenerator
from src.trust_engine import TrustScorer, EscalationEngine
from app.database import log_prediction

router = APIRouter(prefix="/api/support", tags=["Support Workspace"])

# Lazy-loaded pipeline singleton
_pipeline = None


def get_pipeline():
    global _pipeline
    if _pipeline is None:
        config_path = "config/brand_config.yaml"
        retriever_path = "models/retriever.pkl"
        classifier_path = "models/semantic_classifier.pkl"

        taxonomy = IntentTaxonomy(config_path)
        retriever = HistoricalRetriever.load(retriever_path) if HistoricalRetriever else None
        classifier = ClassifierFactory.load(classifier_path, "semantic")
        evidence_scorer = EvidenceScorer()
        trust_scorer = TrustScorer(config_path)
        escalation_engine = EscalationEngine(config_path)
        response_gen = ResponseGenerator(brand_name="Amazon Help")

        _pipeline = {
            "taxonomy": taxonomy,
            "retriever": retriever,
            "classifier": classifier,
            "evidence_scorer": evidence_scorer,
            "trust_scorer": trust_scorer,
            "escalation_engine": escalation_engine,
            "response_gen": response_gen,
        }
    return _pipeline


class AnalyzeRequest(BaseModel):
    message: str = Field(..., description="Customer support message text", min_length=1)
    force_escalate: Optional[bool] = False


class AnalyzeResponse(BaseModel):
    customer_message: str
    intent: str
    intent_name: str
    intent_confidence: float
    intent_risk: str
    probabilities: Dict[str, float]
    evidence_quality: float
    evidence_count: int
    evidence_cases: List[Dict[str, Any]]
    trust_score: int
    trust_level: str
    trust_breakdown: Dict[str, float]
    decision: str
    decision_state: str
    risk: str
    escalation_reason_code: Optional[str]
    escalation_explanation: Optional[str]
    reply: str
    grounding_notes: str
    can_auto_reply: bool


@router.post("/analyze", response_model=AnalyzeResponse)
async def analyze_message(req: AnalyzeRequest):
    """
    Analyze customer message in real-time.
    Executes intent classification -> retrieval -> evidence scoring -> trust calculation -> escalation -> response generation.
    """
    try:
        pipeline = get_pipeline()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Pipeline initialization failed: {str(e)}")

    msg = req.message.strip()
    if not msg:
        raise HTTPException(status_code=400, detail="Customer message cannot be empty.")

    # 1. Intent Detection
    pred_res = pipeline["classifier"].predict_one(msg)
    intent = pred_res["intent"]
    conf = pred_res["confidence"]
    probs = pred_res.get("probabilities", {})
    intent_name = pipeline["taxonomy"].get_intent_name(intent)
    intent_risk = pipeline["taxonomy"].get_intent_risk(intent)

    # 2. Historical Case Retrieval
    retrieved = []
    if pipeline["retriever"]:
        retrieved = pipeline["retriever"].retrieve(msg, top_k=5)

    # 3. Evidence Quality Scoring
    ev_score = pipeline["evidence_scorer"].score_evidence(retrieved, predicted_intent=intent)
    ev_quality = ev_score["evidence_quality"]
    ev_count = ev_score["evidence_count"]

    # 4. Trust Score Calculation
    trust_res = pipeline["trust_scorer"].compute_trust_score(
        intent_confidence=conf,
        evidence_quality=ev_quality,
        historical_consistency=0.92,
        safety_score=1.0 if intent_risk != "HIGH" else 0.4,
    )
    trust_score = trust_res["trust_score"]

    # 5. Escalation Decision
    if req.force_escalate:
        esc_res = {
            "decision": "HUMAN_ESCALATION",
            "state": "RED",
            "risk": "HIGH",
            "reason_code": "CUSTOMER_REQUESTED_HUMAN",
            "reason_explanation": "Manually escalated by support agent.",
            "requires_human_approval": True,
            "can_auto_reply": False,
        }
    else:
        esc_res = pipeline["escalation_engine"].evaluate(
            customer_message=msg,
            intent=intent,
            intent_confidence=conf,
            evidence_quality=ev_quality,
            trust_score=trust_score,
            evidence_count=ev_count,
            intent_risk_level=intent_risk,
        )

    # 6. Response Generation
    gen_res = pipeline["response_gen"].generate_response(
        customer_message=msg,
        predicted_intent=intent,
        evidence_cases=retrieved,
        evidence_quality=ev_quality,
    )

    # 7. Persistent Logging
    try:
        log_prediction(
            customer_message=msg,
            predicted_intent=intent,
            intent_confidence=conf,
            evidence_quality=ev_quality,
            trust_score=trust_score,
            decision=esc_res["decision"],
            decision_state=esc_res["state"],
            risk_level=esc_res["risk"],
            generated_reply=gen_res["reply"],
            escalation_reason_code=esc_res.get("reason_code"),
            escalation_explanation=esc_res.get("reason_explanation"),
            grounding_notes=gen_res.get("grounding_notes"),
        )
    except Exception:
        pass

    return AnalyzeResponse(
        customer_message=msg,
        intent=intent,
        intent_name=intent_name,
        intent_confidence=conf,
        intent_risk=intent_risk,
        probabilities=probs,
        evidence_quality=ev_quality,
        evidence_count=ev_count,
        evidence_cases=retrieved,
        trust_score=trust_score,
        trust_level=trust_res["trust_level"],
        trust_breakdown=trust_res["breakdown"],
        decision=esc_res["decision"],
        decision_state=esc_res["state"],
        risk=esc_res["risk"],
        escalation_reason_code=esc_res.get("reason_code"),
        escalation_explanation=esc_res.get("reason_explanation"),
        reply=gen_res["reply"],
        grounding_notes=gen_res.get("grounding_notes", ""),
        can_auto_reply=esc_res.get("can_auto_reply", False),
    )
