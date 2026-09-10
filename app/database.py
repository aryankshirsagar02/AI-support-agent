"""
Database and logging layer for SupportIQ AI using SQLite and SQLAlchemy.
"""
import os
import datetime
from typing import List, Dict, Any, Optional
from sqlalchemy import create_engine, Column, Integer, String, Float, Text, DateTime, Boolean
from sqlalchemy.orm import declarative_base, sessionmaker

DB_PATH = os.path.abspath("data/supportiq.sqlite3")
os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
engine = create_engine(f"sqlite:///{DB_PATH}", connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()


class User(Base):
    """Support agents and managers."""
    __tablename__ = "users"
    id = Column(String(64), primary_key=True)
    name = Column(String(128), nullable=False)
    role = Column(String(32), default="agent")  # agent, supervisor, admin
    email = Column(String(128), unique=True)
    created_at = Column(DateTime, default=datetime.datetime.utcnow)


class Conversation(Base):
    """Historical and live conversation threads."""
    __tablename__ = "conversations"
    id = Column(String(64), primary_key=True)
    brand_id = Column(String(64), default="AmazonHelp")
    split = Column(String(16), default="train")  # train, val, test, live
    intent = Column(String(64), nullable=True)
    resolution_status = Column(String(32), default="RESOLVED")
    created_at = Column(DateTime, default=datetime.datetime.utcnow)


class Message(Base):
    """Inbound customer messages and brand replies."""
    __tablename__ = "messages"
    id = Column(String(64), primary_key=True)
    conversation_id = Column(String(64), index=True)
    author_type = Column(String(16))  # customer, brand, bot, human_agent
    text = Column(Text, nullable=False)
    cleaned_text = Column(Text, nullable=True)
    created_at = Column(DateTime, default=datetime.datetime.utcnow)


class IntentModel(Base):
    """Brand-derived intent taxonomy."""
    __tablename__ = "intents"
    id = Column(String(64), primary_key=True)
    name = Column(String(128), nullable=False)
    description = Column(Text)
    typical_action = Column(String(16), default="AUTO")
    risk_level = Column(String(16), default="LOW")
    keywords = Column(Text)


class EvidenceItem(Base):
    """Historical resolution evidence vectors and cases."""
    __tablename__ = "evidence"
    id = Column(String(64), primary_key=True)
    conversation_id = Column(String(64), index=True)
    customer_query = Column(Text, nullable=False)
    brand_resolution = Column(Text, nullable=False)
    intent = Column(String(64), index=True)
    similarity_score = Column(Float, default=0.0)


class PredictionLog(Base):
    """Logs incoming customer messages and pipeline predictions."""
    __tablename__ = "predictions"

    id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    request_id = Column(String(64), index=True)
    customer_message = Column(Text, nullable=False)
    predicted_intent = Column(String(64), index=True)
    intent_confidence = Column(Float, nullable=False)
    evidence_quality = Column(Float, nullable=False)
    trust_score = Column(Integer, nullable=False)
    decision = Column(String(32), index=True)
    decision_state = Column(String(16))
    risk_level = Column(String(16))
    escalation_reason_code = Column(String(64), nullable=True)
    escalation_explanation = Column(Text, nullable=True)
    generated_reply = Column(Text, nullable=False)
    grounding_notes = Column(Text, nullable=True)
    created_at = Column(DateTime, default=datetime.datetime.utcnow)


class ResponseRecord(Base):
    """Generated and approved customer support draft replies."""
    __tablename__ = "responses"
    id = Column(String(64), primary_key=True)
    prediction_id = Column(Integer, index=True)
    reply_text = Column(Text, nullable=False)
    grounding_score = Column(Integer, default=90)
    hallucination_risk = Column(String(16), default="LOW")
    status = Column(String(32), default="DRAFT")  # DRAFT, APPROVED, REJECTED, SENT
    created_at = Column(DateTime, default=datetime.datetime.utcnow)


class DecisionRecord(Base):
    """3-Tier state escalation routing decisions."""
    __tablename__ = "decisions"
    id = Column(String(64), primary_key=True)
    prediction_id = Column(Integer, index=True)
    decision_state = Column(String(16), nullable=False)  # GREEN, AMBER, RED
    decision_action = Column(String(32), nullable=False)  # AUTO_HANDLE, REVIEW_RECOMMENDED, HUMAN_ESCALATION
    reason_code = Column(String(64))
    confidence = Column(Float)
    created_at = Column(DateTime, default=datetime.datetime.utcnow)


class GoldenExample(Base):
    """200 Golden Evaluation Set reference cases."""
    __tablename__ = "golden_examples"
    id = Column(String(64), primary_key=True)
    conversation_id = Column(String(64))
    customer_message = Column(Text, nullable=False)
    expected_intent = Column(String(64), nullable=False)
    expected_action = Column(String(16), nullable=False)  # AUTO, HUMAN
    risk_level = Column(String(16), default="LOW")
    gold_reason = Column(Text, nullable=True)
    annotator_id = Column(String(32), default="EXP_ANN_01")
    difficulty = Column(String(16), default="medium")


class EvaluationRun(Base):
    """Execution logs of benchmark runs."""
    __tablename__ = "evaluations"
    id = Column(String(64), primary_key=True)
    timestamp = Column(DateTime, default=datetime.datetime.utcnow)
    system_name = Column(String(64), nullable=False)
    macro_f1 = Column(Float)
    accuracy = Column(Float)
    reply_quality = Column(Float)
    grounding_score = Column(Float)
    unsafe_auto_rate = Column(Float)
    safe_coverage = Column(Float)


class FailureRecord(Base):
    """Error diagnoses and supervisor tags."""
    __tablename__ = "failures"
    id = Column(String(64), primary_key=True)
    customer_message = Column(Text, nullable=False)
    failure_category = Column(String(64), index=True)
    likely_cause = Column(Text)
    proposed_fix = Column(Text)
    user_tag = Column(String(64), nullable=True)
    user_notes = Column(Text, nullable=True)
    updated_at = Column(DateTime, default=datetime.datetime.utcnow)


class FailureTag(Base):
    """Allows support managers to tag and override failure classifications."""
    __tablename__ = "failure_tags"

    id = Column(String(64), primary_key=True)
    failure_category = Column(String(64))
    user_tag = Column(String(64))
    user_notes = Column(Text, nullable=True)
    updated_at = Column(DateTime, default=datetime.datetime.utcnow)


class DecisionLogEntry(Base):
    """Documented architecture design choices."""
    __tablename__ = "decision_log"
    id = Column(String(64), primary_key=True)
    decision_number = Column(Integer)
    title = Column(String(256), nullable=False)
    decision_text = Column(Text, nullable=False)
    why_needed = Column(Text, nullable=False)
    options_considered = Column(Text)
    chosen_approach = Column(Text)
    reason = Column(Text)
    trade_off = Column(Text)



Base.metadata.create_all(bind=engine)


def log_prediction(
    customer_message: str,
    predicted_intent: str,
    intent_confidence: float,
    evidence_quality: float,
    trust_score: int,
    decision: str,
    decision_state: str,
    risk_level: str,
    generated_reply: str,
    escalation_reason_code: Optional[str] = None,
    escalation_explanation: Optional[str] = None,
    grounding_notes: Optional[str] = None,
) -> int:
    """Log a single prediction event to SQLite."""
    import uuid
    session = SessionLocal()
    try:
        log_entry = PredictionLog(
            request_id=f"req_{uuid.uuid4().hex[:8]}",
            customer_message=customer_message,
            predicted_intent=predicted_intent,
            intent_confidence=intent_confidence,
            evidence_quality=evidence_quality,
            trust_score=trust_score,
            decision=decision,
            decision_state=decision_state,
            risk_level=risk_level,
            escalation_reason_code=escalation_reason_code,
            escalation_explanation=escalation_explanation,
            generated_reply=generated_reply,
            grounding_notes=grounding_notes,
        )
        session.add(log_entry)
        session.commit()
        session.refresh(log_entry)
        return log_entry.id
    finally:
        session.close()


def get_recent_predictions(limit: int = 50) -> List[Dict[str, Any]]:
    """Retrieve recent predictions for dashboard/analytics."""
    session = SessionLocal()
    try:
        records = session.query(PredictionLog).order_by(PredictionLog.id.desc()).limit(limit).all()
        return [
            {
                "id": r.id,
                "request_id": r.request_id,
                "customer_message": r.customer_message,
                "predicted_intent": r.predicted_intent,
                "intent_confidence": r.intent_confidence,
                "evidence_quality": r.evidence_quality,
                "trust_score": r.trust_score,
                "decision": r.decision,
                "decision_state": r.decision_state,
                "risk_level": r.risk_level,
                "escalation_reason_code": r.escalation_reason_code,
                "escalation_explanation": r.escalation_explanation,
                "generated_reply": r.generated_reply,
                "created_at": r.created_at.isoformat() if r.created_at else None,
            }
            for r in records
        ]
    finally:
        session.close()
