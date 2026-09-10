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


class FailureTag(Base):
    """Allows support managers to tag and override failure classifications."""
    __tablename__ = "failure_tags"

    id = Column(String(64), primary_key=True)
    failure_category = Column(String(64))
    user_tag = Column(String(64))
    user_notes = Column(Text, nullable=True)
    updated_at = Column(DateTime, default=datetime.datetime.utcnow)


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
