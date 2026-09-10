"""
Unit tests for Trust Scorer and Escalation Engine.
"""
import pytest
from src.trust_engine import TrustScorer, EscalationEngine


def test_trust_scorer():
    scorer = TrustScorer("config/brand_config.yaml")
    res = scorer.compute_trust_score(
        intent_confidence=0.90,
        evidence_quality=0.85,
        historical_consistency=0.90,
        safety_score=1.0,
    )
    assert 0 <= res["trust_score"] <= 100
    assert res["trust_level"] == "HIGH"


def test_escalation_security_trigger():
    engine = EscalationEngine("config/brand_config.yaml")
    res = engine.evaluate(
        customer_message="Someone hacked into my account and changed password",
        intent="account_security",
        intent_confidence=0.95,
        evidence_quality=0.80,
        trust_score=85,
        evidence_count=3,
        intent_risk_level="HIGH",
    )
    assert res["decision"] == "HUMAN_ESCALATION"
    assert res["state"] == "RED"
    assert res["reason_code"] == "SECURITY_OR_FRAUD"
    assert res["can_auto_reply"] is False


def test_escalation_human_request():
    engine = EscalationEngine("config/brand_config.yaml")
    res = engine.evaluate(
        customer_message="I demand to speak to a real human person right now!",
        intent="human_escalation_request",
        intent_confidence=0.98,
        evidence_quality=0.80,
        trust_score=85,
        evidence_count=2,
        intent_risk_level="HIGH",
    )
    assert res["decision"] == "HUMAN_ESCALATION"
    assert res["reason_code"] == "CUSTOMER_REQUESTED_HUMAN"
    assert res["can_auto_reply"] is False


def test_escalation_auto_handle():
    engine = EscalationEngine("config/brand_config.yaml")
    res = engine.evaluate(
        customer_message="Where is my order #112-9988221?",
        intent="order_tracking",
        intent_confidence=0.92,
        evidence_quality=0.85,
        trust_score=88,
        evidence_count=3,
        intent_risk_level="LOW",
    )
    assert res["decision"] == "AUTO_HANDLE"
    assert res["state"] == "GREEN"
    assert res["can_auto_reply"] is True
