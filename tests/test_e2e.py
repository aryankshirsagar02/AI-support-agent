"""
End-to-End Integration tests for SupportIQ AI API and Pipeline.
"""
import pytest
from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)


def test_health_endpoint():
    response = client.get("/api/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "UP"
    assert "SupportIQ AI" in data["app"]


def test_dashboard_endpoint():
    response = client.get("/api/dashboard")
    assert response.status_code == 200
    data = response.json()
    assert "kpis" in data
    assert data["brand_name"] == "Amazon Help"


def test_intents_endpoint():
    response = client.get("/api/intents")
    assert response.status_code == 200
    data = response.json()
    assert data["total_intents"] >= 8
    assert "taxonomy" in data


def test_evidence_endpoint():
    response = client.get("/api/evidence?limit=10")
    assert response.status_code == 200
    data = response.json()
    assert "items" in data
    assert len(data["items"]) <= 10


def test_analyze_support_message_auto_handle():
    payload = {"message": "Where is my package? Order 112-9988771."}
    response = client.post("/api/support/analyze", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["intent"] == "order_tracking"
    assert data["trust_score"] > 0
    assert "reply" in data
    assert len(data["reply"]) > 10


def test_analyze_support_message_escalate():
    payload = {"message": "Someone hacked my account and changed my password!"}
    response = client.post("/api/support/analyze", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["decision"] == "HUMAN_ESCALATION"
    assert data["decision_state"] == "RED"
    assert data["escalation_reason_code"] == "SECURITY_OR_FRAUD"
    assert data["can_auto_reply"] is False


def test_settings_endpoints():
    res_get = client.get("/api/settings")
    assert res_get.status_code == 200
    
    res_test = client.post("/api/settings/test-connection")
    assert res_test.status_code == 200
    data = res_test.json()
    assert data["status"] in ["HEALTHY", "DEGRADED"]


def test_predict_endpoint():
    payload = {"message": "My package was marked delivered but I cannot find it."}
    res = client.post("/predict", json=payload)
    assert res.status_code == 200
    data = res.json()
    assert "intent" in data
    assert "trust_score" in data
    assert "decision" in data
    assert "reply" in data


def test_support_reply_and_decision_endpoints():
    reply_res = client.post("/support/reply", json={"message": "Can I cancel my order?", "intent": "cancellation"})
    assert reply_res.status_code == 200
    assert "reply" in reply_res.json()

    dec_res = client.post("/support/decision", json={"message": "I will sue you in court if you don't call me."})
    assert dec_res.status_code == 200
    dec_data = dec_res.json()
    assert dec_data["decision"] == "HUMAN_ESCALATION"
    assert dec_data["state"] == "RED"


def test_golden_set_and_evaluation_endpoints():
    res_golden = client.get("/golden-set")
    assert res_golden.status_code == 200
    assert res_golden.json()["total"] == 200

    res_eval = client.get("/evaluation/results")
    assert res_eval.status_code == 200
    assert "supportiq" in res_eval.json()

    res_fail = client.get("/failure-analysis")
    assert res_fail.status_code == 200

    res_log = client.get("/decision-log")
    assert res_log.status_code == 200
    assert "raw_markdown" in res_log.json()

