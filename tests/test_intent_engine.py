"""
Unit tests for intent classification engine.
"""
import pytest
from src.intent_engine import IntentTaxonomy, MajorityClassifier, TfidfLogisticClassifier, SemanticClassifier


def test_intent_taxonomy():
    tax = IntentTaxonomy("config/brand_config.yaml")
    intents = tax.get_all_intent_ids()
    assert len(intents) >= 8
    assert "order_tracking" in intents
    assert "account_security" in intents
    assert tax.get_intent_risk("account_security") == "HIGH"


def test_majority_classifier():
    texts = ["order is late", "where is package", "refund money"]
    labels = ["order_tracking", "order_tracking", "refund_returns"]
    clf = MajorityClassifier().fit(texts, labels)
    pred = clf.predict_one("my item is lost")
    assert pred["intent"] == "order_tracking"
    assert pred["confidence"] == pytest.approx(2/3, 0.01)


def test_tfidf_classifier():
    texts = [
        "where is my package tracking delivery",
        "i need a refund for returned shoes",
        "my account was hacked unauthorized login"
    ]
    labels = ["order_tracking", "refund_returns", "account_security"]
    clf = TfidfLogisticClassifier().fit(texts, labels)
    pred = clf.predict_one("where is my delivery package?")
    assert pred["intent"] == "order_tracking"
    assert 0.0 <= pred["confidence"] <= 1.0


def test_semantic_classifier():
    texts = [
        "where is my package tracking delivery",
        "i need a refund for returned shoes",
        "my account was hacked unauthorized login"
    ]
    labels = ["order_tracking", "refund_returns", "account_security"]
    clf = SemanticClassifier("config/brand_config.yaml").fit(texts, labels)
    pred = clf.predict_one("where is my order package?")
    assert pred["intent"] == "order_tracking"
    assert pred["confidence"] >= 0.50
