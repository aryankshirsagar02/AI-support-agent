"""
Unit tests for retrieval engine and evidence scorer.
"""
import pytest
from src.retrieval_engine import HistoricalRetriever, EvidenceScorer


def test_retrieval_and_evidence_scoring():
    corpus = [
        {
            "conversation_id": "conv_1",
            "customer_message": "Where is my package? Tracking is delayed.",
            "cleaned_customer_message": "Where is my package? Tracking is delayed.",
            "brand_response": "We're sorry! Please send us a DM with your order number.",
            "cleaned_brand_response": "We're sorry! Please send us a DM with your order number.",
            "intent": "order_tracking",
        },
        {
            "conversation_id": "conv_2",
            "customer_message": "I returned my shoes 5 days ago. When do I get refunded?",
            "cleaned_customer_message": "I returned my shoes 5 days ago. When do I get refunded?",
            "brand_response": "Refunds take 3-5 business days to post to your card.",
            "cleaned_brand_response": "Refunds take 3-5 business days to post to your card.",
            "intent": "refund_returns",
        }
    ]

    retriever = HistoricalRetriever(top_k=2).build_index(corpus)
    matches = retriever.retrieve("Where is my delivery package?")
    assert len(matches) > 0
    assert matches[0]["conversation_id"] == "conv_1"

    scorer = EvidenceScorer()
    score = scorer.score_evidence(matches, predicted_intent="order_tracking")
    assert 0.0 <= score["evidence_quality"] <= 1.0
    assert score["evidence_count"] >= 1
