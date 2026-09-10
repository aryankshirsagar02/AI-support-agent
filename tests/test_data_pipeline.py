"""
Unit tests for data pipeline modules.
"""
import pytest
import os
import json
from src.data_pipeline.cleaner import clean_tweet_text, mask_sensitive_info
from src.data_pipeline.threading import ConversationThread
from src.data_pipeline.splitter import split_conversations_by_id


def test_clean_tweet_text():
    raw = "@AmazonHelp My order 112-9988771-1234567 is late! Check https://amazon.com/track"
    cleaned = clean_tweet_text(raw)
    assert "@AmazonHelp" not in cleaned
    assert "[URL]" in cleaned
    assert "[ORDER_ID]" in cleaned


def test_mask_sensitive_info():
    text = "My email is customer@gmail.com and order is 114-1234567-7654321"
    masked = mask_sensitive_info(text)
    assert "customer@gmail.com" not in masked
    assert "[EMAIL]" in masked
    assert "[ORDER_ID]" in masked


def test_leakage_free_splitting():
    threads = []
    for i in range(100):
        threads.append(
            ConversationThread(
                conversation_id=f"conv_{i}",
                brand_id="AmazonHelp",
                customer_message=f"Query {i}",
                cleaned_customer_message=f"Query {i}",
                brand_response=f"Reply {i}",
                cleaned_brand_response=f"Reply {i}",
                inbound_tweet_id=f"in_{i}",
                response_tweet_id=f"out_{i}",
                created_at="2025-10-01",
                intent="order_tracking" if i % 2 == 0 else "refund_returns",
            )
        )

    train_t, val_t, test_t = split_conversations_by_id(threads, train_ratio=0.70, val_ratio=0.15, test_ratio=0.15, random_seed=42)
    
    assert len(train_t) + len(val_t) + len(test_t) == 100
    assert len(train_t) >= 68
    assert len(val_t) >= 14
    assert len(test_t) >= 14

    train_ids = {t.conversation_id for t in train_t}
    val_ids = {t.conversation_id for t in val_t}
    test_ids = {t.conversation_id for t in test_t}

    assert train_ids.isdisjoint(val_ids)
    assert train_ids.isdisjoint(test_ids)
    assert val_ids.isdisjoint(test_ids)
