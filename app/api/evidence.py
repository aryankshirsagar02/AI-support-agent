"""
Evidence Explorer API Router for SupportIQ AI.
Enables full-text search, intent filtering, and resolution thread inspection over historical training corpus.
"""
from fastapi import APIRouter, Query
from typing import List, Dict, Any, Optional
import os
import json
from src.retrieval_engine import HistoricalRetriever

router = APIRouter(prefix="/api/evidence", tags=["Evidence Explorer"])

_cached_conversations = None


def get_all_training_conversations() -> List[Dict[str, Any]]:
    global _cached_conversations
    if _cached_conversations is None:
        train_path = "data/processed/train.jsonl"
        if os.path.exists(train_path):
            with open(train_path, "r", encoding="utf-8") as f:
                _cached_conversations = [json.loads(line) for line in f]
        else:
            _cached_conversations = []
    return _cached_conversations


@router.get("")
async def search_evidence(
    query: Optional[str] = Query(None, description="Search term in customer message or response"),
    intent: Optional[str] = Query(None, description="Filter by intent ID"),
    limit: int = Query(25, ge=1, le=100),
    offset: int = Query(0, ge=0),
) -> Dict[str, Any]:
    """
    Search and filter historical training evidence cases.
    """
    all_convs = get_all_training_conversations()
    filtered = all_convs

    # Filter by intent
    if intent and intent != "all":
        filtered = [c for c in filtered if c.get("intent") == intent]

    # Filter by query text if supplied
    if query and query.strip():
        q_lower = query.lower().strip()
        # Semantic search if vector retriever available, else substring matching
        try:
            retriever = HistoricalRetriever.load("models/retriever.pkl")
            ranked = retriever.retrieve(query, top_k=50, filter_intent=intent if intent != "all" else None)
            return {
                "total_count": len(ranked),
                "items": ranked[offset : offset + limit],
                "query": query,
                "intent_filter": intent,
            }
        except Exception:
            filtered = [
                c for c in filtered
                if q_lower in c.get("customer_message", "").lower() or q_lower in c.get("brand_response", "").lower()
            ]

    total_count = len(filtered)
    paginated = filtered[offset : offset + limit]

    return {
        "total_count": total_count,
        "items": paginated,
        "query": query,
        "intent_filter": intent,
        "offset": offset,
        "limit": limit,
    }


@router.get("/{conversation_id}")
async def get_conversation_details(conversation_id: str) -> Dict[str, Any]:
    """Get single conversation thread details."""
    all_convs = get_all_training_conversations()
    for c in all_convs:
        if c.get("conversation_id") == conversation_id:
            return {"conversation": c}
    return {"error": "Conversation not found", "conversation_id": conversation_id}
