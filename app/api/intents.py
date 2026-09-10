"""
Intent Intelligence API Router for SupportIQ AI.
Exposes data-derived intent taxonomy, volume percentages, keyword signatures, and exemplar queries.
"""
from fastapi import APIRouter
import yaml
import json
import os
from typing import Dict, Any, List
from collections import Counter

router = APIRouter(prefix="/api/intents", tags=["Intent Intelligence"])


@router.get("")
async def get_intent_taxonomy() -> Dict[str, Any]:
    """Retrieve full intent taxonomy with volume stats and sample queries."""
    config_path = "config/brand_config.yaml"
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    intents = config.get("intent_taxonomy", [])
    
    # Calculate dataset intent frequencies
    train_path = "data/processed/train.jsonl"
    intent_counts = Counter()
    total_samples = 0
    intent_examples: Dict[str, List[str]] = {i["id"]: [] for i in intents}

    if os.path.exists(train_path):
        with open(train_path, "r", encoding="utf-8") as f:
            for line in f:
                item = json.loads(line)
                i_id = item.get("intent", "other")
                intent_counts[i_id] += 1
                total_samples += 1
                if len(intent_examples.get(i_id, [])) < 3:
                    if i_id not in intent_examples:
                        intent_examples[i_id] = []
                    intent_examples[i_id].append(item.get("customer_message", ""))

    enriched_intents = []
    for item in intents:
        i_id = item["id"]
        count = intent_counts.get(i_id, 0)
        pct = round((count / max(total_samples, 1)) * 100, 1)
        enriched_intents.append({
            "id": i_id,
            "name": item.get("name"),
            "description": item.get("description"),
            "typical_action": item.get("typical_action", "AUTO"),
            "risk_level": item.get("risk_level", "LOW"),
            "keywords": item.get("keywords", []),
            "volume_count": count,
            "volume_percentage": pct,
            "sample_queries": intent_examples.get(i_id, []),
        })

    return {
        "brand_id": config.get("brand", {}).get("id", "AmazonHelp"),
        "total_intents": len(enriched_intents),
        "total_training_samples": total_samples,
        "taxonomy": enriched_intents,
    }
