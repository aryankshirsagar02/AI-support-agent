"""
Intent taxonomy management and lookup for SupportIQ AI.
"""
from typing import List, Dict, Any, Optional
import yaml


class IntentTaxonomy:
    """Manages the brand-specific intent taxonomy."""

    def __init__(self, config_path: str = "config/brand_config.yaml"):
        self.config_path = config_path
        self.intents: List[Dict[str, Any]] = []
        self.intent_by_id: Dict[str, Dict[str, Any]] = {}
        self.load_taxonomy()

    def load_taxonomy(self):
        with open(self.config_path, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)
        self.intents = config.get("intent_taxonomy", [])
        self.intent_by_id = {i["id"]: i for i in self.intents}

    def get_all_intent_ids(self) -> List[str]:
        return [i["id"] for i in self.intents]

    def get_intent_name(self, intent_id: str) -> str:
        if intent_id in self.intent_by_id:
            return self.intent_by_id[intent_id].get("name", intent_id)
        return intent_id.replace("_", " ").title()

    def get_intent_risk(self, intent_id: str) -> str:
        if intent_id in self.intent_by_id:
            return self.intent_by_id[intent_id].get("risk_level", "MEDIUM")
        return "MEDIUM"

    def get_intent_typical_action(self, intent_id: str) -> str:
        if intent_id in self.intent_by_id:
            return self.intent_by_id[intent_id].get("typical_action", "REVIEW")
        return "REVIEW"

    def get_intent_keywords(self, intent_id: str) -> List[str]:
        if intent_id in self.intent_by_id:
            return self.intent_by_id[intent_id].get("keywords", [])
        return []

    def get_metadata(self, intent_id: str) -> Optional[Dict[str, Any]]:
        return self.intent_by_id.get(intent_id)
