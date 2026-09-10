"""
Escalation Engine for SupportIQ AI.
Evaluates intent risk, evidence sufficiency, security triggers, and trust thresholds
to make high-stakes routing decisions with standardized machine- and human-readable reason codes.
"""
from typing import Dict, Any, List, Optional
import yaml


# Standardized Reason Codes
REASON_CODES = {
    "LOW_INTENT_CONFIDENCE": "Intent classification confidence is below the minimum automated threshold.",
    "INSUFFICIENT_EVIDENCE": "Historical case retrieval found insufficient relevant or concordant examples.",
    "SENSITIVE_ACCOUNT_ISSUE": "Inquiry involves sensitive account recovery, billing disputes, or private data.",
    "SECURITY_OR_FRAUD": "Potential unauthorized access, account compromise, or security alert detected.",
    "HIGH_RISK_REQUEST": "Request contains high-risk triggers (e.g., legal threats, chargeback, police).",
    "CONFLICTING_HISTORICAL_CASES": "Retrieved historical cases provide contradictory resolution steps.",
    "OUT_OF_SCOPE": "Customer query is outside supported customer service taxonomy.",
    "CUSTOMER_REQUESTED_HUMAN": "Customer explicitly requested a live human representative or supervisor.",
    "UNSUPPORTED_RESPONSE": "Generated response could not be verified against grounded historical evidence.",
}


class EscalationEngine:
    """Makes safe automated handling vs human escalation decisions."""

    def __init__(self, config_path: str = "config/brand_config.yaml"):
        self.config_path = config_path
        self.auto_handle_min_trust = 78.0
        self.review_recommended_min_trust = 55.0
        self.min_intent_confidence = 0.75
        self.min_evidence_quality = 0.65
        self.auto_escalate_intents: List[str] = ["account_security", "human_escalation_request"]
        self.sensitive_keywords: List[str] = [
            "lawyer", "attorney", "legal action", "sue", "police", "fraud",
            "hacked", "stolen card", "chargeback", "press", "journalist", "court"
        ]
        self.load_config()

    def load_config(self):
        try:
            with open(self.config_path, "r", encoding="utf-8") as f:
                config = yaml.safe_load(f)
            t_cfg = config.get("trust_engine", {}).get("thresholds", {})
            self.auto_handle_min_trust = t_cfg.get("auto_handle_min_trust", self.auto_handle_min_trust)
            self.review_recommended_min_trust = t_cfg.get("review_recommended_min_trust", self.review_recommended_min_trust)
            self.min_intent_confidence = t_cfg.get("min_intent_confidence_for_auto", self.min_intent_confidence)
            self.min_evidence_quality = t_cfg.get("min_evidence_quality_for_auto", self.min_evidence_quality)

            esc_cfg = config.get("escalation_policy", {})
            self.auto_escalate_intents = esc_cfg.get("auto_escalate_intents", self.auto_escalate_intents)
            self.sensitive_keywords = esc_cfg.get("sensitive_keywords", self.sensitive_keywords)
        except Exception:
            pass

    def evaluate(
        self,
        customer_message: str,
        intent: str,
        intent_confidence: float,
        evidence_quality: float,
        trust_score: float,
        evidence_count: int,
        intent_risk_level: str = "LOW",
    ) -> Dict[str, Any]:
        """
        Evaluate customer message and pipeline outputs to determine escalation status.
        
        Returns:
            Dict containing decision (AUTO_HANDLE, REVIEW_RECOMMENDED, HUMAN_ESCALATION),
            risk (LOW, MEDIUM, HIGH, CRITICAL), reason_code, reason_explanation, and color.
        """
        msg_lower = customer_message.lower()

        # 1. Check for explicit human agent request
        if intent == "human_escalation_request" or any(p in msg_lower for p in ["speak to human", "real person", "human agent", "talk to a person", "agent now"]):
            return {
                "decision": "HUMAN_ESCALATION",
                "state": "RED",
                "risk": "HIGH",
                "reason_code": "CUSTOMER_REQUESTED_HUMAN",
                "reason_explanation": REASON_CODES["CUSTOMER_REQUESTED_HUMAN"],
                "requires_human_approval": True,
                "can_auto_reply": False,
            }

        # 2. Check for security or unauthorized account compromise
        if intent == "account_security" or any(p in msg_lower for p in ["hacked", "unauthorized access", "compromised", "stolen account", "suspicious login"]):
            return {
                "decision": "HUMAN_ESCALATION",
                "state": "RED",
                "risk": "HIGH",
                "reason_code": "SECURITY_OR_FRAUD",
                "reason_explanation": "Critical security alert detected requiring immediate verification by an Account Security Specialist.",
                "requires_human_approval": True,
                "can_auto_reply": False,
            }

        # 3. Check for high-risk sensitive legal / regulatory triggers
        for kw in self.sensitive_keywords:
            if kw in msg_lower:
                return {
                    "decision": "HUMAN_ESCALATION",
                    "state": "RED",
                    "risk": "HIGH",
                    "reason_code": "HIGH_RISK_REQUEST",
                    "reason_explanation": f"Message contains high-risk escalation keyword '{kw}'. Routed to Specialized Operations.",
                    "requires_human_approval": True,
                    "can_auto_reply": False,
                }

        # 4. Check for low intent confidence
        if intent_confidence < self.min_intent_confidence:
            return {
                "decision": "REVIEW_RECOMMENDED" if trust_score >= self.review_recommended_min_trust else "HUMAN_ESCALATION",
                "state": "AMBER" if trust_score >= self.review_recommended_min_trust else "RED",
                "risk": "MEDIUM",
                "reason_code": "LOW_INTENT_CONFIDENCE",
                "reason_explanation": f"Intent confidence ({round(intent_confidence * 100, 1)}%) is below the {round(self.min_intent_confidence * 100, 1)}% automation bar.",
                "requires_human_approval": True,
                "can_auto_reply": False,
            }

        # 5. Check for insufficient historical evidence
        if evidence_quality < self.min_evidence_quality or evidence_count < 1:
            return {
                "decision": "REVIEW_RECOMMENDED" if trust_score >= self.review_recommended_min_trust else "HUMAN_ESCALATION",
                "state": "AMBER" if trust_score >= self.review_recommended_min_trust else "RED",
                "risk": "MEDIUM",
                "reason_code": "INSUFFICIENT_EVIDENCE",
                "reason_explanation": f"Historical evidence quality ({round(evidence_quality * 100, 1)}%) is insufficient for automated reply.",
                "requires_human_approval": True,
                "can_auto_reply": False,
            }

        # 6. Evaluate composite Trust Score
        if trust_score >= self.auto_handle_min_trust and intent_risk_level in ["LOW", "MEDIUM"]:
            return {
                "decision": "AUTO_HANDLE",
                "state": "GREEN",
                "risk": "LOW",
                "reason_code": None,
                "reason_explanation": "High intent confidence and strong grounded historical evidence. Safe for automated dispatch.",
                "requires_human_approval": False,
                "can_auto_reply": True,
            }
        elif trust_score >= self.review_recommended_min_trust:
            return {
                "decision": "REVIEW_RECOMMENDED",
                "state": "AMBER",
                "risk": "MEDIUM",
                "reason_code": "SENSITIVE_ACCOUNT_ISSUE" if intent_risk_level == "MEDIUM" else "LOW_INTENT_CONFIDENCE",
                "reason_explanation": "Moderate trust score. Pre-drafted reply generated for human agent review before sending.",
                "requires_human_approval": True,
                "can_auto_reply": False,
            }
        else:
            return {
                "decision": "HUMAN_ESCALATION",
                "state": "RED",
                "risk": "HIGH",
                "reason_code": "INSUFFICIENT_EVIDENCE",
                "reason_explanation": f"Composite trust score ({round(trust_score, 1)}) is below threshold ({self.review_recommended_min_trust}).",
                "requires_human_approval": True,
                "can_auto_reply": False,
            }
