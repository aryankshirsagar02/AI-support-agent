"""
Failure Analysis Engine for SupportIQ AI.
Automatically identifies, categorizes, and diagnoses the top 5 customer support failure modes:
1. Ambiguous Intent
2. Retrieval Failure
3. Hallucinated Policy
4. Over-Escalation
5. Under-Escalation (Unsafe Automation)
"""
from typing import List, Dict, Any, Optional
import json


class FailureAnalyzer:
    """Detects and categorizes pipeline failures."""

    FAILURE_CATEGORIES = {
        "ambiguous_intent": {
            "title": "Ambiguous Intent",
            "description": "Short, vague, or multi-faceted customer query leading to low intent confidence.",
            "default_cause": "Query lacks distinguishing product or transactional keywords.",
            "default_fix": "Add multi-turn disambiguation prompt or clarify with quick-reply buttons.",
        },
        "retrieval_failure": {
            "title": "Retrieval Failure",
            "description": "Historical vector search returned low-similarity or discordant case resolutions.",
            "default_cause": "Corpus gap for niche inquiry or unusual phrasing.",
            "default_fix": "Expand training knowledge base or implement cross-encoder re-ranking.",
        },
        "hallucinated_policy": {
            "title": "Hallucinated Policy",
            "description": "Response asserted unsupported refund sums, deadlines, or unperformed actions.",
            "default_cause": "Weak prompt grounding constraint or loose template generation.",
            "default_fix": "Strengthen grounding validator and mask ungrounded entity tokens.",
        },
        "over_escalation": {
            "title": "Over-Escalation (Excessive Human Routing)",
            "description": "Routine, low-risk query was unnecessarily escalated to human queue.",
            "default_cause": "Conservative trust score weights or stringent similarity threshold.",
            "default_fix": "Calibrate intent confidence thresholds for high-volume routine intents.",
        },
        "under_escalation": {
            "title": "Under-Escalation (Unsafe Automation)",
            "description": "High-risk, security, or legal complaint was mistakenly auto-handled.",
            "default_cause": "Security or risk keywords missed in lexical escalation scanner.",
            "default_fix": "Add zero-shot risk classifier and expand mandatory escalation keyword list.",
        }
    }

    def analyze_predictions(
        self,
        evaluation_records: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Analyze evaluation records and categorize failures.
        """
        failures: List[Dict[str, Any]] = []
        category_counts = {k: 0 for k in self.FAILURE_CATEGORIES.keys()}

        for rec in evaluation_records:
            pred_intent = rec.get("predicted_intent")
            gold_intent = rec.get("gold_intent")
            decision = rec.get("decision")  # AUTO_HANDLE vs HUMAN_ESCALATION
            gold_action = rec.get("gold_action")  # AUTO vs HUMAN
            conf = rec.get("intent_confidence", 1.0)
            evidence_q = rec.get("evidence_quality", 1.0)
            msg = rec.get("customer_message", "")
            reason_code = rec.get("reason_code")

            cat: Optional[str] = None
            cause: Optional[str] = None
            fix: Optional[str] = None

            # 1. Under-escalation (CRITICAL: gold was HUMAN, but system did AUTO_HANDLE)
            if gold_action == "HUMAN" and decision == "AUTO_HANDLE":
                cat = "under_escalation"
                cause = f"High-risk query (Gold: {gold_intent}) bypassed escalation filter."
                fix = "Add mandatory escalation rule for this intent class and enforce security keyword scan."

            # 2. Over-escalation (Gold was AUTO, but system escalated to HUMAN)
            elif gold_action == "AUTO" and decision == "HUMAN_ESCALATION":
                cat = "over_escalation"
                cause = f"System escalated with reason '{reason_code}' despite query being a routine inquiry."
                fix = "Calibrate evidence quality threshold for standard inquiries."

            # 3. Ambiguous intent (Misclassified intent or very low confidence)
            elif pred_intent != gold_intent:
                if conf < 0.70 or len(msg.split()) <= 3:
                    cat = "ambiguous_intent"
                    cause = f"Vague or multi-intent phrasing ('{msg}') led to prediction '{pred_intent}' instead of '{gold_intent}'."
                    fix = "Implement clarification follow-up question before routing."
                elif evidence_q < 0.60:
                    cat = "retrieval_failure"
                    cause = f"Vector search returned low-similarity matches ({evidence_q}) for query."
                    fix = "Expand exemplar seed set for '{gold_intent}' and apply dense re-ranking."

            # 4. Check for hallucination
            elif rec.get("hallucination_score", 1.0) > 3.0:
                cat = "hallucinated_policy"
                cause = "Generated reply contained specific ungrounded promises or entity claims."
                fix = "Enforce stricter anti-hallucination guardrail rules in prompt generator."

            if cat:
                category_counts[cat] += 1
                failures.append({
                    "id": rec.get("id", f"fail_{len(failures)+1}"),
                    "customer_message": msg,
                    "predicted_intent": pred_intent,
                    "gold_intent": gold_intent,
                    "decision": decision,
                    "gold_action": gold_action,
                    "intent_confidence": conf,
                    "evidence_quality": evidence_q,
                    "retrieved_evidence": rec.get("retrieved_evidence", [])[:2],
                    "generated_reply": rec.get("generated_reply", ""),
                    "gold_reason": rec.get("gold_reason"),
                    "failure_category": cat,
                    "category_title": self.FAILURE_CATEGORIES[cat]["title"],
                    "likely_cause": cause or self.FAILURE_CATEGORIES[cat]["default_cause"],
                    "proposed_fix": fix or self.FAILURE_CATEGORIES[cat]["default_fix"],
                    "user_tag": None,
                })

        return {
            "total_evaluated": len(evaluation_records),
            "total_failures": len(failures),
            "failure_rate": round(len(failures) / max(len(evaluation_records), 1), 4),
            "category_counts": category_counts,
            "failure_cases": failures,
        }
