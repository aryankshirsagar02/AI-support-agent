"""
Conversation Threading Engine.
Reconstructs multi-turn conversational threads from raw Twitter Customer Support CSV format.
"""
from dataclasses import dataclass, field, asdict
from typing import List, Dict, Optional, Any
import pandas as pd
from .cleaner import clean_tweet_text


@dataclass
class Turn:
    tweet_id: str
    author_id: str
    text: str
    cleaned_text: str
    created_at: str
    inbound: bool


@dataclass
class ConversationThread:
    conversation_id: str
    brand_id: str
    customer_message: str
    cleaned_customer_message: str
    brand_response: str
    cleaned_brand_response: str
    inbound_tweet_id: str
    response_tweet_id: str
    created_at: str
    turn_count: int = 2
    intent: Optional[str] = None
    resolution_status: str = "RESOLVED"
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def _normalize_id(val: Any) -> str:
    if val is None or pd.isna(val):
        return ""
    val_str = str(val).strip()
    if val_str.endswith(".0"):
        val_str = val_str[:-2]
    return val_str


def reconstruct_conversations(
    df: pd.DataFrame,
    brand_id: str = "AmazonHelp",
    min_customer_len: int = 10,
    min_response_len: int = 15,
    taxonomy_keywords: Optional[Dict[str, List[str]]] = None,
) -> List[ConversationThread]:
    """
    Reconstruct paired customer -> brand conversations from a raw DataFrame.
    
    Expected DataFrame columns (Kaggle TWCS format):
    - tweet_id
    - author_id
    - inbound (True / False)
    - created_at
    - text
    - response_tweet_id
    - in_response_to_tweet_id
    """
    tweet_dict: Dict[str, Dict[str, Any]] = {}
    for _, row in df.iterrows():
        t_id = _normalize_id(row["tweet_id"])
        if not t_id:
            continue
        tweet_dict[t_id] = {
            "tweet_id": t_id,
            "author_id": str(row.get("author_id", "")),
            "inbound": bool(row.get("inbound", False)),
            "created_at": str(row.get("created_at", "")),
            "text": str(row.get("text", "")),
            "response_tweet_id": _normalize_id(row.get("response_tweet_id", "")),
            "in_response_to_tweet_id": _normalize_id(row.get("in_response_to_tweet_id", "")),
        }

    threads: List[ConversationThread] = []
    seen_convs = set()

    for t_id, t_data in tweet_dict.items():
        # Look for brand outbound responses
        if not t_data["inbound"] and t_data["author_id"].lower() == brand_id.lower():
            in_reply_to = t_data["in_response_to_tweet_id"]
            if in_reply_to and in_reply_to in tweet_dict:
                customer_tweet = tweet_dict[in_reply_to]
                if customer_tweet["inbound"]:
                    conv_id = f"conv_{customer_tweet['tweet_id']}_{t_id}"
                    if conv_id in seen_convs:
                        continue
                    seen_convs.add(conv_id)

                    cust_raw = customer_tweet["text"]
                    brand_raw = t_data["text"]
                    cust_clean = clean_tweet_text(cust_raw)
                    brand_clean = clean_tweet_text(brand_raw)

                    if len(cust_clean) >= min_customer_len and len(brand_clean) >= min_response_len:
                        inferred_intent = None
                        if taxonomy_keywords:
                            c_lower = cust_clean.lower()
                            best_score = 0
                            best_intent = "order_tracking"
                            for intent_name, kw_list in taxonomy_keywords.items():
                                sc = sum(1 for kw in kw_list if kw.lower() in c_lower)
                                if sc > best_score:
                                    best_score = sc
                                    best_intent = intent_name
                            inferred_intent = best_intent if best_score > 0 else "order_tracking"

                        threads.append(
                            ConversationThread(
                                conversation_id=conv_id,
                                brand_id=brand_id,
                                customer_message=cust_raw,
                                cleaned_customer_message=cust_clean,
                                brand_response=brand_raw,
                                cleaned_brand_response=brand_clean,
                                inbound_tweet_id=customer_tweet["tweet_id"],
                                response_tweet_id=t_id,
                                created_at=customer_tweet["created_at"],
                                turn_count=2,
                                intent=inferred_intent,
                                resolution_status="RESOLVED",
                            )
                        )

    return threads

