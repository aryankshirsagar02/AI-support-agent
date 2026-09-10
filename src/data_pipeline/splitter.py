"""
Conversation-Level Train / Validation / Test Splitting.
Strictly ensures no conversation leakage across splits.
"""
from typing import List, Tuple, Dict
import random
from .threading import ConversationThread


def split_conversations_by_id(
    threads: List[ConversationThread],
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    random_seed: int = 42,
) -> Tuple[List[ConversationThread], List[ConversationThread], List[ConversationThread]]:
    """
    Split conversations strictly by conversation_id to prevent data leakage.
    
    Returns:
        (train_threads, val_threads, test_threads)
    """
    assert abs((train_ratio + val_ratio + test_ratio) - 1.0) < 1e-5, "Ratios must sum to 1.0"
    
    # Group by intent if available for stratified splitting across conversation IDs
    intent_groups: Dict[str, List[ConversationThread]] = {}
    for t in threads:
        intent = t.intent or "unknown"
        if intent not in intent_groups:
            intent_groups[intent] = []
        intent_groups[intent].append(t)
    
    rng = random.Random(random_seed)
    
    train_threads: List[ConversationThread] = []
    val_threads: List[ConversationThread] = []
    test_threads: List[ConversationThread] = []
    
    for intent, group in sorted(intent_groups.items()):
        # Shuffle group with seed
        shuffled = list(group)
        rng.shuffle(shuffled)
        
        n = len(shuffled)
        n_train = int(n * train_ratio)
        n_val = int(n * val_ratio)
        
        train_threads.extend(shuffled[:n_train])
        val_threads.extend(shuffled[n_train:n_train + n_val])
        test_threads.extend(shuffled[n_train + n_val:])
        
    # Verify zero overlap of conversation IDs
    train_ids = {t.conversation_id for t in train_threads}
    val_ids = {t.conversation_id for t in val_threads}
    test_ids = {t.conversation_id for t in test_threads}
    
    tv_overlap = train_ids.intersection(val_ids)
    tt_overlap = train_ids.intersection(test_ids)
    vt_overlap = val_ids.intersection(test_ids)

    if tv_overlap or tt_overlap or vt_overlap:
        all_offenders = list(tv_overlap | tt_overlap | vt_overlap)
        raise ValueError(
            f"DATA LEAKAGE DETECTED! Overlapping conversation IDs found across splits: {all_offenders[:10]}"
        )
    
    return train_threads, val_threads, test_threads


def verify_leakage_from_files(
    train_path: str = "data/processed/train.jsonl",
    val_path: str = "data/processed/val.jsonl",
    test_path: str = "data/processed/test.jsonl",
) -> Dict[str, Any]:
    """
    Automated check verifying 100% disjoint conversation splits.
    Returns status dict with PASS/FAIL and detailed counts.
    """
    import json
    import os

    for p in [train_path, val_path, test_path]:
        if not os.path.exists(p):
            return {
                "status": "FAIL",
                "leakage_detected": True,
                "error": f"Split file not found: {p}",
                "train_count": 0,
                "val_count": 0,
                "test_count": 0,
                "offending_ids": [],
            }

    train_ids = {json.loads(line)["conversation_id"] for line in open(train_path, "r", encoding="utf-8")}
    val_ids = {json.loads(line)["conversation_id"] for line in open(val_path, "r", encoding="utf-8")}
    test_ids = {json.loads(line)["conversation_id"] for line in open(test_path, "r", encoding="utf-8")}

    tv_overlap = train_ids.intersection(val_ids)
    tt_overlap = train_ids.intersection(test_ids)
    vt_overlap = val_ids.intersection(test_ids)
    offenders = list(tv_overlap | tt_overlap | vt_overlap)

    has_leakage = len(offenders) > 0

    return {
        "status": "FAIL" if has_leakage else "PASS",
        "leakage_detected": has_leakage,
        "train_count": len(train_ids),
        "val_count": len(val_ids),
        "test_count": len(test_ids),
        "total_unique_conversations": len(train_ids | val_ids | test_ids),
        "train_val_overlap": len(tv_overlap),
        "train_test_overlap": len(tt_overlap),
        "val_test_overlap": len(vt_overlap),
        "offending_ids": offenders,
    }

