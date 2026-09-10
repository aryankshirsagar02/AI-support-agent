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
    
    assert train_ids.isdisjoint(val_ids), "Data Leakage Detected: Train and Validation share conversation IDs!"
    assert train_ids.isdisjoint(test_ids), "Data Leakage Detected: Train and Test share conversation IDs!"
    assert val_ids.isdisjoint(test_ids), "Data Leakage Detected: Validation and Test share conversation IDs!"
    
    return train_threads, val_threads, test_threads
