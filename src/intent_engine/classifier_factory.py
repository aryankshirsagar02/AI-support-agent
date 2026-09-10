"""
Classifier Factory and Registry for SupportIQ AI.
"""
from typing import Any, Dict, List
import os
from .majority_classifier import MajorityClassifier
from .tfidf_classifier import TfidfLogisticClassifier
from .semantic_classifier import SemanticClassifier


class ClassifierFactory:
    """Factory to instantiate and load intent classifiers."""

    @staticmethod
    def create(model_type: str = "semantic", config_path: str = "config/brand_config.yaml") -> Any:
        m_type = model_type.lower()
        if m_type in ["majority", "baseline1", "majority_baseline"]:
            return MajorityClassifier()
        elif m_type in ["tfidf", "tfidf_logistic", "baseline2", "logistic"]:
            return TfidfLogisticClassifier()
        elif m_type in ["semantic", "proposed", "supportiq"]:
            return SemanticClassifier(config_path=config_path)
        else:
            raise ValueError(f"Unknown classifier type: {model_type}. Expected: majority, tfidf_logistic, semantic.")

    @staticmethod
    def load(filepath: str, model_type: str = "semantic") -> Any:
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Model file not found: {filepath}")
        
        m_type = model_type.lower()
        if "majority" in m_type:
            return MajorityClassifier.load(filepath)
        elif "tfidf" in m_type:
            return TfidfLogisticClassifier.load(filepath)
        else:
            return SemanticClassifier.load(filepath)
