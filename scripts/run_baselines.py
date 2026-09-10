"""
Baseline Runner Script.
Trains and evaluates all baseline intent classifiers and saves models.
"""
import sys
import os
import json
import argparse

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.intent_engine import MajorityClassifier, TfidfLogisticClassifier, SemanticClassifier


def main():
    parser = argparse.ArgumentParser(description="Train baseline intent models")
    parser.add_argument("--train_file", default="data/processed/train.jsonl", help="Training JSONL")
    parser.add_argument("--config", default="config/brand_config.yaml", help="Config YAML")
    parser.add_argument("--models_dir", default="models", help="Directory to save models")
    args = parser.parse_args()

    print("=" * 60)
    print("SUPPORTIQ AI - BASELINE & PROPOSED MODEL TRAINING")
    print("=" * 60)

    if not os.path.exists(args.train_file):
        print(f"Error: Training file not found at {args.train_file}. Run prepare_data.py first!")
        sys.exit(1)

    os.makedirs(args.models_dir, exist_ok=True)

    train_convs = [json.loads(line) for line in open(args.train_file, "r", encoding="utf-8")]
    texts = [c.get("cleaned_customer_message", c.get("customer_message", "")) for c in train_convs]
    labels = [c.get("intent", "order_tracking") for c in train_convs]

    print(f"Training on {len(texts)} conversation records across {len(set(labels))} intents...")

    # 1. Majority Baseline
    print("\n[1/3] Training Majority Class Baseline...")
    maj_clf = MajorityClassifier()
    maj_clf.fit(texts, labels)
    maj_path = os.path.join(args.models_dir, "majority_classifier.pkl")
    maj_clf.save(maj_path)
    print(f"  -> Saved to {maj_path} (Majority intent: {maj_clf.majority_intent})")

    # 2. TF-IDF + Logistic Regression
    print("\n[2/3] Training TF-IDF + Logistic Regression Baseline...")
    tfidf_clf = TfidfLogisticClassifier()
    tfidf_clf.fit(texts, labels)
    tfidf_path = os.path.join(args.models_dir, "tfidf_logistic_classifier.pkl")
    tfidf_clf.save(tfidf_path)
    print(f"  -> Saved to {tfidf_path}")

    # 3. Proposed Semantic Classifier
    print("\n[3/3] Training Proposed Semantic Classifier...")
    sem_clf = SemanticClassifier(config_path=args.config)
    sem_clf.fit(texts, labels)
    sem_path = os.path.join(args.models_dir, "semantic_classifier.pkl")
    sem_clf.save(sem_path)
    print(f"  -> Saved to {sem_path}")

    print("\nAll intent models successfully trained and serialized!")


if __name__ == "__main__":
    main()
