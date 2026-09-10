"""
Retrieval Index Builder Script.
Builds and saves vector retrieval index over training conversations.
"""
import sys
import os
import json
import argparse

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.retrieval_engine.retriever import HistoricalRetriever


def main():
    parser = argparse.ArgumentParser(description="Build retrieval vector index")
    parser.add_argument("--train_file", default="data/processed/train.jsonl", help="Training conversations JSONL")
    parser.add_argument("--output", default="models/retriever.pkl", help="Output path for retriever pickle")
    parser.add_argument("--config", default="config/brand_config.yaml", help="Brand configuration path")
    args = parser.parse_args()

    print("=" * 60)
    print("SUPPORTIQ AI - RETRIEVAL INDEX BUILDER")
    print("=" * 60)

    if not os.path.exists(args.train_file):
        print(f"Error: Training file not found at {args.train_file}. Run prepare_data.py first!")
        sys.exit(1)

    os.makedirs(os.path.dirname(args.output), exist_ok=True)

    print(f"Loading training conversations from {args.train_file}...")
    train_convs = [json.loads(line) for line in open(args.train_file, "r", encoding="utf-8")]
    print(f"Indexing {len(train_convs)} training conversations (Leakage Prevention Active)...")

    retriever = HistoricalRetriever(config_path=args.config)
    retriever.build_index(train_convs)
    retriever.save(args.output)

    print(f"Retrieval index successfully built and saved to {args.output}!")


if __name__ == "__main__":
    main()
