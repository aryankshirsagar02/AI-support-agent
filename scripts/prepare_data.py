"""
Data Preparation Script.
Executes the data pipeline: cleaning, conversation threading, brand filtering, and train/val/test split.
"""
import sys
import os
import argparse

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.data_pipeline.dataset_builder import load_or_build_dataset


def main():
    parser = argparse.ArgumentParser(description="Prepare dataset for SupportIQ AI")
    parser.add_argument("--config", default="config/brand_config.yaml", help="Path to brand config YAML")
    parser.add_argument("--raw_csv", default="data/raw/twcs.csv", help="Path to raw TWCS CSV")
    parser.add_argument("--output_dir", default="data/processed", help="Directory for processed splits")
    parser.add_argument("--samples", type=int, default=1200, help="Target sample size for curated dataset")
    parser.add_argument("--force", action="store_true", help="Force rebuild even if cache exists")
    args = parser.parse_args()

    if args.force:
        for fname in ["train.jsonl", "val.jsonl", "test.jsonl", "all_conversations.jsonl"]:
            p = os.path.join(args.output_dir, fname)
            if os.path.exists(p):
                os.remove(p)

    print("=" * 60)
    print("SUPPORTIQ AI - DATA PREPARATION PIPELINE")
    print("=" * 60)
    stats = load_or_build_dataset(
        config_path=args.config,
        raw_csv_path=args.raw_csv,
        output_dir=args.output_dir,
        target_sample_size=args.samples,
    )
    print("\nDataset Summary:")
    for k, v in stats.items():
        print(f"  - {k}: {v}")
    print("\nData preparation complete!")


if __name__ == "__main__":
    main()
