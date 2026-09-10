"""
Evaluation CLI Script.
Executes the evaluation pipeline over the Golden Evaluation Set and generates benchmark results.
"""
import sys
import os
import json
import argparse

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.evaluation_engine.evaluator import ComprehensiveEvaluator


def main():
    parser = argparse.ArgumentParser(description="Evaluate SupportIQ AI against baselines")
    parser.add_argument("--config", default="config/brand_config.yaml", help="Brand configuration")
    parser.add_argument("--golden_set", default="data/golden/golden_set.csv", help="Golden set CSV")
    parser.add_argument("--output", default="data/evaluation/benchmark_results.json", help="Output JSON path")
    args = parser.parse_args()

    print("=" * 60)
    print("SUPPORTIQ AI - COMPREHENSIVE BENCHMARK & EVALUATION")
    print("=" * 60)

    evaluator = ComprehensiveEvaluator(
        config_path=args.config,
        golden_set_path=args.golden_set,
    )

    print("\nRunning multi-system baseline benchmarking (5 systems across 200 Golden cases)...")
    results = evaluator.benchmark_all_baselines()

    print("\n" + "=" * 80)
    print("MODEL BENCHMARK RESULTS TABLE")
    print("=" * 80)
    print(f"{'System':<32} | {'Macro-F1':<9} | {'Accuracy':<9} | {'Reply Qual':<10} | {'Grounding':<10} | {'Unsafe Auto':<11}")
    print("-" * 92)
    for row in results["benchmark_table"]:
        print(
            f"{row['system']:<32} | "
            f"{row['macro_f1']:<9.4f} | "
            f"{row['accuracy']:<9.4f} | "
            f"{row['reply_quality']:<10.2f} | "
            f"{row['grounding']:<10.2f} | "
            f"{row['unsafe_automation_rate']:<11.4f}"
        )
    print("-" * 92)

    supportiq_metrics = results["supportiq_full_eval"]
    print("\nSUPPORTIQ AI SYSTEM PERFORMANCE SUMMARY:")
    print(f"  - Intent Accuracy:           {supportiq_metrics['intent_metrics']['accuracy']*100:.1f}%")
    print(f"  - Intent Macro-F1:           {supportiq_metrics['intent_metrics']['macro_f1']*100:.1f}%")
    print(f"  - Retrieval Recall@1:        {supportiq_metrics['retrieval_metrics']['recall_at_1']*100:.1f}%")
    print(f"  - Retrieval Recall@5:        {supportiq_metrics['retrieval_metrics']['recall_at_5']*100:.1f}%")
    print(f"  - Escalation Precision:      {supportiq_metrics['escalation_metrics']['escalation_precision']*100:.1f}%")
    print(f"  - Escalation Recall:         {supportiq_metrics['escalation_metrics']['escalation_recall']*100:.1f}%")
    print(f"  - Unsafe Auto-Handling Rate: {supportiq_metrics['escalation_metrics']['unsafe_auto_handling_rate']*100:.1f}%")
    print(f"  - Safe Automation Coverage:  {supportiq_metrics['escalation_metrics']['safe_automation_coverage']*100:.1f}%")
    print(f"  - Human-Judge Agreement:     {supportiq_metrics['human_agreement']['human_llm_agreement_pct']:.1f}% (Spearman rho={supportiq_metrics['human_agreement']['spearman_correlation']:.3f})")

    print(f"\nBenchmark results saved to {args.output}!")


if __name__ == "__main__":
    main()
