"""
QR+TL+QP experiment setting implementation

Full TACO-SQL: + Question Rewriting + Table Linking + Query Planning
"""

import sys
import os
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from experiments.taco_sql_exp.experiment_runner import run_experiment


def main():
    """Run QR+TL+QP experiment"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Run full TACO-SQL experiment (+ Question Rewriting + Table Linking + Query Planning)")
    parser.add_argument("--model", type=str, default="gpt-4o", help="Model name")
    parser.add_argument("--dataset", type=str, default="taco_beijing", help="Dataset name")
    parser.add_argument("--test_data", type=str, help="Test data path")
    parser.add_argument("--output", type=str, help="Output path")
    parser.add_argument("--qr_temperature", type=float, default=0.3, help="Question Rewriting temperature")
    parser.add_argument("--tl_top_k", type=int, default=5, help="Table Linking Top-K")
    parser.add_argument("--qp_temperature", type=float, default=0.3, help="Query Planning temperature")
    
    args = parser.parse_args()
    
    # Set default output path
    if not args.output:
        args.output = f"experiments/results/qr_tl_qp_{args.model}_{args.dataset}.json"
    
    print(f"Running full TACO-SQL experiment (+ Question Rewriting + Table Linking + Query Planning)")
    print(f"  Model: {args.model}")
    print(f"  Dataset: {args.dataset}")
    print(f"  QR temperature: {args.qr_temperature}")
    print(f"  TL Top-K: {args.tl_top_k}")
    print(f"  QP temperature: {args.qp_temperature}")
    print(f"  Output: {args.output}")
    print()
    
    # Run experiment
    results = run_experiment(
        setting="qr_tl_qp",
        model_name=args.model,
        dataset_name=args.dataset,
        test_data_path=args.test_data,
        output_path=args.output,
        qr_temperature=args.qr_temperature,
        tl_top_k=args.tl_top_k,
        qp_temperature=args.qp_temperature
    )
    
    print(f"\nExperiment complete!")
    print(f"  Queries processed: {len(results)}")
    print(f"  Results saved to: {args.output}")


if __name__ == "__main__":
    main()
