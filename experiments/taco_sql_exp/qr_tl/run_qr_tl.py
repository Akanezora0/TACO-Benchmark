"""
QR+TL experiment setting implementation

+ Question Rewriting + Table Linking
Uses question rewriting and table retrieval
"""

import sys
import os
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from experiments.taco_sql_exp.experiment_runner import run_experiment


def main():
    """Run QR+TL experiment"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Run QR+TL experiment setting (+ Question Rewriting + Table Linking)")
    parser.add_argument("--model", type=str, default="gpt-4o", help="Model name")
    parser.add_argument("--dataset", type=str, default="taco_beijing", help="Dataset name")
    parser.add_argument("--test_data", type=str, help="Test data path")
    parser.add_argument("--output", type=str, help="Output path")
    parser.add_argument("--qr_temperature", type=float, default=0.3, help="Question Rewriting temperature")
    parser.add_argument("--tl_top_k", type=int, default=5, help="Table Linking Top-K")
    
    args = parser.parse_args()
    
    # Set default output path
    if not args.output:
        args.output = f"experiments/results/qr_tl_{args.model}_{args.dataset}.json"
    
    print(f"Running QR+TL experiment setting (+ Question Rewriting + Table Linking)")
    print(f"  Model: {args.model}")
    print(f"  Dataset: {args.dataset}")
    print(f"  QR temperature: {args.qr_temperature}")
    print(f"  TL Top-K: {args.tl_top_k}")
    print(f"  Output: {args.output}")
    print()
    
    # Run experiment
    results = run_experiment(
        setting="qr_tl",
        model_name=args.model,
        dataset_name=args.dataset,
        test_data_path=args.test_data,
        output_path=args.output,
        qr_temperature=args.qr_temperature,
        tl_top_k=args.tl_top_k
    )
    
    print(f"\nExperiment complete!")
    print(f"  Queries processed: {len(results)}")
    print(f"  Results saved to: {args.output}")


if __name__ == "__main__":
    main()
