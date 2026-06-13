"""
Origin experiment setting implementation

Original setting: original query + full schema
Does not use any TACO-SQL components
"""

import sys
import os
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from experiments.taco_sql_exp.experiment_runner import run_experiment


def main():
    """Run Origin experiment"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Run Origin experiment setting")
    parser.add_argument("--model", type=str, default="gpt-4o", help="Model name")
    parser.add_argument("--dataset", type=str, default="taco_beijing", help="Dataset name")
    parser.add_argument("--test_data", type=str, help="Test data path")
    parser.add_argument("--output", type=str, help="Output path")
    
    args = parser.parse_args()
    
    # Set default output path
    if not args.output:
        args.output = f"experiments/results/origin_{args.model}_{args.dataset}.json"
    
    print(f"Running Origin experiment setting")
    print(f"  Model: {args.model}")
    print(f"  Dataset: {args.dataset}")
    print(f"  Output: {args.output}")
    print()
    
    # Run experiment
    results = run_experiment(
        setting="origin",
        model_name=args.model,
        dataset_name=args.dataset,
        test_data_path=args.test_data,
        output_path=args.output
    )
    
    print(f"\nExperiment complete!")
    print(f"  Queries processed: {len(results)}")
    print(f"  Results saved to: {args.output}")


if __name__ == "__main__":
    main()
