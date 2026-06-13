"""
TACO-SQL ablation experiment main script

Unified entry point supporting all experiment settings
"""

import sys
import os
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from experiments.taco_sql_exp.experiment_runner import run_experiment


def main():
    """Run ablation experiment"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Run TACO-SQL ablation experiment")
    parser.add_argument(
        "--setting", 
        type=str, 
        choices=["origin", "qr", "qr_tl", "qr_tl_qp"],
        required=True,
        help="Experiment setting"
    )
    parser.add_argument("--model", type=str, default="gpt-4o", help="Model name")
    parser.add_argument("--dataset", type=str, default="taco_beijing", help="Dataset name")
    parser.add_argument("--test_data", type=str, help="Test data path")
    parser.add_argument("--output", type=str, help="Output path")
    
    # Component parameters
    parser.add_argument("--qr_temperature", type=float, default=0.3, help="Question Rewriting temperature")
    parser.add_argument("--tl_top_k", type=int, default=5, help="Table Linking Top-K")
    parser.add_argument("--qp_temperature", type=float, default=0.3, help="Query Planning temperature")
    
    args = parser.parse_args()
    
    # Set default output path
    if not args.output:
        args.output = f"experiments/results/{args.setting}_{args.model}_{args.dataset}.json"
    
    # Print experiment info
    setting_names = {
        "origin": "Origin (original query + full schema)",
        "qr": "QR (+ Question Rewriting)",
        "qr_tl": "QR+TL (+ Question Rewriting + Table Linking)",
        "qr_tl_qp": "Full TACO-SQL (+ Question Rewriting + Table Linking + Query Planning)"
    }
    
    print(f"Running TACO-SQL ablation experiment")
    print(f"  Experiment setting: {setting_names[args.setting]}")
    print(f"  Model: {args.model}")
    print(f"  Dataset: {args.dataset}")
    if args.setting in ["qr", "qr_tl", "qr_tl_qp"]:
        print(f"  QR temperature: {args.qr_temperature}")
    if args.setting in ["qr_tl", "qr_tl_qp"]:
        print(f"  TL Top-K: {args.tl_top_k}")
    if args.setting == "qr_tl_qp":
        print(f"  QP temperature: {args.qp_temperature}")
    print(f"  Output: {args.output}")
    print()
    
    # Build kwargs
    kwargs = {}
    if args.setting in ["qr", "qr_tl", "qr_tl_qp"]:
        kwargs["qr_temperature"] = args.qr_temperature
    if args.setting in ["qr_tl", "qr_tl_qp"]:
        kwargs["tl_top_k"] = args.tl_top_k
    if args.setting == "qr_tl_qp":
        kwargs["qp_temperature"] = args.qp_temperature
    
    # Run experiment
    results = run_experiment(
        setting=args.setting,
        model_name=args.model,
        dataset_name=args.dataset,
        test_data_path=args.test_data,
        output_path=args.output,
        **kwargs
    )
    
    print(f"\nExperiment complete!")
    print(f"  Queries processed: {len(results)}")
    print(f"  Results saved to: {args.output}")


if __name__ == "__main__":
    main()
