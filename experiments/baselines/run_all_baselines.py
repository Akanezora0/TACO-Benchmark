"""
Run All Baseline Experiments

This script runs baseline experiments for all model types to ensure comprehensive evaluation.
"""

import argparse
import subprocess
import sys
from pathlib import Path


def run_base_llm_experiments(dataset: str, models: list, output_dir: str):
    """Run base LLM experiments"""
    print("\n" + "="*80)
    print("Running Base LLM Experiments")
    print("="*80)
    
    for model in models:
        print(f"\nRunning {model}...")
        cmd = [
            sys.executable,
            "experiments/baselines/base_llm/run_experiment.py",
            "--model", model,
            "--test_data", f"benchmark/data/final/{dataset}/test.json",
            "--output", f"{output_dir}/base_llm_{model}_{dataset}.json",
        ]
        subprocess.run(cmd)


def run_llm_based_experiments(dataset: str, models: list, output_dir: str):
    """Run LLM-based method experiments"""
    print("\n" + "="*80)
    print("Running LLM-Based Method Experiments")
    print("="*80)
    
    methods = {
        "din_sql": "experiments/baselines/llm_based/din_sql/run_din_sql.py",
        "mac_sql": "experiments/baselines/llm_based/mac_sql/run_mac_sql.py"
    }
    
    for method, script_path in methods.items():
        for model in models:
            print(f"\nRunning {method} with {model}...")
            cmd = [
                sys.executable,
                script_path,
                "--model", model,
                "--dataset", dataset,
                "--output", f"{output_dir}/llm_based_{method}_{model}_{dataset}.json"
            ]
            subprocess.run(cmd)


def run_sft_based_experiments(dataset: str, model_paths: dict, output_dir: str):
    """Run SFT-based model experiments"""
    print("\n" + "="*80)
    print("Running SFT-Based Model Experiments")
    print("="*80)
    
    for model_name, model_path in model_paths.items():
        print(f"\nRunning {model_name}...")
        cmd = [
            sys.executable,
            "experiments/baselines/sft_based/codes/run_codes.py",
            "--model_path", model_path,
            "--dataset", dataset,
            "--output", f"{output_dir}/sft_based_{model_name}_{dataset}.json"
        ]
        subprocess.run(cmd)


def run_hybrid_experiments(dataset: str, models: list, output_dir: str):
    """Run hybrid method experiments"""
    print("\n" + "="*80)
    print("Running Hybrid Method Experiments")
    print("="*80)
    
    methods = {
        "chess": "experiments/baselines/hybrid/chess/run_chess.py"
    }
    
    for method, script_path in methods.items():
        for model in models:
            print(f"\nRunning {method} with {model}...")
            cmd = [
                sys.executable,
                script_path,
                "--model", model,
                "--dataset", dataset,
                "--output", f"{output_dir}/hybrid_{method}_{model}_{dataset}.json"
            ]
            subprocess.run(cmd)


def main():
    """Main function"""
    parser = argparse.ArgumentParser(description="Run all baseline experiments")
    parser.add_argument("--dataset", type=str, default="taco_beijing", help="Dataset name")
    parser.add_argument("--output_dir", type=str, default="experiments/results", help="Output directory")
    parser.add_argument("--base_llm", action="store_true", help="Run base LLM experiments")
    parser.add_argument("--llm_based", action="store_true", help="Run LLM-based experiments")
    parser.add_argument("--sft_based", action="store_true", help="Run SFT-based experiments")
    parser.add_argument("--hybrid", action="store_true", help="Run hybrid experiments")
    parser.add_argument("--all", action="store_true", help="Run all experiments")
    
    args = parser.parse_args()
    
    # Default models
    base_llm_models = ["gpt-4o", "gpt-4o-mini", "gpt-o1", "deepseek-r1"]
    llm_based_models = ["gpt-4o"]
    sft_model_paths = {
        "codes-33b": "models/codes-33b",
        "codes-15b": "models/codes-15b"
    }
    hybrid_models = ["gpt-4o"]
    
    if args.all:
        args.base_llm = True
        args.llm_based = True
        args.sft_based = True
        args.hybrid = True
    
    # Create output directory
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    
    print("="*80)
    print("Running All Baseline Experiments")
    print(f"Dataset: {args.dataset}")
    print(f"Output Directory: {args.output_dir}")
    print("="*80)
    
    if args.base_llm:
        run_base_llm_experiments(args.dataset, base_llm_models, args.output_dir)
    
    if args.llm_based:
        run_llm_based_experiments(args.dataset, llm_based_models, args.output_dir)
    
    if args.sft_based:
        run_sft_based_experiments(args.dataset, sft_model_paths, args.output_dir)
    
    if args.hybrid:
        run_hybrid_experiments(args.dataset, hybrid_models, args.output_dir)
    
    print("\n" + "="*80)
    print("All Experiments Completed!")
    print("="*80)


if __name__ == "__main__":
    main()

