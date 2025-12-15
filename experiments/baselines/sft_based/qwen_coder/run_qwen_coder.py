"""
Qwen2.5-Coder Experiment Runner

Qwen2.5-Coder is a code generation model from Qwen, fine-tuned for SQL generation.
"""

import sys
import os
from pathlib import Path
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from typing import Dict, List, Optional
import json
import argparse

# Add parent directories to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent.parent))

from experiments.baselines.sft_based.codes.run_codes import CodeSRunner


class QwenCoderRunner(CodeSRunner):
    """Qwen2.5-Coder runner (inherits from CodeS runner)"""
    
    def __init__(
        self,
        model_path: str,
        device: str = "cuda:0",
        schema_filter_enabled: bool = True,
        max_tables: int = 7,
        max_columns: int = 20
    ):
        """
        Initialize Qwen2.5-Coder runner
        
        Args:
            model_path: Path to Qwen2.5-Coder model
            device: Device to run model on
            schema_filter_enabled: Whether to use schema filtering
            max_tables: Maximum number of tables in filtered schema
            max_columns: Maximum number of columns per table
        """
        super().__init__(
            model_path=model_path,
            device=device,
            schema_filter_enabled=schema_filter_enabled,
            schema_filter_model_path=None,
            max_tables=max_tables,
            max_columns=max_columns
        )


def main():
    """Main function"""
    parser = argparse.ArgumentParser(description="Run Qwen2.5-Coder experiment")
    parser.add_argument("--model_path", type=str, required=True, help="Path to Qwen2.5-Coder model")
    parser.add_argument("--dataset", type=str, default="taco_beijing", help="Dataset name")
    parser.add_argument("--test_data", type=str, help="Test data path")
    parser.add_argument("--output", type=str, help="Output path")
    parser.add_argument("--device", type=str, default="cuda:0", help="Device")
    parser.add_argument("--max_tables", type=int, default=7, help="Max tables in schema")
    parser.add_argument("--max_columns", type=int, default=20, help="Max columns per table")
    parser.add_argument("--no_schema_filter", action="store_true", help="Disable schema filtering")
    
    args = parser.parse_args()
    
    # Load test data
    if args.test_data and os.path.exists(args.test_data):
        with open(args.test_data, 'r', encoding='utf-8') as f:
            test_data = json.load(f)
    else:
        default_path = f"benchmark/data/final/{args.dataset}/test.json"
        if os.path.exists(default_path):
            with open(default_path, 'r', encoding='utf-8') as f:
                test_data = json.load(f)
        else:
            raise FileNotFoundError(f"Test data not found: {args.test_data or default_path}")
    
    # Set output path
    if not args.output:
        model_name = Path(args.model_path).name
        args.output = f"experiments/results/qwen_coder_{model_name}_{args.dataset}.json"
    
    # Run experiment
    runner = QwenCoderRunner(
        model_path=args.model_path,
        device=args.device,
        schema_filter_enabled=not args.no_schema_filter,
        max_tables=args.max_tables,
        max_columns=args.max_columns
    )
    results = runner.run_experiment(test_data, args.output)
    
    # Calculate accuracy
    total = len(results)
    correct = sum(1 for r in results if r['is_correct'])
    accuracy = correct / total if total > 0 else 0
    
    print(f"\nQwen2.5-Coder Experiment Results:")
    print(f"  Model: {args.model_path}")
    print(f"  Dataset: {args.dataset}")
    print(f"  Total queries: {total}")
    print(f"  Correct: {correct}")
    print(f"  Execution Accuracy: {accuracy:.4f}")
    print(f"  Results saved to: {args.output}")


if __name__ == "__main__":
    main()

