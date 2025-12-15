"""
Batch Evaluation Script for Multiple Base LLM Models

Runs baseline experiments for multiple models and aggregates results.
"""

import json
import os
import argparse
from pathlib import Path
from typing import List, Dict
from concurrent.futures import ThreadPoolExecutor, as_completed
import sys

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from experiments.baselines.base_llm.run_experiment import run_experiment
from experiments.baselines.base_llm.experiment_config import BASELINE_MODEL_CONFIGS


def run_batch_evaluation(
    model_names: List[str],
    test_data_path: str,
    output_dir: str = "experiments/results/baselines",
    max_workers: int = 3
) -> Dict[str, List[Dict]]:
    """
    Run baseline experiments for multiple models
    
    Args:
        model_names: List of model names to evaluate
        test_data_path: Path to test data JSON file
        output_dir: Output directory for results
        max_workers: Maximum parallel workers
        
    Returns:
        Dictionary mapping model names to results
    """
    os.makedirs(output_dir, exist_ok=True)
    
    all_results = {}
    
    def run_single_model(model_name: str):
        """Run experiment for a single model"""
        output_path = os.path.join(output_dir, f"baseline_{model_name.replace('-', '_')}.json")
        try:
            results = run_experiment(
                model_name=model_name,
                test_data_path=test_data_path,
                output_path=output_path
            )
            return model_name, results, None
        except Exception as e:
            return model_name, None, str(e)
    
    # Run experiments (with optional parallelization)
    if max_workers > 1:
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {executor.submit(run_single_model, model): model for model in model_names}
            
            for future in as_completed(futures):
                model_name, results, error = future.result()
                if error:
                    print(f"Error running {model_name}: {error}")
                else:
                    all_results[model_name] = results
                    print(f"Completed: {model_name} ({len(results)} queries)")
    else:
        # Sequential execution
        for model_name in model_names:
            model_name, results, error = run_single_model(model_name)
            if error:
                print(f"Error running {model_name}: {error}")
            else:
                all_results[model_name] = results
                print(f"Completed: {model_name} ({len(results)} queries)")
    
    return all_results


def aggregate_results(results_dict: Dict[str, List[Dict]], output_path: str):
    """
    Aggregate results from multiple models
    
    Args:
        results_dict: Dictionary mapping model names to results
        output_path: Output path for aggregated results
    """
    aggregated = {
        'models': list(results_dict.keys()),
        'total_queries': len(list(results_dict.values())[0]) if results_dict else 0,
        'results_by_model': {}
    }
    
    for model_name, results in results_dict.items():
        aggregated['results_by_model'][model_name] = {
            'total': len(results),
            'successful': sum(1 for r in results if r.get('generated_sql')),
            'errors': sum(1 for r in results if 'error' in r.get('generation_info', {}))
        }
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(aggregated, f, ensure_ascii=False, indent=2)
    
    print(f"\nAggregated results saved to: {output_path}")


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description="Batch evaluate multiple base LLM models")
    parser.add_argument("--models", type=str, nargs='+', help="Model names (or 'all' for all models)")
    parser.add_argument("--test_data", type=str, required=True, help="Test data path")
    parser.add_argument("--output_dir", type=str, default="experiments/results/baselines", help="Output directory")
    parser.add_argument("--max_workers", type=int, default=3, help="Maximum parallel workers")
    parser.add_argument("--aggregate", type=str, help="Path to save aggregated results")
    
    args = parser.parse_args()
    
    # Determine models to evaluate
    if args.models and args.models[0] == 'all':
        model_names = list(BASELINE_MODEL_CONFIGS.keys())
    elif args.models:
        model_names = args.models
    else:
        model_names = list(BASELINE_MODEL_CONFIGS.keys())
    
    print(f"Batch evaluation for base LLM models")
    print(f"  Models: {', '.join(model_names)}")
    print(f"  Test data: {args.test_data}")
    print(f"  Output directory: {args.output_dir}")
    print(f"  Max workers: {args.max_workers}")
    print()
    
    # Run batch evaluation
    results = run_batch_evaluation(
        model_names=model_names,
        test_data_path=args.test_data,
        output_dir=args.output_dir,
        max_workers=args.max_workers
    )
    
    # Aggregate results if requested
    if args.aggregate:
        aggregate_results(results, args.aggregate)
    
    print(f"\nBatch evaluation completed!")
    print(f"  Evaluated models: {len(results)}")
    print(f"  Results saved to: {args.output_dir}")


if __name__ == "__main__":
    main()

