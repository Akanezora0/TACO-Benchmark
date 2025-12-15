"""
Main Experiment Runner for Base LLM Baseline Experiments

Provides unified interface for running baseline experiments with all base LLM models.
"""

import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional
import argparse

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from experiments.baselines.base_llm.model_wrapper import create_model_from_config
from experiments.baselines.base_llm.experiment_config import (
    BASELINE_MODEL_CONFIGS,
    format_schema_for_baseline,
    clean_sql_output
)
from experiments.baselines.base_llm.prompt_strategy import BaseLLMPromptStrategy


class BaseLLMExperimentRunner:
    """Runner for base LLM baseline experiments"""
    
    def __init__(self, model_name: str, config: Optional[Dict] = None):
        """
        Initialize experiment runner
        
        Args:
            model_name: Model name (e.g., 'gpt-4o', 'gpt-o1')
            config: Optional additional configuration
        """
        if model_name not in BASELINE_MODEL_CONFIGS:
            raise ValueError(f"Unknown model: {model_name}. Available: {list(BASELINE_MODEL_CONFIGS.keys())}")
        
        self.model_name = model_name
        self.model = create_model_from_config(BASELINE_MODEL_CONFIGS[model_name])
        self.config = config or {}
    
    def run_single_query(
        self,
        query: str,
        schema: Dict,
        database: str,
        ground_truth_sql: Optional[str] = None
    ) -> Dict:
        """
        Run experiment on a single query
        
        Args:
            query: Natural language query
            schema: Schema dictionary
            database: Database name
            ground_truth_sql: Optional ground truth SQL
            
        Returns:
            Result dictionary
        """
        # Format schema
        max_tables = self.config.get('max_tables', None)
        max_columns = self.config.get('max_columns_per_table', None)
        
        schema_text = format_schema_for_baseline(
            schema,
            max_tables=max_tables,
            max_columns_per_table=max_columns
        )
        
        # Generate SQL
        generated_sql, generation_info = self.model.generate_sql(
            query=query,
            schema_text=schema_text,
            database=database
        )
        
        result = {
            'model': self.model_name,
            'query': query,
            'database': database,
            'generated_sql': generated_sql,
            'ground_truth_sql': ground_truth_sql,
            'generation_info': generation_info,
            'schema_info': {
                'max_tables': max_tables,
                'max_columns_per_table': max_columns
            }
        }
        
        return result
    
    def run_batch(
        self,
        test_data: List[Dict],
        output_path: Optional[str] = None
    ) -> List[Dict]:
        """
        Run experiment on batch of queries
        
        Args:
            test_data: List of test data items
            output_path: Optional output path for results
            
        Returns:
            List of result dictionaries
        """
        results = []
        
        for item in test_data:
            query = item.get('natural_language_query', '')
            database = item.get('database', '')
            schema = item.get('schema', {})
            ground_truth_sql = item.get('sql', '')
            item_id = item.get('id', '')
            
            result = self.run_single_query(
                query=query,
                schema=schema,
                database=database,
                ground_truth_sql=ground_truth_sql
            )
            
            result['item_id'] = item_id
            results.append(result)
        
        # Save results
        if output_path:
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(results, f, ensure_ascii=False, indent=2)
        
        return results


def run_experiment(
    model_name: str,
    test_data_path: str,
    output_path: Optional[str] = None,
    max_tables: Optional[int] = None,
    max_columns_per_table: Optional[int] = None
) -> List[Dict]:
    """
    Run baseline experiment for a base LLM model
    
    Args:
        model_name: Model name
        test_data_path: Path to test data JSON file
        output_path: Optional output path
        max_tables: Optional maximum tables in schema
        max_columns_per_table: Optional maximum columns per table
        
    Returns:
        List of results
    """
    # Load test data
    with open(test_data_path, 'r', encoding='utf-8') as f:
        test_data = json.load(f)
    
    # Create runner
    config = {
        'max_tables': max_tables,
        'max_columns_per_table': max_columns_per_table
    }
    runner = BaseLLMExperimentRunner(model_name, config)
    
    # Run experiment
    results = runner.run_batch(test_data, output_path)
    
    return results


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description="Run base LLM baseline experiment")
    parser.add_argument("--model", type=str, required=True, help="Model name")
    parser.add_argument("--test_data", type=str, required=True, help="Test data path")
    parser.add_argument("--output", type=str, help="Output path")
    parser.add_argument("--max_tables", type=int, help="Maximum tables in schema")
    parser.add_argument("--max_columns", type=int, help="Maximum columns per table")
    
    args = parser.parse_args()
    
    # Set default output path
    if not args.output:
        model_safe = args.model.replace('-', '_').replace('.', '_')
        args.output = f"experiments/results/baseline_{model_safe}.json"
    
    print(f"Running baseline experiment")
    print(f"  Model: {args.model}")
    print(f"  Test data: {args.test_data}")
    print(f"  Output: {args.output}")
    if args.max_tables:
        print(f"  Max tables: {args.max_tables}")
    if args.max_columns:
        print(f"  Max columns per table: {args.max_columns}")
    print()
    
    # Run experiment
    results = run_experiment(
        model_name=args.model,
        test_data_path=args.test_data,
        output_path=args.output,
        max_tables=args.max_tables,
        max_columns_per_table=args.max_columns
    )
    
    print(f"\nExperiment completed!")
    print(f"  Processed queries: {len(results)}")
    print(f"  Results saved to: {args.output}")


if __name__ == "__main__":
    main()

