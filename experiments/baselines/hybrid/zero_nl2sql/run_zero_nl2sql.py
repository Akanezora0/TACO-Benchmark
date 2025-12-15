"""
Zero-NL2SQL Experiment Runner

Zero-NL2SQL uses zero-shot learning with enhanced schema understanding
for SQL generation without task-specific training.
"""

import sys
import os
from pathlib import Path

# Add parent directories to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent.parent))

from experiments.baselines.base_llm.experiment_config import (
    format_schema_for_baseline,
    clean_sql_output,
    BASELINE_MODEL_CONFIGS
)
from experiments.evaluation.exec_eval import execute_sql, compare_results
import json
import argparse
from typing import Dict, List, Optional
from openai import OpenAI


class ZeroNL2SQLRunner:
    """Zero-NL2SQL experiment runner"""
    
    def __init__(self, model_name: str = "gpt-4o", api_key: Optional[str] = None):
        """
        Initialize Zero-NL2SQL runner
        
        Args:
            model_name: Base LLM model name
            api_key: API key for the model
        """
        if model_name not in BASELINE_MODEL_CONFIGS:
            raise ValueError(f"Unsupported model: {model_name}")
        
        self.model_config = BASELINE_MODEL_CONFIGS[model_name]
        self.model_name = model_name
        
        # Initialize OpenAI client
        if api_key:
            self.client = OpenAI(api_key=api_key, base_url=self.model_config.base_url)
        else:
            import yaml
            config_path = Path(__file__).parent.parent.parent.parent.parent / 'benchmark' / 'generation' / 'sql_filling' / 'config.yaml'
            if config_path.exists():
                with open(config_path, 'r') as f:
                    config = yaml.safe_load(f)
                api_key = config.get('llm', {}).get('api_key', '')
            
            self.client = OpenAI(api_key=api_key, base_url=self.model_config.base_url)
    
    def build_zero_shot_prompt(self, query: str, schema_text: str, database: str) -> str:
        """
        Build zero-shot prompt with enhanced schema understanding
        
        Args:
            query: Natural language query
            schema_text: Schema text
            database: Database name
            
        Returns:
            Formatted prompt for zero-shot SQL generation
        """
        prompt = f"""You are a SQL expert. Generate SQL queries in a zero-shot manner with enhanced schema understanding.

Zero-Shot SQL Generation Guidelines:
1. Understand the database schema structure
2. Identify relevant tables and columns for the query
3. Map natural language to SQL constructs
4. Generate syntactically correct SQL

Database Schema:
{schema_text}

Natural Language Query: {query}

Schema Analysis:
- Identify tables mentioned or implied in the query
- Identify columns needed for the query
- Determine relationships between tables (if any)

SQL Query:"""
        
        return prompt
    
    def generate_sql(self, query: str, schema_text: str, database: str) -> str:
        """
        Generate SQL using zero-shot approach
        
        Args:
            query: Natural language query
            schema_text: Schema text
            database: Database name
            
        Returns:
            Generated SQL statement
        """
        prompt = self.build_zero_shot_prompt(query, schema_text, database)
        
        try:
            response = self.client.chat.completions.create(
                model=self.model_config.model_name,
                temperature=self.model_config.temperature,
                max_tokens=self.model_config.max_tokens,
                messages=[
                    {"role": "system", "content": "You are a SQL expert using zero-shot learning with enhanced schema understanding."},
                    {"role": "user", "content": prompt}
                ]
            )
            
            sql = response.choices[0].message.content.strip()
            sql = clean_sql_output(sql)
            
            return sql
        except Exception as e:
            print(f"Error generating SQL: {e}")
            return ""
    
    def run_experiment(
        self,
        test_data: List[Dict],
        output_path: Optional[str] = None
    ) -> List[Dict]:
        """
        Run Zero-NL2SQL experiment on test data
        
        Args:
            test_data: List of test queries
            output_path: Path to save results
            
        Returns:
            List of results
        """
        results = []
        
        for item in test_data:
            query = item.get('natural_language_query', '')
            database = item.get('database', '')
            schema = item.get('schema', {})
            ground_truth_sql = item.get('sql', '')
            
            # Format schema
            schema_text = format_schema_for_baseline(schema)
            
            # Generate SQL
            generated_sql = self.generate_sql(query, schema_text, database)
            
            # Evaluate
            db_path = item.get('db_path', '')
            is_correct = False
            if db_path and generated_sql:
                pred_success, pred_results, _ = execute_sql(db_path, generated_sql)
                gt_success, gt_results, _ = execute_sql(db_path, ground_truth_sql)
                
                if pred_success and gt_success:
                    is_correct = compare_results(pred_results, gt_results)
            
            result = {
                'item_id': item.get('id', ''),
                'query': query,
                'database': database,
                'generated_sql': generated_sql,
                'ground_truth_sql': ground_truth_sql,
                'is_correct': is_correct,
                'method': 'Zero-NL2SQL'
            }
            
            results.append(result)
        
        # Save results
        if output_path:
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(results, f, ensure_ascii=False, indent=2)
        
        return results


def main():
    """Main function"""
    parser = argparse.ArgumentParser(description="Run Zero-NL2SQL experiment")
    parser.add_argument("--model", type=str, default="gpt-4o", help="Base LLM model")
    parser.add_argument("--dataset", type=str, default="taco_beijing", help="Dataset name")
    parser.add_argument("--test_data", type=str, help="Test data path")
    parser.add_argument("--output", type=str, help="Output path")
    parser.add_argument("--api_key", type=str, help="API key")
    
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
        args.output = f"experiments/results/zero_nl2sql_{args.model}_{args.dataset}.json"
    
    # Run experiment
    runner = ZeroNL2SQLRunner(model_name=args.model, api_key=args.api_key)
    results = runner.run_experiment(test_data, args.output)
    
    # Calculate accuracy
    total = len(results)
    correct = sum(1 for r in results if r['is_correct'])
    accuracy = correct / total if total > 0 else 0
    
    print(f"\nZero-NL2SQL Experiment Results:")
    print(f"  Model: {args.model}")
    print(f"  Dataset: {args.dataset}")
    print(f"  Total queries: {total}")
    print(f"  Correct: {correct}")
    print(f"  Execution Accuracy: {accuracy:.4f}")
    print(f"  Results saved to: {args.output}")


if __name__ == "__main__":
    main()

