"""
CodeS Model Experiment Runner

CodeS is a code generation model fine-tuned for SQL generation.
Supports schema filtering and beam search decoding.
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

from experiments.baselines.base_llm.experiment_config import (
    format_schema_for_baseline,
    clean_sql_output,
)
from experiments.evaluation.exec_eval import execute_sql, compare_results


class CodeSRunner:
    """CodeS model experiment runner"""
    
    def __init__(
        self,
        model_path: str,
        device: str = "cuda:0",
        schema_filter_enabled: bool = True,
        schema_filter_model_path: Optional[str] = None,
        max_tables: int = 7,
        max_columns: int = 20
    ):
        """
        Initialize CodeS runner
        
        Args:
            model_path: Path to CodeS model
            device: Device to run model on
            schema_filter_enabled: Whether to use schema filtering
            schema_filter_model_path: Path to schema item classifier
            max_tables: Maximum number of tables in filtered schema
            max_columns: Maximum number of columns per table
        """
        self.device = device
        self.schema_filter_enabled = schema_filter_enabled
        self.max_tables = max_tables
        self.max_columns = max_columns
        
        # Load tokenizer and model
        print(f"Loading CodeS model from {model_path}...")
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            trust_remote_code=True
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            device_map=device,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True
        )
        self.model.eval()
        
        # Load schema filter if enabled
        self.schema_filter = None
        if schema_filter_enabled and schema_filter_model_path:
            # TODO: Load schema item classifier
            # from utils.model_utils.classifier_model import SchemaItemClassifierInference
            # self.schema_filter = SchemaItemClassifierInference(schema_filter_model_path)
            pass
    
    def prepare_input(self, query: str, schema_text: str) -> Dict:
        """
        Prepare input for CodeS model
        
        Args:
            query: Natural language query
            schema_text: Schema text
            
        Returns:
            Prepared input dictionary
        """
        # Format input sequence
        prefix_seq = f"{schema_text}\n\nNatural language query: {query}\n\nSQL query:"
        
        # Tokenize
        input_ids = self.tokenizer(prefix_seq, truncation=False, return_tensors="pt")["input_ids"]
        
        # Truncate if too long (keep last 4096 tokens for context)
        max_length = 4096
        if input_ids.shape[1] > max_length:
            input_ids = input_ids[:, -max_length:]
        
        return {
            "input_ids": input_ids.to(self.device),
            "attention_mask": torch.ones_like(input_ids).to(self.device)
        }
    
    def generate_sql(
        self,
        query: str,
        schema_text: str,
        num_beams: int = 4,
        max_new_tokens: int = 512
    ) -> str:
        """
        Generate SQL using CodeS model
        
        Args:
            query: Natural language query
            schema_text: Schema text
            num_beams: Number of beams for beam search
            max_new_tokens: Maximum tokens to generate
            
        Returns:
            Generated SQL statement
        """
        # Prepare input
        inputs = self.prepare_input(query, schema_text)
        input_length = inputs["input_ids"].shape[1]
        
        # Generate
        with torch.no_grad():
            generate_ids = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                num_beams=num_beams,
                num_return_sequences=1,
                temperature=0.1,
                do_sample=False
            )
        
        # Decode
        generated_sqls = self.tokenizer.batch_decode(
            generate_ids[:, input_length:],
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False
        )
        
        sql = generated_sqls[0] if generated_sqls else ""
        sql = clean_sql_output(sql)
        
        return sql
    
    def run_experiment(
        self,
        test_data: List[Dict],
        output_path: Optional[str] = None
    ) -> List[Dict]:
        """
        Run CodeS experiment on test data
        
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
            
            # Format schema (with filtering if enabled)
            if self.schema_filter_enabled and self.schema_filter:
                # TODO: Apply schema filtering
                schema_text = format_schema_for_baseline(schema)
            else:
                schema_text = format_schema_for_baseline(
                    schema,
                    max_tables=self.max_tables,
                    max_columns_per_table=self.max_columns
                )
            
            # Generate SQL
            generated_sql = self.generate_sql(query, schema_text)
            
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
                'method': 'CodeS',
                'model_path': str(Path(self.model.config.name_or_path)),
                'schema_filtered': self.schema_filter_enabled
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
    parser = argparse.ArgumentParser(description="Run CodeS experiment")
    parser.add_argument("--model_path", type=str, required=True, help="Path to CodeS model")
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
        args.output = f"experiments/results/codes_{model_name}_{args.dataset}.json"
    
    # Run experiment
    runner = CodeSRunner(
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
    
    print(f"\nCodeS Experiment Results:")
    print(f"  Model: {args.model_path}")
    print(f"  Dataset: {args.dataset}")
    print(f"  Total queries: {total}")
    print(f"  Correct: {correct}")
    print(f"  Execution Accuracy: {accuracy:.4f}")
    print(f"  Results saved to: {args.output}")


if __name__ == "__main__":
    main()

