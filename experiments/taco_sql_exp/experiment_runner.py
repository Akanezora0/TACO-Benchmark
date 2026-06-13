"""
TACO-SQL experiment runner

Runs different experiment configurations based on experiment settings
"""

from typing import Dict, List, Optional, Any
import json
import os
from pathlib import Path

from .config import ExperimentConfig, create_experiment_config
from .prompts import (
    create_rewriting_prompt_builder,
    create_planning_prompt_builder,
    create_sql_prompt_builder
)
from .utils import format_schema_simple, format_schema_filtered


class ExperimentRunner:
    """Experiment runner"""
    
    def __init__(self, config: ExperimentConfig):
        """
        Initialize experiment runner
        
        Args:
            config: Experiment configuration
        """
        self.config = config
        
        # Initialize prompt builders
        self.qr_prompt_builder = None
        self.qp_prompt_builder = None
        self.sql_prompt_builder = None
        
        if config.component_config.qr_enabled:
            self.qr_prompt_builder = create_rewriting_prompt_builder(
                temperature=config.component_config.qr_temperature,
                top_p=config.component_config.qr_top_p
            )
        
        if config.component_config.qp_enabled:
            self.qp_prompt_builder = create_planning_prompt_builder(
                temperature=config.component_config.qp_temperature,
                max_tokens=config.component_config.qp_max_tokens
            )
        
        self.sql_prompt_builder = create_sql_prompt_builder(
            temperature=config.component_config.sg_temperature,
            max_tokens=config.component_config.sg_max_tokens,
            use_filtered_schema=config.component_config.tl_enabled
        )
    
    def run_single_query(
        self,
        query: str,
        schema: Dict,
        database: str,
        ground_truth_sql: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Run experiment for a single query
        
        Args:
            query: Original query
            schema: Schema dictionary
            database: Database name
            ground_truth_sql: Ground truth SQL (optional)
            
        Returns:
            Experiment result dictionary
        """
        result = {
            'original_query': query,
            'rewritten_query': None,
            'relevant_tables': None,
            'execution_plan': None,
            'generated_sql': None,
            'schema_info': {},
            'errors': []
        }
        
        try:
            # Step 1: Question Rewriting
            if self.config.component_config.qr_enabled and self.qr_prompt_builder:
                rewritten_query = self._rewrite_query(query)
                result['rewritten_query'] = rewritten_query
            else:
                result['rewritten_query'] = query
            
            # Step 2: Table Linking
            if self.config.component_config.tl_enabled:
                relevant_tables = self._retrieve_tables(result['rewritten_query'])
                result['relevant_tables'] = relevant_tables
                
                # Format filtered schema
                schema_text, schema_info = format_schema_filtered(
                    schema,
                    relevant_tables
                )
            else:
                result['relevant_tables'] = []
                
                # Format full schema
                schema_text, schema_info = format_schema_simple(schema)
            
            result['schema_info'] = schema_info
            
            # Step 3: Query Planning
            if self.config.component_config.qp_enabled and self.qp_prompt_builder:
                execution_plan = self._plan_query(
                    result['rewritten_query'],
                    result['relevant_tables'],
                    schema
                )
                result['execution_plan'] = execution_plan
            else:
                # Default single-step plan
                result['execution_plan'] = [{
                    'subquery': result['rewritten_query'],
                    'tables': result['relevant_tables'],
                    'order': 1,
                    'dependencies': []
                }]
            
            # Step 4: SQL Generation
            if result['execution_plan']:
                # Use the first subquery to generate SQL (simplified implementation)
                plan_item = result['execution_plan'][0]
                generated_sql = self._generate_sql(
                    query=result['rewritten_query'],
                    schema_text=schema_text,
                    database=database,
                    is_filtered=self.config.component_config.tl_enabled
                )
                result['generated_sql'] = generated_sql
        
        except Exception as e:
            result['errors'].append(str(e))
        
        return result
    
    def _rewrite_query(self, query: str) -> str:
        """
        Rewrite query (placeholder implementation)
        
        Args:
            query: Original query
            
        Returns:
            Rewritten query
        """
        # TODO: Implement actual LLM call
        # messages = self.qr_prompt_builder.build_messages(query)
        # response = llm_client.chat.completions.create(...)
        # return response.choices[0].message.content
        
        # Placeholder: return original query
        return query
    
    def _retrieve_tables(self, query: str) -> List[str]:
        """
        Retrieve relevant tables (placeholder implementation)
        
        Args:
            query: Query text
            
        Returns:
            List of relevant tables
        """
        # TODO: Implement actual Table Linking
        # from taco_sql.table_linking.retrieval.table_retrieval import TableRetriever
        # retriever = TableRetriever(...)
        # tables = retriever.retrieve_top_k_tables([query], k=self.config.component_config.tl_top_k)
        # return tables[0]
        
        # Placeholder: return empty list
        return []
    
    def _plan_query(
        self,
        query: str,
        relevant_tables: List[str],
        schema: Dict
    ) -> List[Dict]:
        """
        Plan query (placeholder implementation)
        
        Args:
            query: Query text
            relevant_tables: List of relevant tables
            schema: Schema dictionary
            
        Returns:
            Execution plan list
        """
        # TODO: Implement actual Query Planning
        # prompt = self.qp_prompt_builder.build_prompt(query, relevant_tables, schema)
        # response = llm_client.chat.completions.create(...)
        # plan_json = response.choices[0].message.content
        # plan = self.qp_prompt_builder.parse_plan(plan_json)
        # return plan
        
        # Placeholder: return default plan
        return [{
            'subquery': query,
            'tables': relevant_tables,
            'order': 1,
            'dependencies': []
        }]
    
    def _generate_sql(
        self,
        query: str,
        schema_text: str,
        database: str,
        is_filtered: bool = False
    ) -> str:
        """
        Generate SQL (placeholder implementation)
        
        Args:
            query: Query text
            schema_text: Schema text
            database: Database name
            is_filtered: Whether schema has been filtered
            
        Returns:
            Generated SQL
        """
        # TODO: Implement actual SQL generation
        # prompt = self.sql_prompt_builder.build_prompt(
        #     query, schema_text, database, is_filtered=is_filtered
        # )
        # response = llm_client.chat.completions.create(...)
        # sql = response.choices[0].message.content
        # sql = self.sql_prompt_builder.clean_sql(sql)
        # return sql
        
        # Placeholder: return placeholder SQL
        return "SELECT * FROM \"table_name\";"
    
    def run_batch(
        self,
        test_data: List[Dict],
        output_path: Optional[str] = None
    ) -> List[Dict]:
        """
        Run experiments in batch
        
        Args:
            test_data: Test data list
            output_path: Output path (optional)
            
        Returns:
            Experiment result list
        """
        results = []
        
        for item in test_data:
            query = item.get('natural_language_query', '')
            database = item.get('database', '')
            schema = item.get('schema', {})
            ground_truth_sql = item.get('sql', '')
            
            result = self.run_single_query(
                query=query,
                schema=schema,
                database=database,
                ground_truth_sql=ground_truth_sql
            )
            
            # Add metadata
            result['item_id'] = item.get('id', '')
            result['database'] = database
            result['ground_truth_sql'] = ground_truth_sql
            
            results.append(result)
        
        # Save results
        if output_path:
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(results, f, ensure_ascii=False, indent=2)
        
        return results


def run_experiment(
    setting: str,
    model_name: str,
    dataset_name: str = "taco_beijing",
    test_data_path: Optional[str] = None,
    output_path: Optional[str] = None,
    **kwargs
) -> List[Dict]:
    """
    Main function to run an experiment
    
    Args:
        setting: Experiment setting ("origin", "qr", "qr_tl", "qr_tl_qp")
        model_name: Model name
        dataset_name: Dataset name
        test_data_path: Test data path
        output_path: Output path
        **kwargs: Other configuration parameters
        
    Returns:
        Experiment result list
    """
    # Create configuration
    config = create_experiment_config(
        setting=setting,
        model_name=model_name,
        dataset_name=dataset_name,
        **kwargs
    )
    
    # Create runner
    runner = ExperimentRunner(config)
    
    # Load test data
    if test_data_path and os.path.exists(test_data_path):
        with open(test_data_path, 'r', encoding='utf-8') as f:
            test_data = json.load(f)
    else:
        # Use default path
        default_path = f"benchmark/data/final/{dataset_name}/test.json"
        if os.path.exists(default_path):
            with open(default_path, 'r', encoding='utf-8') as f:
                test_data = json.load(f)
        else:
            raise FileNotFoundError(f"Test data not found: {test_data_path or default_path}")
    
    # Run experiment
    results = runner.run_batch(test_data, output_path)
    
    return results


# Example usage
if __name__ == "__main__":
    # Example: run Origin setting experiment
    print("Running Origin setting experiment...")
    origin_results = run_experiment(
        setting="origin",
        model_name="gpt-4o",
        dataset_name="taco_beijing",
        output_path="results/origin_gpt4o.json"
    )
    print(f"Done, processed {len(origin_results)} queries")
    
    # Example: run full TACO-SQL experiment
    print("\nRunning full TACO-SQL experiment...")
    full_results = run_experiment(
        setting="qr_tl_qp",
        model_name="gpt-4o",
        dataset_name="taco_beijing",
        output_path="results/qr_tl_qp_gpt4o.json"
    )
    print(f"Done, processed {len(full_results)} queries")
