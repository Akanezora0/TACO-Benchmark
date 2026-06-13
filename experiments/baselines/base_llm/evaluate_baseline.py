"""
Baseline experiment framework: simple Text-to-SQL evaluation
Does not use the TACO-SQL framework or complex rule matching
Provides sufficient context for the model to perform direct Text-to-SQL conversion
"""

import json
import os
import sqlite3
import yaml
from tqdm import tqdm
from openai import OpenAI
from typing import Dict, List, Tuple, Optional
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading

# Load configuration
def load_config():
    # Look up config.yaml from project root
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.join(current_dir, '..', '..', '..')
    config_path = os.path.join(project_root, 'benchmark', 'generation', 'sql_filling', 'config.yaml')
    if os.path.exists(config_path):
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        return config
    return None

config = load_config()

# Model configuration
MODEL_CONFIGS = {
    'gpt-4': {
        'api_key': config['llm']['api_key'] if config else '',
        'base_url': config['llm']['api_url'] if config else '',
        'model': 'gpt-4',
        'temperature': 0.1,
        'max_tokens': 2000,
        'context_window': 8192
    },
    'gpt-4o': {
        'api_key': config['llm']['api_key'] if config else '',
        'base_url': config['llm']['api_url'] if config else '',
        'model': 'gpt-4o',
        'temperature': 0.1,
        'max_tokens': 2000,
        'context_window': 128000  # GPT-4o has a large context window
    },
    'gpt-4o-mini': {
        'api_key': config['llm']['api_key'] if config else '',
        'base_url': config['llm']['api_url'] if config else '',
        'model': 'gpt-4o-mini',
        'temperature': 0.1,
        'max_tokens': 2000,
        'context_window': 128000  # GPT-4o-mini also has a large context window
    },
    'gpt-o1': {
        'api_key': config['llm']['api_key'] if config else '',
        'base_url': config['llm']['api_url'] if config else '',
        'model': 'o1',
        'temperature': 0.1,
        'max_tokens': 2000,
        'context_window': 200000
    },
    'deepseek-r1': {
        'api_key': config['llm']['api_key'] if config else '',
        'base_url': config['llm']['api_url'] if config else '',
        'model': 'deepseek-r1',
        'temperature': 0.1,
        'max_tokens': 2000,
        'context_window': 64000
    }
}

# Thread-local client storage
thread_local = threading.local()

def get_client(model_name: str) -> OpenAI:
    """Get client for the specified model (thread-safe)"""
    if model_name not in MODEL_CONFIGS:
        raise ValueError(f"Unsupported model: {model_name}")
    
    # Create an independent client for each thread and model combination
    key = f"{model_name}_{threading.current_thread().ident}"
    if not hasattr(thread_local, 'clients'):
        thread_local.clients = {}
    
    if key not in thread_local.clients:
        model_config = MODEL_CONFIGS[model_name]
        thread_local.clients[key] = OpenAI(
            api_key=model_config['api_key'],
            base_url=model_config['base_url']
        )
    
    return thread_local.clients[key]

def load_schema(schema_file: str) -> Dict:
    """Load schema information"""
    with open(schema_file, 'r', encoding='utf-8') as f:
        schema = json.load(f)
    return schema

def format_schema_simple(schema: Dict, max_tables: int = None, max_columns_per_table: int = None) -> Tuple[str, Dict]:
    """
    Simple schema formatting: include as many tables as possible
    No complex rule matching; include enough tables based on model context window size
    
    By default includes all tables, since GPT-4o has 128K tokens and can fit the full schema
    """
    all_tables = schema.get('tables', [])
    
    # If max_tables is not specified, include all tables
    # For GPT-4o (128K tokens), all tables can be included (~10% of context window)
    if max_tables is None:
        selected_tables = all_tables
    else:
        selected_tables = all_tables[:max_tables]
    
    # Format schema text
    text = "Database Schema Information:\n\n"
    
    total_tables = len(selected_tables)
    total_columns = 0
    
    for table in selected_tables:
        table_name = table.get('table_name', '')
        columns = table.get('columns', [])
        
        # If max_columns_per_table is not specified, include all columns
        if max_columns_per_table is not None:
            columns = columns[:max_columns_per_table]
        total_columns += len(columns)
        
        text += f"Table: {table_name}\n"
        text += "  Columns:\n"
        for col in columns:
            col_name = col.get('column_name', '')
            col_type = col.get('data_type', 'TEXT')
            text += f"    - {col_name} ({col_type})\n"
        text += "\n"
    
    # Record configuration info
    config_info = {
        'total_tables_in_schema': len(all_tables),
        'included_tables_count': total_tables,
        'included_columns_count': total_columns,
        'max_tables': max_tables,
        'max_columns_per_table': max_columns_per_table,
        'schema_text_length': len(text),
        'estimated_tokens': len(text) // 4  # Rough estimate
    }
    
    return text, config_info

def generate_sql_baseline(
    client: OpenAI, 
    model_name: str, 
    query: str, 
    schema_text: str, 
    database: str,
    config_info: Dict
) -> Tuple[str, Dict]:
    """Generate SQL using baseline method (simple direct prompt)"""
    model_config = MODEL_CONFIGS[model_name]
    
    # Simple prompt without complex rules
    prompt = f"""You are a SQL expert. Generate SQL queries based on natural language queries and database schema.

{schema_text}

Natural language query: {query}

Requirements:
1. Generate complete, executable SQL statements
2. All table and column names must be wrapped in double quotes (including Chinese and special characters)
3. Ensure SQL syntax is correct and can be executed on SQLite
4. Output only SQL statements, no explanations or comments

Database: {database}

SQL query:"""
    
    # Estimate prompt token count
    prompt_tokens = len(prompt) // 4  # Rough estimate
    
    try:
        response = client.chat.completions.create(
            model=model_config['model'],
            temperature=model_config['temperature'],
            max_tokens=model_config['max_tokens'],
            messages=[
                {"role": "system", "content": "You are a SQL expert."},
                {"role": "user", "content": prompt},
            ],
        )
        sql = response.choices[0].message.content.strip()
        
        # Clean SQL
        if sql.startswith('```'):
            lines = sql.split('\n')
            sql = '\n'.join(lines[1:-1]) if len(lines) > 2 else sql
        sql = sql.strip().rstrip(';') + ';'
        
        # Record generation info
        generation_info = {
            'prompt_tokens_estimated': prompt_tokens,
            'response_tokens_estimated': len(sql) // 4,
            'total_tokens_estimated': prompt_tokens + len(sql) // 4,
            'context_window': model_config['context_window'],
            'truncated': (prompt_tokens + len(sql) // 4) > model_config['context_window'] * 0.9,
            **config_info
        }
        
        return sql, generation_info
    except Exception as e:
        print(f"Failed to generate SQL: {e}")
        return "", {'error': str(e), **config_info}

def execute_sql(db_path: str, sql: str) -> Tuple[bool, Optional[List], Optional[str]]:
    """Execute SQL and return results"""
    try:
        conn = sqlite3.connect(db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        
        cursor.execute(sql)
        results = cursor.fetchall()
        
        result_list = [list(row) for row in results]
        
        conn.close()
        return True, result_list, None
    except Exception as e:
        return False, None, str(e)

def normalize_sql(sql: str) -> str:
    """Normalize SQL (for comparison)"""
    sql = ' '.join(sql.split())
    sql = sql.upper()
    sql = sql.replace('"', '')
    return sql

def compare_results(result1: List, result2: List) -> bool:
    """Compare whether two query results are identical"""
    if len(result1) != len(result2):
        return False
    
    def normalize_row(row):
        return tuple(str(v).strip() if v is not None else '' for v in row)
    
    set1 = set(normalize_row(row) for row in result1)
    set2 = set(normalize_row(row) for row in result2)
    
    return set1 == set2

def evaluate_single_query(
    nl_query_file: str,
    db_path: str,
    schema_file: str,
    model_name: str,
    ground_truth_sql: str,
    ground_truth_results: List,
    max_tables: int = 100,
    max_columns_per_table: int = 30
) -> Dict:
    """Evaluate a single query"""
    # Load NL query
    with open(nl_query_file, 'r', encoding='utf-8') as f:
        nl_data = json.load(f)
    
    query = nl_data.get('natural_language_query', '')
    database = nl_data.get('database', '')
    
    if not query:
        return {
            'success': False,
            'error': 'Missing natural_language_query'
        }
    
    # Load schema
    schema = load_schema(schema_file)
    
    # Format schema (simple and direct, include as many tables as possible)
    schema_text, config_info = format_schema_simple(schema, max_tables, max_columns_per_table)
    
    # Generate SQL
    client = get_client(model_name)
    generated_sql, generation_info = generate_sql_baseline(
        client, model_name, query, schema_text, database, config_info
    )
    
    if not generated_sql:
        return {
            'success': False,
            'error': 'Failed to generate SQL',
            'generation_info': generation_info
        }
    
    # Execute generated SQL
    exec_success, exec_results, exec_error = execute_sql(db_path, generated_sql)
    
    # Execute ground truth SQL
    gt_exec_success, gt_results, gt_error = execute_sql(db_path, ground_truth_sql)
    
    # Evaluate
    result = {
        'query': query,
        'ground_truth_sql': ground_truth_sql,
        'generated_sql': generated_sql,
        'exec_success': exec_success,
        'exec_error': exec_error,
        'exec_results': exec_results if exec_success else None,
        'gt_exec_success': gt_exec_success,
        'gt_results': gt_results if gt_exec_success else None,
        'results_match': False,
        'sql_exact_match': False,
        'generation_info': generation_info
    }
    
    # Exact SQL match
    if normalize_sql(generated_sql) == normalize_sql(ground_truth_sql):
        result['sql_exact_match'] = True
    
    # Result match
    if exec_success and gt_exec_success:
        if len(exec_results) == 0 and len(gt_results) == 0:
            result['results_match'] = True
        elif len(exec_results) > 0 and len(gt_results) > 0:
            if compare_results(exec_results, gt_results):
                result['results_match'] = True
        else:
            result['results_match'] = False
    
    result['success'] = exec_success
    
    return result

def evaluate_database(
    nl_query_dir: str,
    db_path: str,
    schema_file: str,
    model_name: str,
    sql_dir: str,
    max_tables: Optional[int] = None,
    max_columns_per_table: Optional[int] = None,
    limit: Optional[int] = None,
    max_workers: int = 5
) -> Dict:
    """Evaluate all queries for a database (concurrent version)"""
    results = []
    
    # Get NL query file list
    nl_files = [f for f in os.listdir(nl_query_dir) if f.startswith('generated_nl_query_') and f.endswith('.json')]
    nl_files.sort(key=lambda x: int(x.split('_')[-1].split('.')[0]) if x.split('_')[-1].split('.')[0].isdigit() else 0)
    
    if limit:
        nl_files = nl_files[:limit]
    
    # Get SQL file list and index mapping
    sql_file_list = sorted([f for f in os.listdir(sql_dir) if f.startswith('generated_sql_') and f.endswith('.json') and '_error' not in f],
                          key=lambda x: int(x.split('_')[-1].split('.')[0]) if x.split('_')[-1].split('.')[0].isdigit() else 0)
    
    # Create mapping from SQL index to filename
    sql_index_map = {}
    for sql_file in sql_file_list:
        file_idx_str = sql_file.split('_')[-1].split('.')[0]
        if file_idx_str.isdigit():
            sql_index_map[int(file_idx_str)] = sql_file
    
    sql_indices = sorted(sql_index_map.keys())
    sql_count = len(sql_indices)
    
    # Prepare task list
    tasks = []
    for nl_file in nl_files:
        nl_file_path = os.path.join(nl_query_dir, nl_file)
        file_idx_str = nl_file.split('_')[-1].split('.')[0]
        
        if not file_idx_str.isdigit():
            continue
        
        file_idx = int(file_idx_str)
        
        # Compute corresponding SQL file index
        # If NL query index is less than SQL count, use the corresponding SQL index directly
        # Otherwise compute the corresponding base_idx (via modulo)
        if file_idx < sql_count:
            sql_idx = sql_indices[file_idx] if file_idx < len(sql_indices) else None
        else:
            # Compute which variant this is and find the corresponding base_idx
            base_idx = file_idx % sql_count
            sql_idx = sql_indices[base_idx] if base_idx < len(sql_indices) else None
        
        if sql_idx is None:
            continue
        
        sql_file = os.path.join(sql_dir, sql_index_map[sql_idx])
        
        if not os.path.exists(sql_file):
            continue
        
        # Load ground truth SQL
        with open(sql_file, 'r', encoding='utf-8') as f:
            sql_data = json.load(f)
        
        ground_truth_sql = sql_data.get('sql', '')
        ground_truth_results = sql_data.get('results', [])
        
        if not ground_truth_sql:
            continue
        
        tasks.append((nl_file_path, nl_file, ground_truth_sql, ground_truth_results))
    
    # Concurrent evaluation
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(
                evaluate_single_query,
                nl_file_path,
                db_path,
                schema_file,
                model_name,
                ground_truth_sql,
                ground_truth_results,
                max_tables,
                max_columns_per_table
            ): (nl_file_path, nl_file)
            for nl_file_path, nl_file, ground_truth_sql, ground_truth_results in tasks
        }
        
        for future in tqdm(as_completed(futures), total=len(futures), desc=f"Baseline eval {model_name}"):
            nl_file_path, nl_file = futures[future]
            try:
                result = future.result()
                result['file'] = nl_file
                results.append(result)
            except Exception as e:
                print(f"Evaluation failed {nl_file_path}: {e}")
                results.append({
                    'file': nl_file,
                    'success': False,
                    'error': str(e)
                })
    
    # Statistics
    total = len(results)
    exec_success = sum(1 for r in results if r.get('exec_success', False))
    results_match = sum(1 for r in results if r.get('results_match', False))
    sql_exact_match = sum(1 for r in results if r.get('sql_exact_match', False))
    
    # Configuration statistics
    if results:
        avg_schema_tokens = sum(r.get('generation_info', {}).get('estimated_tokens', 0) for r in results) / total
        truncated_count = sum(1 for r in results if r.get('generation_info', {}).get('truncated', False))
    else:
        avg_schema_tokens = 0
        truncated_count = 0
    
    config_stats = {
        'max_tables': max_tables,
        'max_columns_per_table': max_columns_per_table,
        'avg_schema_tokens': avg_schema_tokens,
        'truncated_count': truncated_count,
        'context_window': MODEL_CONFIGS[model_name]['context_window']
    }
    
    return {
        'model': model_name,
        'total': total,
        'exec_success': exec_success,
        'exec_success_rate': exec_success / total if total > 0 else 0,
        'results_match': results_match,
        'results_match_rate': results_match / total if total > 0 else 0,
        'sql_exact_match': sql_exact_match,
        'sql_exact_match_rate': sql_exact_match / total if total > 0 else 0,
        'config': {
            'max_tables': max_tables,
            'max_columns_per_table': max_columns_per_table,
            'context_window': MODEL_CONFIGS[model_name]['context_window']
        },
        'config_stats': config_stats,
        'results': results
    }

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Baseline evaluation: simple Text-to-SQL')
    parser.add_argument('--nl_query_dir', type=str, required=True, help='NL query file directory')
    parser.add_argument('--sql_dir', type=str, required=True, help='SQL file directory')
    parser.add_argument('--db_path', type=str, required=True, help='Database file path')
    parser.add_argument('--schema_file', type=str, required=True, help='Schema file path')
    parser.add_argument('--model', type=str, required=True, choices=['gpt-4', 'gpt-4o', 'gpt-4o-mini', 'gpt-o1', 'deepseek-r1'], help='Model name')
    parser.add_argument('--output_file', type=str, required=True, help='Output results file')
    parser.add_argument('--max_tables', type=int, default=None, help='Maximum number of tables (None means include all tables; default is all tables)')
    parser.add_argument('--max_columns_per_table', type=int, default=None, help='Maximum columns per table (None means include all columns; default is all columns)')
    parser.add_argument('--limit', type=int, default=None, help='Limit number of evaluations (for testing)')
    parser.add_argument('--max_workers', type=int, default=5, help='Number of concurrent threads (default 5)')
    
    args = parser.parse_args()
    
    print(f"Baseline evaluation configuration:")
    print(f"  Model: {args.model}")
    print(f"  Context window: {MODEL_CONFIGS[args.model]['context_window']} tokens")
    print(f"  Max tables: {args.max_tables}")
    print(f"  Max columns per table: {args.max_columns_per_table}")
    
    # Evaluate
    eval_result = evaluate_database(
        args.nl_query_dir,
        args.db_path,
        args.schema_file,
        args.model,
        args.sql_dir,
        args.max_tables,
        args.max_columns_per_table,
        args.limit,
        args.max_workers
    )
    
    # Save results
    os.makedirs(os.path.dirname(args.output_file), exist_ok=True)
    with open(args.output_file, 'w', encoding='utf-8') as f:
        json.dump(eval_result, f, ensure_ascii=False, indent=2)
    
    # Print statistics
    print(f"\nBaseline evaluation results ({args.model}):")
    print(f"  Total: {eval_result['total']}")
    print(f"  Execution success: {eval_result['exec_success']} ({eval_result['exec_success_rate']*100:.2f}%)")
    print(f"  Result match: {eval_result['results_match']} ({eval_result['results_match_rate']*100:.2f}%)")
    print(f"  Exact SQL match: {eval_result['sql_exact_match']} ({eval_result['sql_exact_match_rate']*100:.2f}%)")
    print(f"  Average schema tokens: {eval_result['config_stats']['avg_schema_tokens']:.0f}")
    print(f"  Truncated count: {eval_result['config_stats']['truncated_count']}")
    print(f"\nResults saved to: {args.output_file}")

if __name__ == '__main__':
    main()
