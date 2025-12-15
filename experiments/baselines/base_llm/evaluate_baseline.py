"""
Baseline实验框架：简单的Text-to-SQL评测
不使用TACO-SQL框架，不使用复杂的规则匹配
只提供足够的上下文信息，让模型直接进行Text-to-SQL转换
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

# 加载配置
def load_config():
    # 从项目根目录查找config.yaml
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.join(current_dir, '..', '..', '..')
    config_path = os.path.join(project_root, 'benchmark', 'generation', 'sql_filling', 'config.yaml')
    if os.path.exists(config_path):
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        return config
    return None

config = load_config()

# 模型配置
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
        'context_window': 128000  # GPT-4o的上下文窗口很大
    },
    'gpt-4o-mini': {
        'api_key': config['llm']['api_key'] if config else '',
        'base_url': config['llm']['api_url'] if config else '',
        'model': 'gpt-4o-mini',
        'temperature': 0.1,
        'max_tokens': 2000,
        'context_window': 128000  # GPT-4o-mini的上下文窗口也很大
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

# 线程本地存储客户端
thread_local = threading.local()

def get_client(model_name: str) -> OpenAI:
    """获取指定模型的客户端（线程安全）"""
    if model_name not in MODEL_CONFIGS:
        raise ValueError(f"不支持的模型: {model_name}")
    
    # 为每个线程和模型组合创建独立的客户端
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
    """加载Schema信息"""
    with open(schema_file, 'r', encoding='utf-8') as f:
        schema = json.load(f)
    return schema

def format_schema_simple(schema: Dict, max_tables: int = None, max_columns_per_table: int = None) -> Tuple[str, Dict]:
    """
    简单格式化Schema：包含尽可能多的表信息
    不使用复杂的规则匹配，只根据模型的上下文窗口大小包含足够多的表
    
    默认包含所有表，因为GPT-4o有128K tokens，可以容纳完整的Schema
    """
    all_tables = schema.get('tables', [])
    
    # 如果未指定max_tables，则包含所有表
    # 对于GPT-4o (128K tokens)，可以包含所有表（约10%的上下文窗口）
    if max_tables is None:
        selected_tables = all_tables
    else:
        selected_tables = all_tables[:max_tables]
    
    # 格式化Schema文本
    text = "数据库Schema信息：\n\n"
    
    total_tables = len(selected_tables)
    total_columns = 0
    
    for table in selected_tables:
        table_name = table.get('table_name', '')
        columns = table.get('columns', [])
        
        # 如果未指定max_columns_per_table，则包含所有列
        if max_columns_per_table is not None:
            columns = columns[:max_columns_per_table]
        total_columns += len(columns)
        
        text += f"表：{table_name}\n"
        text += "  列：\n"
        for col in columns:
            col_name = col.get('column_name', '')
            col_type = col.get('data_type', 'TEXT')
            text += f"    - {col_name} ({col_type})\n"
        text += "\n"
    
    # 记录配置信息
    config_info = {
        'total_tables_in_schema': len(all_tables),
        'included_tables_count': total_tables,
        'included_columns_count': total_columns,
        'max_tables': max_tables,
        'max_columns_per_table': max_columns_per_table,
        'schema_text_length': len(text),
        'estimated_tokens': len(text) // 4  # 粗略估算
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
    """使用Baseline方法生成SQL（简单直接的prompt）"""
    model_config = MODEL_CONFIGS[model_name]
    
    # 简单的prompt，不添加复杂规则
    prompt = f"""你是一个SQL专家。根据自然语言查询和数据库Schema，生成对应的SQL查询语句。

{schema_text}

自然语言查询：{query}

要求：
1. 生成完整、可执行的SQL语句
2. 所有表名和列名必须用双引号包裹（包括中文和特殊字符）
3. 确保SQL语法正确，可以在SQLite上执行
4. 只输出SQL语句，不要添加任何解释或注释

数据库：{database}

SQL查询："""
    
    # 估算prompt token数
    prompt_tokens = len(prompt) // 4  # 粗略估算
    
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
        
        # 清理SQL
        if sql.startswith('```'):
            lines = sql.split('\n')
            sql = '\n'.join(lines[1:-1]) if len(lines) > 2 else sql
        sql = sql.strip().rstrip(';') + ';'
        
        # 记录生成信息
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
        print(f"生成SQL失败: {e}")
        return "", {'error': str(e), **config_info}

def execute_sql(db_path: str, sql: str) -> Tuple[bool, Optional[List], Optional[str]]:
    """执行SQL并返回结果"""
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
    """标准化SQL（用于比较）"""
    sql = ' '.join(sql.split())
    sql = sql.upper()
    sql = sql.replace('"', '')
    return sql

def compare_results(result1: List, result2: List) -> bool:
    """比较两个查询结果是否相同"""
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
    """评测单个query"""
    # 加载NL查询
    with open(nl_query_file, 'r', encoding='utf-8') as f:
        nl_data = json.load(f)
    
    query = nl_data.get('natural_language_query', '')
    database = nl_data.get('database', '')
    
    if not query:
        return {
            'success': False,
            'error': 'Missing natural_language_query'
        }
    
    # 加载Schema
    schema = load_schema(schema_file)
    
    # 格式化Schema（简单直接，包含尽可能多的表）
    schema_text, config_info = format_schema_simple(schema, max_tables, max_columns_per_table)
    
    # 生成SQL
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
    
    # 执行生成的SQL
    exec_success, exec_results, exec_error = execute_sql(db_path, generated_sql)
    
    # 执行ground truth SQL
    gt_exec_success, gt_results, gt_error = execute_sql(db_path, ground_truth_sql)
    
    # 评估
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
    
    # SQL精确匹配
    if normalize_sql(generated_sql) == normalize_sql(ground_truth_sql):
        result['sql_exact_match'] = True
    
    # 结果匹配
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
    """评测一个数据库的所有query（并发版本）"""
    results = []
    
    # 获取NL查询文件列表
    nl_files = [f for f in os.listdir(nl_query_dir) if f.startswith('generated_nl_query_') and f.endswith('.json')]
    nl_files.sort(key=lambda x: int(x.split('_')[-1].split('.')[0]) if x.split('_')[-1].split('.')[0].isdigit() else 0)
    
    if limit:
        nl_files = nl_files[:limit]
    
    # 获取SQL文件列表和索引映射
    sql_file_list = sorted([f for f in os.listdir(sql_dir) if f.startswith('generated_sql_') and f.endswith('.json') and '_error' not in f],
                          key=lambda x: int(x.split('_')[-1].split('.')[0]) if x.split('_')[-1].split('.')[0].isdigit() else 0)
    
    # 创建SQL索引到文件名的映射
    sql_index_map = {}
    for sql_file in sql_file_list:
        file_idx_str = sql_file.split('_')[-1].split('.')[0]
        if file_idx_str.isdigit():
            sql_index_map[int(file_idx_str)] = sql_file
    
    sql_indices = sorted(sql_index_map.keys())
    sql_count = len(sql_indices)
    
    # 准备任务列表
    tasks = []
    for nl_file in nl_files:
        nl_file_path = os.path.join(nl_query_dir, nl_file)
        file_idx_str = nl_file.split('_')[-1].split('.')[0]
        
        if not file_idx_str.isdigit():
            continue
        
        file_idx = int(file_idx_str)
        
        # 计算对应的SQL文件索引
        # 如果NL查询索引小于SQL数量，直接使用对应的SQL索引
        # 否则计算对应的base_idx（通过取模）
        if file_idx < sql_count:
            sql_idx = sql_indices[file_idx] if file_idx < len(sql_indices) else None
        else:
            # 计算是第几个变体，找到对应的base_idx
            base_idx = file_idx % sql_count
            sql_idx = sql_indices[base_idx] if base_idx < len(sql_indices) else None
        
        if sql_idx is None:
            continue
        
        sql_file = os.path.join(sql_dir, sql_index_map[sql_idx])
        
        if not os.path.exists(sql_file):
            continue
        
        # 加载ground truth SQL
        with open(sql_file, 'r', encoding='utf-8') as f:
            sql_data = json.load(f)
        
        ground_truth_sql = sql_data.get('sql', '')
        ground_truth_results = sql_data.get('results', [])
        
        if not ground_truth_sql:
            continue
        
        tasks.append((nl_file_path, nl_file, ground_truth_sql, ground_truth_results))
    
    # 并发评测
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
        
        for future in tqdm(as_completed(futures), total=len(futures), desc=f"Baseline评测 {model_name}"):
            nl_file_path, nl_file = futures[future]
            try:
                result = future.result()
                result['file'] = nl_file
                results.append(result)
            except Exception as e:
                print(f"评测失败 {nl_file_path}: {e}")
                results.append({
                    'file': nl_file,
                    'success': False,
                    'error': str(e)
                })
    
    # 统计
    total = len(results)
    exec_success = sum(1 for r in results if r.get('exec_success', False))
    results_match = sum(1 for r in results if r.get('results_match', False))
    sql_exact_match = sum(1 for r in results if r.get('sql_exact_match', False))
    
    # 统计配置信息
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
    
    parser = argparse.ArgumentParser(description='Baseline评测：简单的Text-to-SQL')
    parser.add_argument('--nl_query_dir', type=str, required=True, help='NL查询文件目录')
    parser.add_argument('--sql_dir', type=str, required=True, help='SQL文件目录')
    parser.add_argument('--db_path', type=str, required=True, help='数据库文件路径')
    parser.add_argument('--schema_file', type=str, required=True, help='Schema文件路径')
    parser.add_argument('--model', type=str, required=True, choices=['gpt-4', 'gpt-4o', 'gpt-4o-mini', 'gpt-o1', 'deepseek-r1'], help='模型名称')
    parser.add_argument('--output_file', type=str, required=True, help='输出结果文件')
    parser.add_argument('--max_tables', type=int, default=None, help='最大表数量（None表示包含所有表，默认包含所有表）')
    parser.add_argument('--max_columns_per_table', type=int, default=None, help='每个表最大列数（None表示包含所有列，默认包含所有列）')
    parser.add_argument('--limit', type=int, default=None, help='限制评测数量（用于测试）')
    parser.add_argument('--max_workers', type=int, default=5, help='并发线程数（默认5）')
    
    args = parser.parse_args()
    
    print(f"Baseline评测配置:")
    print(f"  模型: {args.model}")
    print(f"  上下文窗口: {MODEL_CONFIGS[args.model]['context_window']} tokens")
    print(f"  最大表数: {args.max_tables}")
    print(f"  每表最大列数: {args.max_columns_per_table}")
    
    # 评测
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
    
    # 保存结果
    os.makedirs(os.path.dirname(args.output_file), exist_ok=True)
    with open(args.output_file, 'w', encoding='utf-8') as f:
        json.dump(eval_result, f, ensure_ascii=False, indent=2)
    
    # 打印统计
    print(f"\nBaseline评测结果 ({args.model}):")
    print(f"  总数: {eval_result['total']}")
    print(f"  执行成功: {eval_result['exec_success']} ({eval_result['exec_success_rate']*100:.2f}%)")
    print(f"  结果匹配: {eval_result['results_match']} ({eval_result['results_match_rate']*100:.2f}%)")
    print(f"  SQL精确匹配: {eval_result['sql_exact_match']} ({eval_result['sql_exact_match_rate']*100:.2f}%)")
    print(f"  平均Schema tokens: {eval_result['config_stats']['avg_schema_tokens']:.0f}")
    print(f"  截断数量: {eval_result['config_stats']['truncated_count']}")
    print(f"\n结果已保存到: {args.output_file}")

if __name__ == '__main__':
    main()

