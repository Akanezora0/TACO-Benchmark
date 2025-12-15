"""
评测框架：在生成的query-SQL对上测试不同模型的表现
支持GPT-4, GPT-4o, GPT-o1, DeepSeek-R1等模型
"""

import json
import os
import sqlite3
import yaml
from tqdm import tqdm
from openai import OpenAI
from typing import Dict, List, Tuple, Optional
import time

# 加载配置
def load_config():
    config_path = os.path.join(os.path.dirname(__file__), '..', 'sql_filling', 'config.yaml')
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
        'temperature': 0.1
    },
    'gpt-4o': {
        'api_key': config['llm']['api_key'] if config else '',
        'base_url': config['llm']['api_url'] if config else '',
        'model': 'gpt-4o',
        'temperature': 0.1
    },
    'gpt-o1': {
        'api_key': config['llm']['api_key'] if config else '',
        'base_url': config['llm']['api_url'] if config else '',
        'model': 'o1',
        'temperature': 0.1
    },
    'deepseek-r1': {
        'api_key': config['llm']['api_key'] if config else '',
        'base_url': config['llm']['api_url'] if config else '',
        'model': 'deepseek-r1',
        'temperature': 0.1
    }
}

def get_client(model_name: str) -> OpenAI:
    """获取指定模型的客户端"""
    if model_name not in MODEL_CONFIGS:
        raise ValueError(f"不支持的模型: {model_name}")
    
    config = MODEL_CONFIGS[model_name]
    return OpenAI(
        api_key=config['api_key'],
        base_url=config['base_url']
    )

def load_schema(schema_file: str) -> Dict:
    """加载Schema信息"""
    with open(schema_file, 'r', encoding='utf-8') as f:
        schema = json.load(f)
    return schema

def format_schema_for_prompt(schema: Dict, max_tables: int = 10, max_columns_per_table: int = 10) -> str:
    """格式化Schema为Prompt（精简版）"""
    text = "数据库Schema信息：\n\n"
    
    tables = schema.get('tables', [])[:max_tables]
    for table in tables:
        table_name = table.get('table_name', '')
        text += f"表：{table_name}\n"
        text += "  列：\n"
        columns = table.get('columns', [])[:max_columns_per_table]
        for col in columns:
            col_name = col.get('column_name', '')
            col_type = col.get('data_type', 'TEXT')
            text += f"    - {col_name} ({col_type})\n"
        text += "\n"
    
    return text

def generate_sql_with_model(client: OpenAI, model_name: str, query: str, schema_text: str, database: str) -> str:
    """使用指定模型生成SQL"""
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
    
    try:
        model_config = MODEL_CONFIGS[model_name]
        response = client.chat.completions.create(
            model=model_config['model'],
            temperature=model_config['temperature'],
            messages=[
                {"role": "system", "content": "You are a SQL expert."},
                {"role": "user", "content": prompt},
            ],
        )
        sql = response.choices[0].message.content.strip()
        
        # 清理SQL（移除代码块标记等）
        if sql.startswith('```'):
            lines = sql.split('\n')
            sql = '\n'.join(lines[1:-1]) if len(lines) > 2 else sql
        sql = sql.strip().rstrip(';') + ';'
        
        return sql
    except Exception as e:
        print(f"生成SQL失败: {e}")
        return ""

def execute_sql(db_path: str, sql: str) -> Tuple[bool, Optional[List], Optional[str]]:
    """执行SQL并返回结果"""
    try:
        conn = sqlite3.connect(db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        
        cursor.execute(sql)
        results = cursor.fetchall()
        
        # 转换为列表
        result_list = [list(row) for row in results]
        
        conn.close()
        return True, result_list, None
    except Exception as e:
        return False, None, str(e)

def normalize_sql(sql: str) -> str:
    """标准化SQL（用于比较）"""
    # 移除多余空格
    sql = ' '.join(sql.split())
    # 统一大小写
    sql = sql.upper()
    # 移除双引号（用于比较）
    sql = sql.replace('"', '')
    return sql

def compare_results(result1: List, result2: List) -> bool:
    """比较两个查询结果是否相同"""
    if len(result1) != len(result2):
        return False
    
    # 转换为可比较的格式
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
    ground_truth_results: List
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
    schema_text = format_schema_for_prompt(schema)
    
    # 生成SQL
    client = get_client(model_name)
    generated_sql = generate_sql_with_model(client, model_name, query, schema_text, database)
    
    if not generated_sql:
        return {
            'success': False,
            'error': 'Failed to generate SQL'
        }
    
    # 执行生成的SQL
    exec_success, exec_results, exec_error = execute_sql(db_path, generated_sql)
    
    # 执行ground truth SQL获取结果
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
        'sql_exact_match': False
    }
    
    # SQL精确匹配
    if normalize_sql(generated_sql) == normalize_sql(ground_truth_sql):
        result['sql_exact_match'] = True
    
    # 结果匹配
    if exec_success and gt_exec_success:
        # 处理空结果的情况
        if len(exec_results) == 0 and len(gt_results) == 0:
            result['results_match'] = True
        elif len(exec_results) > 0 and len(gt_results) > 0:
            if compare_results(exec_results, gt_results):
                result['results_match'] = True
        # 如果一个是空结果，另一个不是，则不匹配
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
    limit: Optional[int] = None
) -> Dict:
    """评测一个数据库的所有query"""
    results = []
    
    # 获取NL查询文件列表
    nl_files = [f for f in os.listdir(nl_query_dir) if f.startswith('generated_nl_query_') and f.endswith('.json')]
    nl_files.sort(key=lambda x: int(x.split('_')[-1].split('.')[0]) if x.split('_')[-1].split('.')[0].isdigit() else 0)
    
    if limit:
        nl_files = nl_files[:limit]
    
    for nl_file in tqdm(nl_files, desc=f"评测 {model_name}"):
        nl_file_path = os.path.join(nl_query_dir, nl_file)
        
        # 找到对应的SQL文件
        file_idx = nl_file.split('_')[-1].split('.')[0]
        sql_file = os.path.join(sql_dir, f'generated_sql_{file_idx}.json')
        
        if not os.path.exists(sql_file):
            continue
        
        # 加载ground truth SQL
        with open(sql_file, 'r', encoding='utf-8') as f:
            sql_data = json.load(f)
        
        ground_truth_sql = sql_data.get('sql', '')
        ground_truth_results = sql_data.get('results', [])
        
        if not ground_truth_sql:
            continue
        
        # 评测
        result = evaluate_single_query(
            nl_file_path,
            db_path,
            schema_file,
            model_name,
            ground_truth_sql,
            ground_truth_results
        )
        
        result['file'] = nl_file
        results.append(result)
        
        # 添加延迟避免API限流
        time.sleep(0.5)
    
    # 统计
    total = len(results)
    exec_success = sum(1 for r in results if r.get('exec_success', False))
    results_match = sum(1 for r in results if r.get('results_match', False))
    sql_exact_match = sum(1 for r in results if r.get('sql_exact_match', False))
    
    return {
        'model': model_name,
        'total': total,
        'exec_success': exec_success,
        'exec_success_rate': exec_success / total if total > 0 else 0,
        'results_match': results_match,
        'results_match_rate': results_match / total if total > 0 else 0,
        'sql_exact_match': sql_exact_match,
        'sql_exact_match_rate': sql_exact_match / total if total > 0 else 0,
        'results': results
    }

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='评测模型在生成的query-SQL对上的表现')
    parser.add_argument('--nl_query_dir', type=str, required=True, help='NL查询文件目录')
    parser.add_argument('--sql_dir', type=str, required=True, help='SQL文件目录')
    parser.add_argument('--db_path', type=str, required=True, help='数据库文件路径')
    parser.add_argument('--schema_file', type=str, required=True, help='Schema文件路径')
    parser.add_argument('--model', type=str, required=True, choices=['gpt-4', 'gpt-4o', 'gpt-o1', 'deepseek-r1'], help='模型名称')
    parser.add_argument('--output_file', type=str, required=True, help='输出结果文件')
    parser.add_argument('--limit', type=int, default=None, help='限制评测数量（用于测试）')
    
    args = parser.parse_args()
    
    print(f"开始评测模型: {args.model}")
    print(f"NL查询目录: {args.nl_query_dir}")
    print(f"SQL目录: {args.sql_dir}")
    print(f"数据库: {args.db_path}")
    
    # 评测
    eval_result = evaluate_database(
        args.nl_query_dir,
        args.db_path,
        args.schema_file,
        args.model,
        args.sql_dir,
        args.limit
    )
    
    # 保存结果
    os.makedirs(os.path.dirname(args.output_file), exist_ok=True)
    with open(args.output_file, 'w', encoding='utf-8') as f:
        json.dump(eval_result, f, ensure_ascii=False, indent=2)
    
    # 打印统计
    print(f"\n评测结果 ({args.model}):")
    print(f"  总数: {eval_result['total']}")
    print(f"  执行成功: {eval_result['exec_success']} ({eval_result['exec_success_rate']*100:.2f}%)")
    print(f"  结果匹配: {eval_result['results_match']} ({eval_result['results_match_rate']*100:.2f}%)")
    print(f"  SQL精确匹配: {eval_result['sql_exact_match']} ({eval_result['sql_exact_match_rate']*100:.2f}%)")
    print(f"\n结果已保存到: {args.output_file}")

if __name__ == '__main__':
    main()

