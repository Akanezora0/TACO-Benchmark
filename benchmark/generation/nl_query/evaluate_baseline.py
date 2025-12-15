"""
Baseline实验框架：不使用TACO-SQL框架，直接进行Text-to-SQL评测

特点：
1. 使用更大的输入窗口（支持更多表信息）
2. 智能提取相关表（基于NL查询或GT SQL涉及的表）
3. 明确配置参数（token限制、表数量限制等）
4. 记录详细的配置信息供论文使用
"""

import json
import os
import sqlite3
import yaml
import re
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
        'temperature': 0.1,
        'max_tokens': 2000,
        'context_window': 8192  # GPT-4的上下文窗口
    },
    'gpt-4o': {
        'api_key': config['llm']['api_key'] if config else '',
        'base_url': config['llm']['api_url'] if config else '',
        'model': 'gpt-4o',
        'temperature': 0.1,
        'max_tokens': 2000,
        'context_window': 128000  # GPT-4o的上下文窗口
    },
    'gpt-o1': {
        'api_key': config['llm']['api_key'] if config else '',
        'base_url': config['llm']['api_url'] if config else '',
        'model': 'o1',
        'temperature': 0.1,
        'max_tokens': 2000,
        'context_window': 200000  # GPT-o1的上下文窗口
    },
    'deepseek-r1': {
        'api_key': config['llm']['api_key'] if config else '',
        'base_url': config['llm']['api_url'] if config else '',
        'model': 'deepseek-r1',
        'temperature': 0.1,
        'max_tokens': 2000,
        'context_window': 64000  # DeepSeek-R1的上下文窗口
    }
}

# Baseline实验配置
BASELINE_CONFIG = {
    'strategy': 'relevant_tables',  # 'all_tables', 'relevant_tables', 'first_n_tables'
    'max_tables': 50,  # 最大表数量（当strategy为first_n_tables时）
    'max_columns_per_table': 20,  # 每个表最大列数
    'include_all_columns': False,  # 是否包含所有列（如果为False，只包含相关列）
    'use_gt_tables': True,  # 是否使用GT SQL涉及的表作为相关表（仅用于分析，实际评测时不使用）
    'reserve_tokens_for_query': 1000,  # 为查询保留的token数
    'reserve_tokens_for_response': 500,  # 为响应保留的token数
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

def extract_relevant_tables_from_nl_query(query: str, schema: Dict, gt_tables: Optional[List[str]] = None) -> List[str]:
    """
    从NL查询中提取相关的表
    
    策略（Baseline不使用GT信息）：
    1. 基于NL查询中的关键词匹配表名
    2. 使用表名的语义部分进行匹配
    3. 如果匹配不到，返回更多表作为fallback（增加表数量）
    """
    # 从NL查询中提取关键词
    query_lower = query.lower()
    
    # 提取可能的表名关键词
    matched_tables = []
    all_tables = schema.get('tables', [])
    
    # 策略1: 精确关键词匹配
    for table in all_tables:
        table_name = table.get('table_name', '')
        table_name_lower = table_name.lower()
        
        # 提取表名的语义部分（移除区域前缀和数字后缀）
        # 例如："密云区-密云区特种设备使用登记-3396" -> "特种设备使用登记"
        table_parts = re.split(r'[-_\s]+', table_name_lower)
        semantic_parts = []
        for part in table_parts:
            # 跳过区域前缀（通常是重复的）和数字
            if not part.isdigit() and len(part) > 1:
                # 移除重复的区域前缀
                if len(semantic_parts) == 0 or part != semantic_parts[0]:
                    semantic_parts.append(part)
        
        # 检查语义部分是否在查询中出现
        for part in semantic_parts:
            if len(part) > 2 and part in query_lower:
                matched_tables.append(table_name)
                break
    
    # 策略2: 如果匹配的表太少，使用更宽松的匹配
    if len(matched_tables) < 3:
        # 提取查询中的关键词（中文词汇）
        query_keywords = re.findall(r'[\u4e00-\u9fa5]{2,}', query)
        
        for table in all_tables:
            table_name = table.get('table_name', '')
            if table_name in matched_tables:
                continue
            
            # 检查表名中是否包含查询关键词
            for keyword in query_keywords:
                if keyword in table_name:
                    matched_tables.append(table_name)
                    break
                    if len(matched_tables) >= BASELINE_CONFIG['max_tables']:
                        break
            if len(matched_tables) >= BASELINE_CONFIG['max_tables']:
                break
    
    # 如果匹配的表还是太少，返回更多表（增加覆盖率）
    if len(matched_tables) < 10:
        # 补充前N个表，确保有足够的表供选择
        tables = [t.get('table_name', '') for t in all_tables[:BASELINE_CONFIG['max_tables']]]
        # 合并，去重
        matched_tables = list(dict.fromkeys(matched_tables + tables))
    
    return matched_tables[:BASELINE_CONFIG['max_tables']]

def format_schema_for_baseline(
    schema: Dict, 
    relevant_tables: Optional[List[str]] = None,
    max_tables: Optional[int] = None,
    max_columns_per_table: Optional[int] = None
) -> Tuple[str, Dict]:
    """
    格式化Schema为Baseline Prompt
    
    返回：
    - schema_text: 格式化的Schema文本
    - config_info: 配置信息（用于记录）
    """
    strategy = BASELINE_CONFIG['strategy']
    max_tables = max_tables or BASELINE_CONFIG['max_tables']
    max_columns_per_table = max_columns_per_table or BASELINE_CONFIG['max_columns_per_table']
    
    all_tables = schema.get('tables', [])
    
    # 选择要包含的表
    if strategy == 'all_tables':
        selected_tables = all_tables
    elif strategy == 'relevant_tables' and relevant_tables:
        # 优先包含相关表
        relevant_table_set = set(relevant_tables)
        selected_tables = [t for t in all_tables if t.get('table_name', '') in relevant_table_set]
        
        # 如果相关表太少，补充一些表（确保有足够的表）
        if len(selected_tables) < max_tables:
            # 补充其他表，但优先保持相关表在前面
            remaining_tables = [t for t in all_tables if t.get('table_name', '') not in relevant_table_set]
            selected_tables = selected_tables + remaining_tables[:max_tables - len(selected_tables)]
    else:
        # first_n_tables
        selected_tables = all_tables[:max_tables]
    
    # 限制表数量
    selected_tables = selected_tables[:max_tables]
    
    # 格式化Schema文本
    text = "数据库Schema信息：\n\n"
    
    total_tables = len(selected_tables)
    total_columns = 0
    
    for table in selected_tables:
        table_name = table.get('table_name', '')
        columns = table.get('columns', [])
        
        # 限制每表的列数
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
        'strategy': strategy,
        'total_tables_in_schema': len(all_tables),
        'selected_tables_count': total_tables,
        'total_columns_included': total_columns,
        'max_tables': max_tables,
        'max_columns_per_table': max_columns_per_table,
        'schema_text_length': len(text),
        'estimated_tokens': len(text) // 4  # 粗略估算
    }
    
    return text, config_info

def generate_sql_with_baseline(
    client: OpenAI, 
    model_name: str, 
    query: str, 
    schema_text: str, 
    database: str,
    config_info: Dict
) -> Tuple[str, Dict]:
    """使用Baseline方法生成SQL"""
    model_config = MODEL_CONFIGS[model_name]
    
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

def evaluate_single_query_baseline(
    nl_query_file: str,
    db_path: str,
    schema_file: str,
    model_name: str,
    ground_truth_sql: str,
    ground_truth_results: List,
    gt_tables: Optional[List[str]] = None
) -> Dict:
    """使用Baseline方法评测单个query"""
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
    
    # 提取相关表（仅用于分析，实际评测时不使用GT信息）
    relevant_tables = extract_relevant_tables_from_nl_query(query, schema, gt_tables)
    
    # 格式化Schema
    schema_text, config_info = format_schema_for_baseline(schema, relevant_tables)
    
    # 生成SQL
    client = get_client(model_name)
    generated_sql, generation_info = generate_sql_with_baseline(
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
        'generation_info': generation_info,
        'gt_tables': gt_tables,
        'relevant_tables_used': relevant_tables[:10]  # 只记录前10个
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

def evaluate_database_baseline(
    nl_query_dir: str,
    db_path: str,
    schema_file: str,
    model_name: str,
    sql_dir: str,
    limit: Optional[int] = None
) -> Dict:
    """使用Baseline方法评测一个数据库的所有query"""
    results = []
    
    # 获取NL查询文件列表
    nl_files = [f for f in os.listdir(nl_query_dir) if f.startswith('generated_nl_query_') and f.endswith('.json')]
    nl_files.sort(key=lambda x: int(x.split('_')[-1].split('.')[0]) if x.split('_')[-1].split('.')[0].isdigit() else 0)
    
    if limit:
        nl_files = nl_files[:limit]
    
    for nl_file in tqdm(nl_files, desc=f"Baseline评测 {model_name}"):
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
        gt_tables = list(sql_data.get('tables', {}).keys())
        
        if not ground_truth_sql:
            continue
        
        # 评测（注意：这里使用gt_tables仅用于分析，实际评测时不应该使用）
        result = evaluate_single_query_baseline(
            nl_file_path,
            db_path,
            schema_file,
            model_name,
            ground_truth_sql,
            ground_truth_results,
            gt_tables=gt_tables  # 仅用于分析相关表提取效果
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
    
    # 统计配置信息
    config_stats = {
        'strategy': BASELINE_CONFIG['strategy'],
        'max_tables': BASELINE_CONFIG['max_tables'],
        'max_columns_per_table': BASELINE_CONFIG['max_columns_per_table'],
        'avg_schema_tokens': sum(r.get('generation_info', {}).get('estimated_tokens', 0) for r in results) / total if total > 0 else 0,
        'truncated_count': sum(1 for r in results if r.get('generation_info', {}).get('truncated', False))
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
        'baseline_config': BASELINE_CONFIG.copy(),
        'config_stats': config_stats,
        'results': results
    }

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Baseline评测：不使用TACO-SQL框架')
    parser.add_argument('--nl_query_dir', type=str, required=True, help='NL查询文件目录')
    parser.add_argument('--sql_dir', type=str, required=True, help='SQL文件目录')
    parser.add_argument('--db_path', type=str, required=True, help='数据库文件路径')
    parser.add_argument('--schema_file', type=str, required=True, help='Schema文件路径')
    parser.add_argument('--model', type=str, required=True, choices=['gpt-4', 'gpt-4o', 'gpt-o1', 'deepseek-r1'], help='模型名称')
    parser.add_argument('--output_file', type=str, required=True, help='输出结果文件')
    parser.add_argument('--strategy', type=str, default='relevant_tables', choices=['all_tables', 'relevant_tables', 'first_n_tables'], help='表选择策略')
    parser.add_argument('--max_tables', type=int, default=50, help='最大表数量')
    parser.add_argument('--max_columns_per_table', type=int, default=20, help='每个表最大列数')
    parser.add_argument('--limit', type=int, default=None, help='限制评测数量（用于测试）')
    
    args = parser.parse_args()
    
    # 更新配置
    BASELINE_CONFIG['strategy'] = args.strategy
    BASELINE_CONFIG['max_tables'] = args.max_tables
    BASELINE_CONFIG['max_columns_per_table'] = args.max_columns_per_table
    
    print(f"Baseline评测配置:")
    print(f"  模型: {args.model}")
    print(f"  策略: {args.strategy}")
    print(f"  最大表数: {args.max_tables}")
    print(f"  每表最大列数: {args.max_columns_per_table}")
    print(f"  上下文窗口: {MODEL_CONFIGS[args.model]['context_window']}")
    
    # 评测
    eval_result = evaluate_database_baseline(
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

