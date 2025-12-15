"""
重新生成简单SQL的NL查询（使用更简单的prompt）
- 从每个数据库中选择50条简单的SQL
- 使用gpt-3.5-turbo生成（节省费用）
- 使用简化的prompt降低难度
- 覆盖原来的NL查询文件
"""

import json
import os
import re
import yaml
import argparse
from tqdm import tqdm
from openai import OpenAI
import sqlparse
from typing import Dict, List, Any, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading

# 加载配置
def load_config():
    config_path = os.path.join(os.path.dirname(__file__), '..', 'sql_filling', 'config.yaml')
    if os.path.exists(config_path):
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        return config
    return None

config = load_config()
if config:
    api_key = config['llm']['api_key']
    api_url = config['llm']['api_url']
else:
    api_key = ""
    api_url = ""

# 使用gpt-3.5-turbo生成NL查询（节省费用）
model_name = "gpt-3.5-turbo"

# 线程本地存储客户端
thread_local = threading.local()

def get_client():
    """获取线程本地的OpenAI客户端"""
    if not hasattr(thread_local, 'client'):
        thread_local.client = OpenAI(
            api_key=api_key,
            base_url=api_url
        )
    return thread_local.client

def generate_text(user_input, temperature=0.5, max_retries=3):
    """调用LLM生成文本"""
    client = get_client()
    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model=model_name,
                temperature=temperature,
                messages=[
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": user_input},
                ],
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            if attempt < max_retries - 1:
                print(f"API调用失败，重试中... ({attempt + 1}/{max_retries}): {e}")
                continue
            else:
                print(f"API调用失败: {e}")
                return ""

def is_simple_sql(sql: str, metadata: Dict) -> bool:
    """判断SQL是否简单（没有JOIN、没有子查询）"""
    sql_upper = sql.upper()
    
    # 检查是否有JOIN
    has_join = 'JOIN' in sql_upper or metadata.get('has_join', False)
    
    # 检查是否有子查询（多个SELECT，排除UNION的情况）
    # 简单判断：如果SELECT出现多次，且不是UNION，可能是子查询
    select_count = sql_upper.count('SELECT')
    # 更精确：检查是否有括号内的SELECT
    has_subquery = False
    if select_count > 1:
        # 检查是否有嵌套的SELECT（在括号内）
        import re
        # 查找括号内的SELECT
        paren_pattern = r'\([^)]*SELECT[^)]*\)'
        if re.search(paren_pattern, sql_upper):
            has_subquery = True
    
    # 简单SQL：没有JOIN、没有子查询
    return not has_join and not has_subquery

def select_simple_sqls(sql_dir: str, limit: int = 50) -> List[Dict]:
    """选择简单的SQL"""
    sql_files = sorted([f for f in os.listdir(sql_dir) if f.startswith('generated_sql_') and f.endswith('.json') and '_error' not in f],
                       key=lambda x: int(x.split('_')[-1].split('.')[0]) if x.split('_')[-1].split('.')[0].isdigit() else 0)
    
    simple_sqls = []
    
    for sql_file in sql_files:
        sql_path = os.path.join(sql_dir, sql_file)
        try:
            with open(sql_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            sql = data.get('sql', '')
            metadata = data.get('metadata', {})
            
            if is_simple_sql(sql, metadata):
                file_idx = int(sql_file.split('_')[-1].split('.')[0])
                simple_sqls.append({
                    'file': sql_file,
                    'idx': file_idx,
                    'sql': sql,
                    'data': data
                })
            
            if len(simple_sqls) >= limit:
                break
        except Exception as e:
            print(f"读取SQL文件失败 {sql_file}: {e}")
            continue
    
    return simple_sqls

def generate_simple_nl_query(sql: str, sql_data: Dict, schema_info: Dict) -> str:
    """生成简单的NL查询（使用简化的prompt）"""
    
    # 提取表名和列名（用于理解查询意图）
    tables = sql_data.get('tables', {})
    table_names = list(tables.keys())
    
    # 从SQL中提取列名
    columns = []
    for table_name, table_cols in tables.items():
        for col in table_cols:
            if '.' in col:
                col_name = col.split('.')[-1]
                columns.append(col_name)
    
    # 简化的prompt：直接、清晰、不模糊
    prompt = f"""根据以下SQL查询，生成一个简单、直接的自然语言查询。

SQL查询：
{sql}

要求：
1. **查询意图清晰**：直接表达要查询什么，不要模糊或冗余
2. **语言简洁**：用简单明了的语言，避免不必要的背景信息
3. **重点突出**：明确说明要查询的数据和条件
4. **不要包含表名和列名**：用业务术语描述，不要直接提到技术术语
5. **保持自然**：像真实用户提问，但表达要清晰

示例：
- SQL: SELECT "企业名称", "统一社会信用代码" FROM "表名" WHERE "状态" = '正常'
- NL查询: 查询所有状态为正常的企业名称和统一社会信用代码

请生成自然语言查询："""
    
    nl_query = generate_text(prompt, temperature=0.3)
    
    # 清理生成的查询
    if nl_query:
        # 移除可能的引号
        nl_query = nl_query.strip().strip('"').strip("'")
        # 移除可能的"NL查询："前缀
        if nl_query.startswith('NL查询：') or nl_query.startswith('自然语言查询：'):
            nl_query = nl_query.split('：', 1)[-1].strip()
    
    return nl_query

def process_single_sql(sql_info: Dict, sql_dir: str, schema_file: str, output_dir: str, database: str) -> Dict:
    """处理单个SQL，生成NL查询"""
    try:
        sql_data = sql_info['data']
        sql = sql_info['sql']
        file_idx = sql_info['idx']
        
        # 提取Schema信息（精简）
        schema_info = {
            'tables': sql_data.get('tables', {}),
            'involved_tables': list(sql_data.get('tables', {}).keys())
        }
        
        # 生成简单的NL查询
        nl_query = generate_simple_nl_query(sql, sql_data, schema_info)
        
        if not nl_query:
            return {
                'success': False,
                'idx': file_idx,
                'error': 'Failed to generate NL query'
            }
        
        # 构建输出数据
        output_data = {
            'sql': sql,
            'sql_skeleton': sql_data.get('sql_skeleton', ''),
            'natural_language_query': nl_query,
            'database': database,
            'tables': sql_data.get('tables', {}),
            'metadata': sql_data.get('metadata', {}),
            'is_simple': True,
            'regenerated': True
        }
        
        # 保存到文件（覆盖原来的文件）
        output_file = os.path.join(output_dir, f'generated_nl_query_{file_idx}.json')
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, ensure_ascii=False, indent=2)
        
        return {
            'success': True,
            'idx': file_idx,
            'nl_query': nl_query[:100]
        }
    
    except Exception as e:
        return {
            'success': False,
            'idx': sql_info.get('idx', -1),
            'error': str(e)
        }

def main():
    parser = argparse.ArgumentParser(description='重新生成简单SQL的NL查询')
    parser.add_argument('--sql_dir', type=str, required=True, help='SQL文件目录')
    parser.add_argument('--schema_dir', type=str, required=True, help='Schema文件目录')
    parser.add_argument('--output_dir', type=str, required=True, help='输出目录')
    parser.add_argument('--database', type=str, required=True, help='数据库名称')
    parser.add_argument('--limit', type=int, default=50, help='要生成的NL查询数量（默认50）')
    parser.add_argument('--max_workers', type=int, default=5, help='并发线程数（默认5）')
    
    args = parser.parse_args()
    
    print(f"开始为数据库 '{args.database}' 重新生成简单SQL的NL查询")
    print(f"使用模型: {model_name}")
    print(f"目标数量: {args.limit}")
    print(f"并发线程数: {args.max_workers}")
    print("="*80)
    
    # 选择简单的SQL
    print(f"正在选择简单的SQL...")
    simple_sqls = select_simple_sqls(args.sql_dir, args.limit)
    
    if len(simple_sqls) < args.limit:
        print(f"警告：只找到 {len(simple_sqls)} 条简单SQL，少于目标数量 {args.limit}")
    
    print(f"找到 {len(simple_sqls)} 条简单SQL")
    print("="*80)
    
    # 获取Schema文件路径
    schema_file = os.path.join(args.schema_dir, args.database, f'{args.database}.json')
    
    if not os.path.exists(schema_file):
        print(f"警告：Schema文件不存在: {schema_file}")
    
    # 确保输出目录存在
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 并发处理
    results = []
    with ThreadPoolExecutor(max_workers=args.max_workers) as executor:
        futures = {
            executor.submit(process_single_sql, sql_info, args.sql_dir, schema_file, args.output_dir, args.database): sql_info
            for sql_info in simple_sqls
        }
        
        with tqdm(total=len(simple_sqls), desc=f"生成NL查询 ({model_name})") as pbar:
            for future in as_completed(futures):
                result = future.result()
                results.append(result)
                pbar.update(1)
                
                if result['success']:
                    pbar.set_postfix({'成功': sum(1 for r in results if r['success'])})
                else:
                    pbar.set_postfix({'失败': sum(1 for r in results if not r['success'])})
    
    # 统计结果
    success_count = sum(1 for r in results if r['success'])
    fail_count = len(results) - success_count
    
    print("="*80)
    print(f"生成完成:")
    print(f"  成功: {success_count}/{len(results)}")
    print(f"  失败: {fail_count}/{len(results)}")
    print(f"  输出目录: {args.output_dir}")
    print("="*80)

if __name__ == '__main__':
    main()

