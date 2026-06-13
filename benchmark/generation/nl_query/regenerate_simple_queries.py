"""
Regenerate NL queries for simple SQL (using a simpler prompt)
- Select 50 simple SQL statements from each database
- Generate using gpt-3.5-turbo (cost savings)
- Use a simplified prompt to reduce difficulty
- Overwrite the original NL query files
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

# Load configuration
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

# Use gpt-3.5-turbo to generate NL queries (cost savings)
model_name = "gpt-3.5-turbo"

# Thread-local client storage
thread_local = threading.local()

def get_client():
    """Get thread-local OpenAI client"""
    if not hasattr(thread_local, 'client'):
        thread_local.client = OpenAI(
            api_key=api_key,
            base_url=api_url
        )
    return thread_local.client

def generate_text(user_input, temperature=0.5, max_retries=3):
    """Call LLM to generate text"""
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
                print(f"API call failed, retrying... ({attempt + 1}/{max_retries}): {e}")
                continue
            else:
                print(f"API call failed: {e}")
                return ""

def is_simple_sql(sql: str, metadata: Dict) -> bool:
    """Check whether SQL is simple (no JOIN, no subquery)"""
    sql_upper = sql.upper()
    
    # Check for JOIN
    has_join = 'JOIN' in sql_upper or metadata.get('has_join', False)
    
    # Check for subqueries (multiple SELECT, excluding UNION cases)
    # Simple heuristic: if SELECT appears multiple times and it's not UNION, it may be a subquery
    select_count = sql_upper.count('SELECT')
    # More precise: check for SELECT inside parentheses
    has_subquery = False
    if select_count > 1:
        # Check for nested SELECT (inside parentheses)
        import re
        # Find SELECT inside parentheses
        paren_pattern = r'\([^)]*SELECT[^)]*\)'
        if re.search(paren_pattern, sql_upper):
            has_subquery = True
    
    # Simple SQL: no JOIN, no subquery
    return not has_join and not has_subquery

def select_simple_sqls(sql_dir: str, limit: int = 50) -> List[Dict]:
    """Select simple SQL statements"""
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
            print(f"Failed to read SQL file {sql_file}: {e}")
            continue
    
    return simple_sqls

def generate_simple_nl_query(sql: str, sql_data: Dict, schema_info: Dict) -> str:
    """Generate a simple NL query (using a simplified prompt)"""
    
    # Extract table and column names (for understanding query intent)
    tables = sql_data.get('tables', {})
    table_names = list(tables.keys())
    
    # Extract column names from SQL
    columns = []
    for table_name, table_cols in tables.items():
        for col in table_cols:
            if '.' in col:
                col_name = col.split('.')[-1]
                columns.append(col_name)
    
    # Simplified prompt: direct, clear, unambiguous
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
    
    # Clean up generated query
    if nl_query:
        # Remove possible quotes
        nl_query = nl_query.strip().strip('"').strip("'")
        # Remove possible "NL query:" prefix (Chinese: "NL查询：" or "自然语言查询：")
        if nl_query.startswith('NL查询：') or nl_query.startswith('自然语言查询：'):
            nl_query = nl_query.split('：', 1)[-1].strip()
    
    return nl_query

def process_single_sql(sql_info: Dict, sql_dir: str, schema_file: str, output_dir: str, database: str) -> Dict:
    """Process a single SQL statement and generate an NL query"""
    try:
        sql_data = sql_info['data']
        sql = sql_info['sql']
        file_idx = sql_info['idx']
        
        # Extract schema info (compact)
        schema_info = {
            'tables': sql_data.get('tables', {}),
            'involved_tables': list(sql_data.get('tables', {}).keys())
        }
        
        # Generate simple NL query
        nl_query = generate_simple_nl_query(sql, sql_data, schema_info)
        
        if not nl_query:
            return {
                'success': False,
                'idx': file_idx,
                'error': 'Failed to generate NL query'
            }
        
        # Build output data
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
        
        # Save to file (overwrite original file)
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
    parser = argparse.ArgumentParser(description='Regenerate NL queries for simple SQL')
    parser.add_argument('--sql_dir', type=str, required=True, help='SQL file directory')
    parser.add_argument('--schema_dir', type=str, required=True, help='Schema file directory')
    parser.add_argument('--output_dir', type=str, required=True, help='Output directory')
    parser.add_argument('--database', type=str, required=True, help='Database name')
    parser.add_argument('--limit', type=int, default=50, help='Number of NL queries to generate (default: 50)')
    parser.add_argument('--max_workers', type=int, default=5, help='Number of concurrent worker threads (default: 5)')
    
    args = parser.parse_args()
    
    print(f"Starting NL query regeneration for simple SQL in database '{args.database}'")
    print(f"Model: {model_name}")
    print(f"Target count: {args.limit}")
    print(f"Concurrent workers: {args.max_workers}")
    print("="*80)
    
    # Select simple SQL statements
    print(f"Selecting simple SQL statements...")
    simple_sqls = select_simple_sqls(args.sql_dir, args.limit)
    
    if len(simple_sqls) < args.limit:
        print(f"Warning: found only {len(simple_sqls)} simple SQL statements, fewer than target {args.limit}")
    
    print(f"Found {len(simple_sqls)} simple SQL statements")
    print("="*80)
    
    # Get schema file path
    schema_file = os.path.join(args.schema_dir, args.database, f'{args.database}.json')
    
    if not os.path.exists(schema_file):
        print(f"Warning: schema file does not exist: {schema_file}")
    
    # Ensure output directory exists
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Process concurrently
    results = []
    with ThreadPoolExecutor(max_workers=args.max_workers) as executor:
        futures = {
            executor.submit(process_single_sql, sql_info, args.sql_dir, schema_file, args.output_dir, args.database): sql_info
            for sql_info in simple_sqls
        }
        
        with tqdm(total=len(simple_sqls), desc=f"Generating NL queries ({model_name})") as pbar:
            for future in as_completed(futures):
                result = future.result()
                results.append(result)
                pbar.update(1)
                
                if result['success']:
                    pbar.set_postfix({'success': sum(1 for r in results if r['success'])})
                else:
                    pbar.set_postfix({'failed': sum(1 for r in results if not r['success'])})
    
    # Summarize results
    success_count = sum(1 for r in results if r['success'])
    fail_count = len(results) - success_count
    
    print("="*80)
    print(f"Generation complete:")
    print(f"  Success: {success_count}/{len(results)}")
    print(f"  Failed: {fail_count}/{len(results)}")
    print(f"  Output directory: {args.output_dir}")
    print("="*80)

if __name__ == '__main__':
    main()
