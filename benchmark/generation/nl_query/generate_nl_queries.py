"""
Improved NL query generation script.
- 4-step CoT reasoning
- Uses schema information (compact)
- Diverse examples
- Adapted to the new framework data structure
"""

import json
import os
import re
import yaml
import random
from tqdm import tqdm
from openai import OpenAI
import sqlparse
from typing import Dict, List, Any, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading

# Load template library
TEMPLATE_LIBRARY_PATH = os.path.join(os.path.dirname(__file__), 'template_library.json')
_template_library_cache = None

def load_template_library() -> List[Dict]:
    """Load template library."""
    global _template_library_cache
    if _template_library_cache is None:
        if os.path.exists(TEMPLATE_LIBRARY_PATH):
            with open(TEMPLATE_LIBRARY_PATH, 'r', encoding='utf-8') as f:
                _template_library_cache = json.load(f)
        else:
            print(f"Warning: template library file not found: {TEMPLATE_LIBRARY_PATH}")
            _template_library_cache = []
    return _template_library_cache

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
    # Default to gpt-3.5-turbo (can be overridden via CLI)
    model_name = config.get('llm', {}).get('model', 'gpt-3.5-turbo')
else:
    # Default configuration
    api_key = ""
    api_url = ""
    model_name = "gpt-3.5-turbo"

# Thread-local client storage (one client per thread)
thread_local = threading.local()

def get_client():
    """Get thread-local OpenAI client."""
    if not hasattr(thread_local, 'client'):
        thread_local.client = OpenAI(
            api_key=api_key,
            base_url=api_url
        )
    return thread_local.client

def generate_text(user_input, temperature=0.5, max_retries=3):
    """Call LLM to generate text."""
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

def extract_table_semantic(table_name: str) -> str:
    """Infer semantics from table name."""
    # Remove region prefix and numeric suffix
    # e.g. "密云区-密云区特种设备使用登记-3396" -> "特种设备使用登记" (region prefix + numeric suffix removed)
    parts = table_name.split('-')
    if len(parts) >= 2:
        # Take middle part
        semantic = parts[1] if len(parts) > 1 else parts[0]
        # Remove numeric suffix
        semantic = re.sub(r'-\d+$', '', semantic)
        return semantic
    return table_name

def extract_schema_info(sql_data: Dict, schema_file: str) -> Dict:
    """Extract compact schema information."""
    schema_info = {
        'tables': {},
        'involved_tables': []
    }
    
    # Get involved tables from SQL data
    involved_tables = list(sql_data.get('tables', {}).keys())
    schema_info['involved_tables'] = involved_tables
    
    # Load schema file
    if os.path.exists(schema_file):
        with open(schema_file, 'r', encoding='utf-8') as f:
            schema = json.load(f)
        
        # Extract only involved tables
        for table in schema.get('tables', []):
            table_name = table.get('table_name', '')
            if table_name in involved_tables:
                # Extract semantics
                semantic = extract_table_semantic(table_name)
                
                # Extract involved columns
                involved_columns = []
                if table_name in sql_data.get('tables', {}):
                    involved_columns = sql_data['tables'][table_name]
                
                # Build compact column info
                columns_info = []
                for col in table.get('columns', []):
                    col_name = col.get('column_name', '')
                    # Include only involved columns
                    full_col_name = f"{table_name}.{col_name}"
                    if any(full_col_name == inv_col or col_name in inv_col for inv_col in involved_columns):
                        columns_info.append({
                            'name': col_name,
                            'type': col.get('data_type', 'TEXT')
                        })
                
                schema_info['tables'][table_name] = {
                    'semantic': semantic,
                    'columns': columns_info[:10]  # max 10 columns to avoid excessive length
                }
    
    return schema_info

def analyze_sql_structure(sql: str) -> Dict:
    """Analyze SQL structure."""
    try:
        parsed = sqlparse.parse(sql)[0]
        
        analysis = {
            'has_join': 'JOIN' in sql.upper(),
            'has_subquery': False,
            'has_aggregate': any(func in sql.upper() for func in ['COUNT', 'SUM', 'AVG', 'MAX', 'MIN', 'GROUP BY']),
            'operation_type': 'SELECT',
            'conditions': []
        }
        
        # Check for subqueries
        if 'SELECT' in sql.upper():
            select_count = sql.upper().count('SELECT')
            if select_count > 1:
                analysis['has_subquery'] = True
        
        # Extract WHERE conditions (simplified)
        if 'WHERE' in sql.upper():
            where_part = sql.upper().split('WHERE')[1].split(';')[0]
            # Simple keyword extraction
            if 'AND' in where_part:
                analysis['conditions'].append('多条件AND')
            if 'OR' in where_part:
                analysis['conditions'].append('多条件OR')
            if 'LIKE' in where_part:
                analysis['conditions'].append('模糊匹配')
            if 'IS NOT NULL' in where_part or 'IS NULL' in where_part:
                analysis['conditions'].append('空值检查')
        
        return analysis
    except:
        return {
            'has_join': 'JOIN' in sql.upper(),
            'has_subquery': False,
            'has_aggregate': any(func in sql.upper() for func in ['COUNT', 'SUM', 'AVG', 'MAX', 'MIN']),
            'operation_type': 'SELECT',
            'conditions': []
        }

def format_schema_for_prompt(schema_info: Dict) -> str:
    """Format schema information for prompt."""
    text = "数据库Schema信息（仅包含SQL中涉及的表和列）：\n\n"
    
    for table_name, table_info in schema_info['tables'].items():
        semantic = table_info.get('semantic', table_name)
        text += f"表：{semantic}\n"
        text += f"  表名（技术名称）：{table_name}\n"
        text += "  相关列：\n"
        for col in table_info.get('columns', [])[:8]:  # max 8 columns
            text += f"    - {col['name']} ({col['type']})\n"
        text += "\n"
    
    return text

def count_tokens(text: str) -> int:
    """Count tokens (Chinese characters + English words)."""
    chinese_chars = len([c for c in text if '\u4e00' <= c <= '\u9fff'])
    english_words = len([w for w in text.split() if any(c.isalpha() for c in w)])
    return chinese_chars + english_words

def get_examples_from_template_library(sql_analysis: Dict, variant: int = 0) -> List[Dict]:
    """Select examples from template library (by SQL type and scenario)."""
    template_library = load_template_library()
    
    if not template_library:
        # If template library is empty, return empty list and use fallback
        return []
    
    # Filter candidate templates by SQL features
    candidates = []
    
    # Filter by SQL type
    for template in template_library:
        # Check SQL type match
        template_sql = template.get('sql', '').upper()
        is_join = 'JOIN' in template_sql or 'LEFT JOIN' in template_sql or 'RIGHT JOIN' in template_sql
        is_aggregate = any(func in template_sql for func in ['COUNT', 'SUM', 'AVG', 'MAX', 'MIN', 'GROUP BY'])
        
        # Match SQL type
        if sql_analysis.get('has_join') and is_join:
            candidates.append(template)
        elif sql_analysis.get('has_aggregate') and is_aggregate:
            candidates.append(template)
        elif not sql_analysis.get('has_join') and not sql_analysis.get('has_aggregate'):
            # Simple query: select non-JOIN, non-aggregate templates
            if not is_join and not is_aggregate:
                candidates.append(template)
    
    # If not enough candidates, use all templates
    if len(candidates) < 5:
        candidates = template_library.copy()
    
    # Select templates with different styles by variant (increase diversity)
    # variant=0: random selection
    # variant=1: different styles
    # variant=2: more diverse selection
    selected = []
    used_styles = set()
    
    # Shuffle order
    random.shuffle(candidates)
    
    # Adjust selection strategy by variant
    if variant == 0:
        # First pass: prefer templates with different styles
        for template in candidates:
            if len(selected) >= 3:
                break
            style = template.get('style', '')
            if style not in used_styles or len(selected) < 2:
                selected.append(template)
                used_styles.add(style)
    elif variant == 1:
        # Second pass: choose styles different from the first pass
        for template in candidates:
            if len(selected) >= 3:
                break
            style = template.get('style', '')
            if style not in used_styles:
                selected.append(template)
                used_styles.add(style)
    else:
        # Third pass and beyond: choose more diverse templates
        for template in candidates:
            if len(selected) >= 3:
                break
            style = template.get('style', '')
            scenario = template.get('scenario', '')
            # Prefer different style-scenario combinations
            key = f"{style}_{scenario}"
            if key not in used_styles or len(selected) < 2:
                selected.append(template)
                used_styles.add(key)
    
    # Fill remaining slots randomly if needed
    while len(selected) < 3 and len(selected) < len(candidates):
        remaining = [t for t in candidates if t not in selected]
        if remaining:
            selected.append(random.choice(remaining))
        else:
            break
    
    # Convert to example format
    examples = []
    for template in selected[:3]:
        examples.append({
            'query': template.get('query', ''),
            'style': template.get('style', ''),
            'scenario': template.get('scenario', ''),
            'tokens': template.get('tokens', 0)
        })
    
    return examples

def get_examples_by_type(sql_analysis: Dict, variant: int = 0) -> List[Dict]:
    """Select examples by SQL type (prefer template library, fallback to hardcoded examples)."""
    # Prefer template library
    template_examples = get_examples_from_template_library(sql_analysis, variant)
    
    if template_examples:
        return template_examples
    
    # Fallback: hardcoded examples (if template library unavailable)
    all_examples = [
        {
            'type': 'simple',
            'style': 'direct_query',
            'problem_description': '查询顺义区所有残疾人辅助器具站的名称和联系电话。',
            'natural_language_query': '查询顺义区所有残疾人辅助器具站的名称和联系电话。'
        },
        {
            'type': 'simple',
            'style': 'need_based',
            'problem_description': '需要获取所有单位食堂的企业名称和统一社会信用代码。',
            'natural_language_query': '需要获取所有单位食堂的企业名称和统一社会信用代码，用于后续的核查工作。'
        },
        {
            'type': 'simple',
            'style': 'problem_report',
            'problem_description': '查询顺义区杨镇汉石桥事地村的隐患名称，包括可能存在的卫生、安全等问题。',
            'natural_language_query': '反映顺义区杨镇汉石桥事地村在桥头有售卖小孩玩的烟花爆竹，存在安全隐患，希望尽快进行核实处理。'
        }
    ]
    
    candidate_examples = []
    if sql_analysis.get('has_join'):
        candidate_examples.extend([e for e in all_examples if e['type'] == 'join'])
    if sql_analysis.get('has_aggregate'):
        candidate_examples.extend([e for e in all_examples if e['type'] == 'aggregate'])
    if not candidate_examples:
        candidate_examples.extend([e for e in all_examples if e['type'] == 'simple'])
    
    return candidate_examples[:3]

def step1_sql_analysis(sql: str, schema_info: Dict, sql_analysis: Dict) -> str:
    """Step 1: SQL semantic analysis."""
    schema_text = format_schema_for_prompt(schema_info)
    
    prompt = f"""请分析以下SQL语句的查询意图和语义。

{schema_text}

SQL语句：
{sql}

请按照以下格式输出分析结果：
1. **查询操作类型**：SELECT/JOIN/聚合等
2. **涉及的业务实体**：从表名和列名推断涉及的业务实体（如：企业、设备、许可等）
3. **查询条件**：分析WHERE子句中的条件含义
4. **查询目的**：这个查询想要获取什么信息
5. **业务场景**：这个查询可能用于什么业务场景

请用自然语言详细描述，不要直接列出表名和列名，而是用业务术语描述。"""
    
    return generate_text(prompt, temperature=0.3)

def step2_business_scenario(sql_analysis_result: str, schema_info: Dict) -> str:
    """Step 2: Infer business scenario."""
    prompt = f"""基于以下SQL分析结果，推断可能的业务场景。

SQL分析结果：
{sql_analysis_result}

请推断：
1. **典型业务场景**：在什么情况下会需要这样的查询？
2. **用户角色**：可能是谁在使用这个查询？（如：市民、企业、政府部门等）
3. **使用场景**：用户可能遇到什么问题或需求？
4. **查询背景**：用户可能处于什么背景或情境下？

请用自然语言描述，要具体、真实。"""
    
    return generate_text(prompt, temperature=0.5)

def step3_user_scenario(business_scenario: str, sql_analysis: Dict) -> str:
    """Step 3: Build user scenario."""
    prompt = f"""基于以下业务场景，构建一个真实的用户场景描述。

业务场景：
{business_scenario}

请构建一个具体的用户场景，要求：
1. **具体化**：包含具体的用户身份、时间、地点等细节
2. **问题描述**：用户遇到的具体问题或需求，要相对明确
3. **背景信息**：可以包含一些背景信息，但不要过度冗余
4. **查询需求清晰**：明确表达用户想要查询什么数据、需要什么条件

请用自然语言描述，要像真实用户的口吻，但表达要相对清晰。"""
    
    return generate_text(prompt, temperature=0.5)

def step4_nl_generation(user_scenario: str, examples: List[Dict], variant: int = 0) -> str:
    """Step 4: Generate natural language query (using template library)."""
    example_text = ""
    
    # Check example format (template library vs legacy format)
    is_template_format = examples and 'query' in examples[0]
    
    if is_template_format:
        # Use template library format
        for i, example in enumerate(examples, 1):
            style_label = example.get('style', '未知风格')
            scenario_label = example.get('scenario', '')
            tokens = example.get('tokens', 0)
            query = example.get('query', '')
            
            example_text += f"### 示例 {i}（{style_label}风格"
            if scenario_label:
                example_text += f"，{scenario_label}场景"
            example_text += f"，约{tokens}个tokens）\n"
            example_text += f"自然语言查询：{query}\n\n"
    else:
        # Use legacy format (fallback)
        for i, example in enumerate(examples, 1):
            style_label = example.get('style', 'default')
            example_text += f"### 示例 {i}（{style_label}风格）\n"
            if 'problem_description' in example:
                example_text += f"问题描述：{example['problem_description']}\n"
            example_text += f"自然语言查询：{example.get('natural_language_query', example.get('query', ''))}\n\n"
    
    # Adjust prompt by variant to encourage different styles
    style_hint = ""
    if variant == 0:
        style_hint = "请参考示例的多样性，使用不同的开头方式和表述风格（如：市民反映、市民投诉、市民咨询、工单来源等），确保生成的查询风格多样。"
    elif variant == 1:
        style_hint = "这是第2个变体，请使用与第1个变体完全不同的开头方式和表述风格，参考示例中的不同风格，确保多样性。"
    else:
        style_hint = f"这是第{variant+1}个变体，请使用与前几个变体完全不同的开头方式和表述风格，确保最大程度的多样性。"
    
    # Length control notes (avg ~100 tokens, allow variance, emphasize diversity)
    length_hint = """**长度要求**：
- 目标平均长度：约100个tokens（中文字符数 + 英文单词数）
- 允许偏差：不同查询的长度可以有所不同（短至80 tokens，长至150 tokens都是合理的）
- **重要**：优先保证多样性和自然性，长度是次要考虑因素
- 不要为了达到100 tokens而刻意拉长或缩短查询
- 参考示例的长度分布，保持自然的长度变化"""
    
    prompt = f"""根据以下用户场景，生成一段自然语言查询。

要求：
1. **语言风格多样化**：参考示例的真实风格，可以使用不同的开头方式：
   - 市民反映："市民反映..."、"反映..."、"来电反映..."
   - 市民投诉："市民投诉..."、"投诉..."、"来电投诉..."
   - 市民咨询："市民咨询..."、"想了解..."、"咨询..."
   - 工单来源："工单来源..."、"来源..."等
   - 企业反映："企业反映..."等
   - 避免总是使用同一种开头，确保多样性
2. **查询意图明确**：用户知道要查询什么，用相对明确的表述
3. **信息冗余**：可以包含一些背景信息、时间、地点等细节，但不要过度冗余影响理解
4. **情感表达**：可以包含一些情感色彩（如不满、疑惑、希望等），但不要过度
5. **禁止技术术语**：绝对不要出现表名、列名等数据库技术术语
6. **查询重点突出**：明确表达要查询的内容、条件、筛选要求等
7. **真实性**：生成的查询应该像真实用户的口吻，参考示例的真实风格

{length_hint}

注意：
- 查询应该让模型能够理解要查询什么数据、需要什么条件、返回什么结果
- 保持自然语言风格，但不要过于模糊导致理解困难
- 可以包含一些冗余的背景信息，但核心查询意图要清晰
- {style_hint}
- **最重要的是多样性**：确保生成的查询在风格、长度、表述方式上都有所不同

{example_text}

用户场景：
{user_scenario}

请生成对应的自然语言查询（直接输出查询内容，不要添加任何解释或格式）："""
    
    return generate_text(prompt, temperature=0.7)

def process_single_sql(sql_file: str, schema_file: str, output_file: str, variant: int = 0) -> bool:
    """Process a single SQL file."""
    try:
        # Load SQL data
        with open(sql_file, 'r', encoding='utf-8') as f:
            sql_data = json.load(f)
        
        sql = sql_data.get('sql', '')
        sql_skeleton = sql_data.get('sql_skeleton', '')
        database = sql_data.get('database', '')
        tables = sql_data.get('tables', {})
        metadata = sql_data.get('metadata', {})
        
        if not sql:
            print(f"SQL file missing 'sql' field: {sql_file}")
            return False
        
        # Extract schema information
        schema_info = extract_schema_info(sql_data, schema_file)
        
        # Analyze SQL structure
        sql_analysis = analyze_sql_structure(sql)
        # Merge metadata
        if metadata:
            sql_analysis.update(metadata)
        
        # 4-step CoT generation
        # Step 1: SQL semantic analysis
        sql_analysis_result = step1_sql_analysis(sql, schema_info, sql_analysis)
        if not sql_analysis_result:
            print(f"Step 1 failed: {sql_file}")
            return False
        
        # Step 2: Business scenario inference
        business_scenario = step2_business_scenario(sql_analysis_result, schema_info)
        if not business_scenario:
            print(f"Step 2 failed: {sql_file}")
            return False
        
        # Step 3: User scenario construction
        user_scenario = step3_user_scenario(business_scenario, sql_analysis)
        if not user_scenario:
            print(f"Step 3 failed: {sql_file}")
            return False
        
        # Step 4: Natural language generation
        examples = get_examples_by_type(sql_analysis, variant)
        # If variant > 0, prompt for a different style in generation
        natural_language_query = step4_nl_generation(user_scenario, examples, variant)
        if not natural_language_query:
            print(f"Step 4 failed: {sql_file}")
            return False
        
        # Save result
        result = {
            'sql': sql,
            'sql_skeleton': sql_skeleton,
            'natural_language_query': natural_language_query,
            'database': database,
            'tables': tables,
            'metadata': metadata,
            'cot_steps': {
                'step1_sql_analysis': sql_analysis_result,
                'step2_business_scenario': business_scenario,
                'step3_user_scenario': user_scenario
            }
        }
        
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
        
        return True
        
    except Exception as e:
        print(f"Failed to process file {sql_file}: {e}")
        return False

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Generate NL queries (improved version)')
    parser.add_argument('--sql_dir', type=str, required=True, help='SQL file directory')
    parser.add_argument('--schema_dir', type=str, required=True, help='Schema file directory')
    parser.add_argument('--output_dir', type=str, required=True, help='Output directory')
    parser.add_argument('--database', type=str, default=None, help='Specify database (optional)')
    parser.add_argument('--limit', type=int, default=None, help='Limit processing count (for testing)')
    parser.add_argument('--max_workers', type=int, default=5, help='Concurrent worker threads (default: 5)')
    parser.add_argument('--log_dir', type=str, default=None, help='Log directory (default: benchmark/generation/nl_query/)')
    parser.add_argument('--model', type=str, default=None, help='Model to use (default: gpt-3.5-turbo)')
    
    args = parser.parse_args()
    
    # Override default model if specified
    if args.model:
        global model_name
        model_name = args.model
        print(f"Using model: {model_name}")
    
    # Iterate over databases
    databases = []
    if args.database:
        databases = [args.database]
    else:
        databases = [d for d in os.listdir(args.sql_dir) if os.path.isdir(os.path.join(args.sql_dir, d))]
    
    total_processed = 0
    total_success = 0
    
    for database in databases:
        print(f"\nProcessing database: {database}")
        sql_db_dir = os.path.join(args.sql_dir, database)
        schema_file = os.path.join(args.schema_dir, database, f"{database}.json")
        output_db_dir = os.path.join(args.output_dir, database)
        
        if not os.path.exists(sql_db_dir):
            print(f"SQL directory does not exist: {sql_db_dir}")
            continue
        
        if not os.path.exists(schema_file):
            print(f"Schema file does not exist: {schema_file}")
            continue
        
        # Get SQL file list
        sql_files = [f for f in os.listdir(sql_db_dir) if f.startswith('generated_sql_') and f.endswith('.json') and '_error' not in f]
        sql_files.sort(key=lambda x: int(re.findall(r'\d+', x)[0]) if re.findall(r'\d+', x) else 0)
        
        # Prepare task list
        tasks = []
        target_count = args.limit if args.limit else len(sql_files)
        
        # Compute how many NL query variants each SQL needs
        # If SQL count is below target, generate multiple variants per SQL
        if len(sql_files) < target_count:
            # Compute NL queries per SQL
            queries_per_sql = (target_count + len(sql_files) - 1) // len(sql_files)
            # Cap at 3 variants per SQL (avoid excessive duplication)
            queries_per_sql = min(queries_per_sql, 3)
            # Do not exceed target count
            actual_target = min(len(sql_files) * queries_per_sql, target_count)
            
            print(f"  Database has {len(sql_files)} SQLs, target {target_count} NL queries")
            print(f"  Generating {queries_per_sql} NL query variants per SQL, expected total {actual_target}")
            
            # Generate multiple NL query variants per SQL
            for sql_file in sql_files:
                sql_file_path = os.path.join(sql_db_dir, sql_file)
                base_idx = int(re.findall(r'\d+', sql_file)[0]) if re.findall(r'\d+', sql_file) else 0
                
                # Generate multiple NL query variants per SQL
                for variant in range(queries_per_sql):
                    if len(tasks) >= target_count:
                        break
                    
                    # First variant uses original filename; later variants use indexed filenames
                    if variant == 0:
                        output_file = os.path.join(output_db_dir, f'generated_nl_query_{base_idx}.json')
                    else:
                        # Use new index to avoid overwriting
                        new_idx = len(sql_files) * variant + base_idx
                        output_file = os.path.join(output_db_dir, f'generated_nl_query_{new_idx}.json')
                    
                    # Skip existing files (avoid regeneration)
                    if os.path.exists(output_file):
                        continue
                    
                    tasks.append((sql_file_path, schema_file, output_file, variant))
        else:
            # If enough SQLs, prioritize SQLs without NL queries
            print(f"  Database has {len(sql_files)} SQLs, target {target_count} NL queries")
            print(f"  Prioritizing SQLs without NL queries")
            
            # Find SQL files missing NL queries
            missing_nl_sqls = []
            for sql_file in sql_files:
                sql_file_path = os.path.join(sql_db_dir, sql_file)
                file_idx = int(re.findall(r'\d+', sql_file)[0]) if re.findall(r'\d+', sql_file) else -1
                if file_idx < 0:
                    continue
                
                output_file = os.path.join(output_db_dir, f'generated_nl_query_{file_idx}.json')
                
                # Add to task list if NL query file is missing
                if not os.path.exists(output_file):
                    missing_nl_sqls.append((sql_file_path, file_idx, output_file))
            
            print(f"  Found {len(missing_nl_sqls)} SQLs without NL queries")
            
            # Process SQLs missing NL queries first
            for sql_file_path, file_idx, output_file in missing_nl_sqls[:target_count]:
                tasks.append((sql_file_path, schema_file, output_file, 0))
            
            # If still below target, generate variants for SQLs that already have NL queries
            if len(tasks) < target_count:
                remaining = target_count - len(tasks)
                print(f"  Need {remaining} more; generating variants for SQLs with existing NL queries")
                
                # Find SQLs that already have NL queries
                existing_nl_sqls = []
                for sql_file in sql_files:
                    sql_file_path = os.path.join(sql_db_dir, sql_file)
                    file_idx = int(re.findall(r'\d+', sql_file)[0]) if re.findall(r'\d+', sql_file) else -1
                    if file_idx < 0:
                        continue
                    
                    output_file = os.path.join(output_db_dir, f'generated_nl_query_{file_idx}.json')
                    
                    # If NL query exists, can generate variant
                    if os.path.exists(output_file):
                        # Generate variant with new index
                        variant_idx = len(sql_files) + file_idx
                        variant_output_file = os.path.join(output_db_dir, f'generated_nl_query_{variant_idx}.json')
                        if not os.path.exists(variant_output_file):
                            existing_nl_sqls.append((sql_file_path, variant_output_file, 1))
                
                # Add variant tasks
                for sql_file_path, variant_output_file, variant in existing_nl_sqls[:remaining]:
                    tasks.append((sql_file_path, schema_file, variant_output_file, variant))
            
            print(f"  Will process {len(tasks)} tasks total")
        
        # Process concurrently
        if tasks:
            with ThreadPoolExecutor(max_workers=args.max_workers) as executor:
                futures = {executor.submit(process_single_sql, sql_path, schema_file, out_file, variant): (sql_path, out_file) 
                          for sql_path, _, out_file, variant in tasks}
                
                for future in tqdm(as_completed(futures), total=len(futures), desc=f"Processing {database}"):
                    sql_path, out_file = futures[future]
                    total_processed += 1
                    try:
                        if future.result():
                            total_success += 1
                    except Exception as e:
                        print(f"Processing failed {sql_path}: {e}")
    
    print(f"\nDone! Processed: {total_processed}, Succeeded: {total_success}")

if __name__ == '__main__':
    main()

