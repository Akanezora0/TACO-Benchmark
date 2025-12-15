"""
改进的NL查询生成脚本
- 4步CoT推理
- 利用Schema信息（精简）
- 多样化示例
- 适配新框架的数据结构
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

# 加载模板库
TEMPLATE_LIBRARY_PATH = os.path.join(os.path.dirname(__file__), 'template_library.json')
_template_library_cache = None

def load_template_library() -> List[Dict]:
    """加载模板库"""
    global _template_library_cache
    if _template_library_cache is None:
        if os.path.exists(TEMPLATE_LIBRARY_PATH):
            with open(TEMPLATE_LIBRARY_PATH, 'r', encoding='utf-8') as f:
                _template_library_cache = json.load(f)
        else:
            print(f"警告：模板库文件不存在: {TEMPLATE_LIBRARY_PATH}")
            _template_library_cache = []
    return _template_library_cache

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
    # 默认使用gpt-3.5-turbo（可以通过命令行参数覆盖）
    model_name = config.get('llm', {}).get('model', 'gpt-3.5-turbo')
else:
    # 默认配置
    api_key = ""
    api_url = ""
    model_name = "gpt-3.5-turbo"

# 线程本地存储客户端（每个线程一个客户端）
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

def extract_table_semantic(table_name: str) -> str:
    """从表名推断语义"""
    # 移除区域前缀和数字后缀
    # 例如："密云区-密云区特种设备使用登记-3396" -> "特种设备使用登记"
    parts = table_name.split('-')
    if len(parts) >= 2:
        # 取中间部分
        semantic = parts[1] if len(parts) > 1 else parts[0]
        # 移除数字后缀
        semantic = re.sub(r'-\d+$', '', semantic)
        return semantic
    return table_name

def extract_schema_info(sql_data: Dict, schema_file: str) -> Dict:
    """提取精简的Schema信息"""
    schema_info = {
        'tables': {},
        'involved_tables': []
    }
    
    # 从SQL数据中获取涉及的表
    involved_tables = list(sql_data.get('tables', {}).keys())
    schema_info['involved_tables'] = involved_tables
    
    # 加载Schema文件
    if os.path.exists(schema_file):
        with open(schema_file, 'r', encoding='utf-8') as f:
            schema = json.load(f)
        
        # 只提取涉及的表的信息
        for table in schema.get('tables', []):
            table_name = table.get('table_name', '')
            if table_name in involved_tables:
                # 提取语义
                semantic = extract_table_semantic(table_name)
                
                # 提取涉及的列
                involved_columns = []
                if table_name in sql_data.get('tables', {}):
                    involved_columns = sql_data['tables'][table_name]
                
                # 构建精简的列信息
                columns_info = []
                for col in table.get('columns', []):
                    col_name = col.get('column_name', '')
                    # 只包含涉及的列
                    full_col_name = f"{table_name}.{col_name}"
                    if any(full_col_name == inv_col or col_name in inv_col for inv_col in involved_columns):
                        columns_info.append({
                            'name': col_name,
                            'type': col.get('data_type', 'TEXT')
                        })
                
                schema_info['tables'][table_name] = {
                    'semantic': semantic,
                    'columns': columns_info[:10]  # 最多10个列，避免过长
                }
    
    return schema_info

def analyze_sql_structure(sql: str) -> Dict:
    """分析SQL结构"""
    try:
        parsed = sqlparse.parse(sql)[0]
        
        analysis = {
            'has_join': 'JOIN' in sql.upper(),
            'has_subquery': False,
            'has_aggregate': any(func in sql.upper() for func in ['COUNT', 'SUM', 'AVG', 'MAX', 'MIN', 'GROUP BY']),
            'operation_type': 'SELECT',
            'conditions': []
        }
        
        # 检查子查询
        if 'SELECT' in sql.upper():
            select_count = sql.upper().count('SELECT')
            if select_count > 1:
                analysis['has_subquery'] = True
        
        # 提取WHERE条件（简化）
        if 'WHERE' in sql.upper():
            where_part = sql.upper().split('WHERE')[1].split(';')[0]
            # 简单提取条件关键词
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
    """格式化Schema信息为Prompt"""
    text = "数据库Schema信息（仅包含SQL中涉及的表和列）：\n\n"
    
    for table_name, table_info in schema_info['tables'].items():
        semantic = table_info.get('semantic', table_name)
        text += f"表：{semantic}\n"
        text += f"  表名（技术名称）：{table_name}\n"
        text += "  相关列：\n"
        for col in table_info.get('columns', [])[:8]:  # 最多8个列
            text += f"    - {col['name']} ({col['type']})\n"
        text += "\n"
    
    return text

def count_tokens(text: str) -> int:
    """计算token数（中文字符数 + 英文单词数）"""
    chinese_chars = len([c for c in text if '\u4e00' <= c <= '\u9fff'])
    english_words = len([w for w in text.split() if any(c.isalpha() for c in w)])
    return chinese_chars + english_words

def get_examples_from_template_library(sql_analysis: Dict, variant: int = 0) -> List[Dict]:
    """从模板库中选择示例（根据SQL类型和场景）"""
    template_library = load_template_library()
    
    if not template_library:
        # 如果模板库为空，返回空列表，使用fallback
        return []
    
    # 根据SQL特征筛选候选模板
    candidates = []
    
    # 根据SQL类型筛选
    for template in template_library:
        # 检查SQL类型匹配
        template_sql = template.get('sql', '').upper()
        is_join = 'JOIN' in template_sql or 'LEFT JOIN' in template_sql or 'RIGHT JOIN' in template_sql
        is_aggregate = any(func in template_sql for func in ['COUNT', 'SUM', 'AVG', 'MAX', 'MIN', 'GROUP BY'])
        
        # 匹配SQL类型
        if sql_analysis.get('has_join') and is_join:
            candidates.append(template)
        elif sql_analysis.get('has_aggregate') and is_aggregate:
            candidates.append(template)
        elif not sql_analysis.get('has_join') and not sql_analysis.get('has_aggregate'):
            # 简单查询，选择非JOIN和非聚合的模板
            if not is_join and not is_aggregate:
                candidates.append(template)
    
    # 如果候选不足，使用所有模板
    if len(candidates) < 5:
        candidates = template_library.copy()
    
    # 根据variant选择不同风格的模板（增加多样性）
    # variant=0: 随机选择
    # variant=1: 选择不同风格的
    # variant=2: 选择更多样化的
    selected = []
    used_styles = set()
    
    # 打乱顺序
    random.shuffle(candidates)
    
    # 根据variant调整选择策略
    if variant == 0:
        # 第一次：优先选择不同风格的模板
        for template in candidates:
            if len(selected) >= 3:
                break
            style = template.get('style', '')
            if style not in used_styles or len(selected) < 2:
                selected.append(template)
                used_styles.add(style)
    elif variant == 1:
        # 第二次：选择与第一次不同风格的模板
        for template in candidates:
            if len(selected) >= 3:
                break
            style = template.get('style', '')
            if style not in used_styles:
                selected.append(template)
                used_styles.add(style)
    else:
        # 第三次及以后：选择更多样化的模板
        for template in candidates:
            if len(selected) >= 3:
                break
            style = template.get('style', '')
            scenario = template.get('scenario', '')
            # 优先选择不同风格和场景的组合
            key = f"{style}_{scenario}"
            if key not in used_styles or len(selected) < 2:
                selected.append(template)
                used_styles.add(key)
    
    # 如果还不够，随机补充
    while len(selected) < 3 and len(selected) < len(candidates):
        remaining = [t for t in candidates if t not in selected]
        if remaining:
            selected.append(random.choice(remaining))
        else:
            break
    
    # 转换为示例格式
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
    """根据SQL类型选择示例（优先使用模板库，fallback到硬编码示例）"""
    # 优先从模板库中选择
    template_examples = get_examples_from_template_library(sql_analysis, variant)
    
    if template_examples:
        return template_examples
    
    # Fallback：使用硬编码示例（如果模板库不可用）
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
    """Step 1: SQL语义分析"""
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
    """Step 2: 业务场景推断"""
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
    """Step 3: 用户场景构建"""
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
    """Step 4: 自然语言查询生成（使用模板库）"""
    example_text = ""
    
    # 检查示例格式（模板库格式 vs 旧格式）
    is_template_format = examples and 'query' in examples[0]
    
    if is_template_format:
        # 使用模板库格式
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
        # 使用旧格式（fallback）
        for i, example in enumerate(examples, 1):
            style_label = example.get('style', 'default')
            example_text += f"### 示例 {i}（{style_label}风格）\n"
            if 'problem_description' in example:
                example_text += f"问题描述：{example['problem_description']}\n"
            example_text += f"自然语言查询：{example.get('natural_language_query', example.get('query', ''))}\n\n"
    
    # 根据variant调整提示，鼓励不同风格
    style_hint = ""
    if variant == 0:
        style_hint = "请参考示例的多样性，使用不同的开头方式和表述风格（如：市民反映、市民投诉、市民咨询、工单来源等），确保生成的查询风格多样。"
    elif variant == 1:
        style_hint = "这是第2个变体，请使用与第1个变体完全不同的开头方式和表述风格，参考示例中的不同风格，确保多样性。"
    else:
        style_hint = f"这是第{variant+1}个变体，请使用与前几个变体完全不同的开头方式和表述风格，确保最大程度的多样性。"
    
    # 长度控制说明（平均100 tokens，但允许偏差，强调多样性）
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
    """处理单个SQL文件"""
    try:
        # 加载SQL数据
        with open(sql_file, 'r', encoding='utf-8') as f:
            sql_data = json.load(f)
        
        sql = sql_data.get('sql', '')
        sql_skeleton = sql_data.get('sql_skeleton', '')
        database = sql_data.get('database', '')
        tables = sql_data.get('tables', {})
        metadata = sql_data.get('metadata', {})
        
        if not sql:
            print(f"SQL文件缺少sql字段: {sql_file}")
            return False
        
        # 提取Schema信息
        schema_info = extract_schema_info(sql_data, schema_file)
        
        # 分析SQL结构
        sql_analysis = analyze_sql_structure(sql)
        # 合并metadata中的信息
        if metadata:
            sql_analysis.update(metadata)
        
        # 4步CoT生成
        # Step 1: SQL语义分析
        sql_analysis_result = step1_sql_analysis(sql, schema_info, sql_analysis)
        if not sql_analysis_result:
            print(f"Step 1失败: {sql_file}")
            return False
        
        # Step 2: 业务场景推断
        business_scenario = step2_business_scenario(sql_analysis_result, schema_info)
        if not business_scenario:
            print(f"Step 2失败: {sql_file}")
            return False
        
        # Step 3: 用户场景构建
        user_scenario = step3_user_scenario(business_scenario, sql_analysis)
        if not user_scenario:
            print(f"Step 3失败: {sql_file}")
            return False
        
        # Step 4: 自然语言生成
        examples = get_examples_by_type(sql_analysis, variant)
        # 如果variant > 0，在prompt中提示生成不同风格的查询
        natural_language_query = step4_nl_generation(user_scenario, examples, variant)
        if not natural_language_query:
            print(f"Step 4失败: {sql_file}")
            return False
        
        # 保存结果
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
        print(f"处理文件失败 {sql_file}: {e}")
        return False

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='生成NL查询（改进版）')
    parser.add_argument('--sql_dir', type=str, required=True, help='SQL文件目录')
    parser.add_argument('--schema_dir', type=str, required=True, help='Schema文件目录')
    parser.add_argument('--output_dir', type=str, required=True, help='输出目录')
    parser.add_argument('--database', type=str, default=None, help='指定数据库（可选）')
    parser.add_argument('--limit', type=int, default=None, help='限制处理数量（用于测试）')
    parser.add_argument('--max_workers', type=int, default=5, help='并发线程数（默认5）')
    parser.add_argument('--log_dir', type=str, default=None, help='日志目录（默认：benchmark/generation/nl_query/）')
    parser.add_argument('--model', type=str, default=None, help='使用的模型（默认：gpt-3.5-turbo）')
    
    args = parser.parse_args()
    
    # 如果指定了模型，覆盖默认值
    if args.model:
        global model_name
        model_name = args.model
        print(f"使用模型: {model_name}")
    
    # 遍历数据库
    databases = []
    if args.database:
        databases = [args.database]
    else:
        databases = [d for d in os.listdir(args.sql_dir) if os.path.isdir(os.path.join(args.sql_dir, d))]
    
    total_processed = 0
    total_success = 0
    
    for database in databases:
        print(f"\n处理数据库: {database}")
        sql_db_dir = os.path.join(args.sql_dir, database)
        schema_file = os.path.join(args.schema_dir, database, f"{database}.json")
        output_db_dir = os.path.join(args.output_dir, database)
        
        if not os.path.exists(sql_db_dir):
            print(f"SQL目录不存在: {sql_db_dir}")
            continue
        
        if not os.path.exists(schema_file):
            print(f"Schema文件不存在: {schema_file}")
            continue
        
        # 获取SQL文件列表
        sql_files = [f for f in os.listdir(sql_db_dir) if f.startswith('generated_sql_') and f.endswith('.json') and '_error' not in f]
        sql_files.sort(key=lambda x: int(re.findall(r'\d+', x)[0]) if re.findall(r'\d+', x) else 0)
        
        # 准备待处理文件列表
        tasks = []
        target_count = args.limit if args.limit else len(sql_files)
        
        # 计算每个SQL需要生成多少个NL查询变体
        # 如果SQL数量少于目标数量，为每个SQL生成多个变体
        if len(sql_files) < target_count:
            # 计算每个SQL需要生成多少个NL查询
            queries_per_sql = (target_count + len(sql_files) - 1) // len(sql_files)
            # 限制最多每个SQL生成3个变体（避免过度重复）
            queries_per_sql = min(queries_per_sql, 3)
            # 确保不超过目标数量
            actual_target = min(len(sql_files) * queries_per_sql, target_count)
            
            print(f"  数据库有 {len(sql_files)} 个SQL，目标 {target_count} 条NL查询")
            print(f"  每个SQL将生成 {queries_per_sql} 个NL查询变体，预计生成 {actual_target} 条")
            
            # 为每个SQL生成多个NL查询变体
            for sql_file in sql_files:
                sql_file_path = os.path.join(sql_db_dir, sql_file)
                base_idx = int(re.findall(r'\d+', sql_file)[0]) if re.findall(r'\d+', sql_file) else 0
                
                # 为每个SQL生成多个NL查询变体
                for variant in range(queries_per_sql):
                    if len(tasks) >= target_count:
                        break
                    
                    # 第一个变体使用原始文件名，后续变体使用带索引的文件名
                    if variant == 0:
                        output_file = os.path.join(output_db_dir, f'generated_nl_query_{base_idx}.json')
                    else:
                        # 使用新的索引，确保不覆盖
                        new_idx = len(sql_files) * variant + base_idx
                        output_file = os.path.join(output_db_dir, f'generated_nl_query_{new_idx}.json')
                    
                    # 跳过已存在的文件（避免重复生成）
                    if os.path.exists(output_file):
                        continue
                    
                    tasks.append((sql_file_path, schema_file, output_file, variant))
        else:
            # 如果SQL数量足够，优先处理没有NL查询的SQL
            print(f"  数据库有 {len(sql_files)} 个SQL，目标 {target_count} 条NL查询")
            print(f"  优先为没有NL查询的SQL生成NL查询")
            
            # 找出所有没有NL查询的SQL文件
            missing_nl_sqls = []
            for sql_file in sql_files:
                sql_file_path = os.path.join(sql_db_dir, sql_file)
                file_idx = int(re.findall(r'\d+', sql_file)[0]) if re.findall(r'\d+', sql_file) else -1
                if file_idx < 0:
                    continue
                
                output_file = os.path.join(output_db_dir, f'generated_nl_query_{file_idx}.json')
                
                # 如果NL查询文件不存在，加入待处理列表
                if not os.path.exists(output_file):
                    missing_nl_sqls.append((sql_file_path, file_idx, output_file))
            
            print(f"  找到 {len(missing_nl_sqls)} 个没有NL查询的SQL")
            
            # 优先处理没有NL查询的SQL
            for sql_file_path, file_idx, output_file in missing_nl_sqls[:target_count]:
                tasks.append((sql_file_path, schema_file, output_file, 0))
            
            # 如果还不够目标数量，为已有NL查询的SQL生成变体
            if len(tasks) < target_count:
                remaining = target_count - len(tasks)
                print(f"  还需要 {remaining} 条，为已有NL查询的SQL生成变体")
                
                # 找出已有NL查询的SQL
                existing_nl_sqls = []
                for sql_file in sql_files:
                    sql_file_path = os.path.join(sql_db_dir, sql_file)
                    file_idx = int(re.findall(r'\d+', sql_file)[0]) if re.findall(r'\d+', sql_file) else -1
                    if file_idx < 0:
                        continue
                    
                    output_file = os.path.join(output_db_dir, f'generated_nl_query_{file_idx}.json')
                    
                    # 如果NL查询文件存在，可以生成变体
                    if os.path.exists(output_file):
                        # 生成变体，使用新的索引
                        variant_idx = len(sql_files) + file_idx
                        variant_output_file = os.path.join(output_db_dir, f'generated_nl_query_{variant_idx}.json')
                        if not os.path.exists(variant_output_file):
                            existing_nl_sqls.append((sql_file_path, variant_output_file, 1))
                
                # 添加变体任务
                for sql_file_path, variant_output_file, variant in existing_nl_sqls[:remaining]:
                    tasks.append((sql_file_path, schema_file, variant_output_file, variant))
            
            print(f"  总共将处理 {len(tasks)} 个任务")
        
        # 并发处理
        if tasks:
            with ThreadPoolExecutor(max_workers=args.max_workers) as executor:
                futures = {executor.submit(process_single_sql, sql_path, schema_file, out_file, variant): (sql_path, out_file) 
                          for sql_path, _, out_file, variant in tasks}
                
                for future in tqdm(as_completed(futures), total=len(futures), desc=f"处理 {database}"):
                    sql_path, out_file = futures[future]
                    total_processed += 1
                    try:
                        if future.result():
                            total_success += 1
                    except Exception as e:
                        print(f"处理失败 {sql_path}: {e}")
    
    print(f"\n完成！处理: {total_processed}, 成功: {total_success}")

if __name__ == '__main__':
    main()

