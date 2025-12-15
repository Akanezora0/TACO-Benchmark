#!/usr/bin/env python3
"""
改进的SQL骨架填充脚本

关键改进：
1. 真正利用图结构来选择相关的表和列
2. 利用外键关系选择可以JOIN的表
3. 增强Prompt，包含表描述、列信息、外键关系
4. 智能推理：根据SQL骨架的语义选择最合适的表
5. 集成API配置
"""

import json
import os
import re
from tqdm import tqdm
import random
import sqlparse
import sqlite3
import networkx as nx
from openai import OpenAI
from collections import defaultdict
import argparse
import yaml
from concurrent.futures import ThreadPoolExecutor, as_completed
import time

# 导入图信息提取模块
try:
    from .graph_extractor import extract_relevant_nodes_from_graph, format_extracted_info_for_prompt
except ImportError:
    # 如果相对导入失败，尝试绝对导入
    import sys
    import os
    script_dir = os.path.dirname(os.path.abspath(__file__))
    sys.path.insert(0, script_dir)
    from graph_extractor import extract_relevant_nodes_from_graph, format_extracted_info_for_prompt

# 加载API配置
def load_config(config_file=None):
    """加载配置文件"""
    if config_file is None:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        config_file = os.path.join(script_dir, 'config.yaml')
    
    if os.path.exists(config_file):
        with open(config_file, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        return config.get('llm', {})
    else:
        # 默认配置（与用户提供的配置一致）
        return {
            "api_url": "https://35.aigcbest.top/v1",
            "api_key": "sk-SeJvPPUTe9rGLtPP182bD0320779480a9705C39d25Be0215",
            "model": "gpt-4o-mini",
            "temperature": 0.1,
            "max_tokens": 8000
        }

# 全局配置和客户端（延迟初始化）
API_CONFIG = None
client = None

def get_client():
    """获取OpenAI客户端（延迟初始化）"""
    global client, API_CONFIG
    if client is None:
        if API_CONFIG is None:
            API_CONFIG = load_config()
        # 确保base_url格式正确（不以斜杠结尾，OpenAI SDK会自动添加）
        api_url = API_CONFIG["api_url"].rstrip('/')
        client = OpenAI(
            base_url=api_url,
            api_key=API_CONFIG["api_key"]
        )
    return client

def load_schema(schema_file):
    """加载数据库的schema信息"""
    with open(schema_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # 检查是否是标准schema格式（包含'tables'键）
    if 'tables' in data:
        return data
    
    # 如果不是标准格式，从数据库JSON文件中提取schema
    # 数据库JSON格式：{表名: {columns: [...], data: [...]}}
    schema = {'tables': []}
    for table_name, table_data in data.items():
        columns = []
        if 'columns' in table_data:
            # 从columns列表提取列名
            for col_name in table_data['columns']:
                # 尝试推断数据类型（默认为TEXT）
                columns.append({
                    'column_name': col_name,
                    'data_type': 'TEXT'  # 默认类型
                })
        
        schema['tables'].append({
            'table_name': table_name,
            'table_comment': table_name,
            'table_description': 'No description available.',
            'columns': columns,
            'primary_keys': [],
            'foreign_keys': []
        })
    
    return schema

def load_graph(graph_file):
    """加载图文件"""
    if not os.path.exists(graph_file):
        return None
    return nx.read_graphml(graph_file)

def load_graph_metadata(metadata_file):
    """加载图元数据"""
    if not os.path.exists(metadata_file):
        return None
    with open(metadata_file, 'r', encoding='utf-8') as f:
        return json.load(f)

def analyze_sql_skeleton(sql_skeleton):
    """
    分析SQL骨架，提取语义信息。
    返回：
    - has_join: 是否包含JOIN
    - has_subquery: 是否包含子查询
    - has_aggregate: 是否包含聚合函数
    - required_tables: 估计需要的表数量
    """
    sql_upper = sql_skeleton.upper()
    
    has_join = 'JOIN' in sql_upper
    has_subquery = '(' in sql_skeleton and 'SELECT' in sql_upper
    has_aggregate = any(func in sql_upper for func in ['COUNT', 'SUM', 'AVG', 'MAX', 'MIN', 'GROUP BY'])
    
    # 根据占位符数量和JOIN数量估计需要的表数
    num_placeholders = sql_skeleton.count('_')
    num_joins = sql_upper.count('JOIN')
    
    if has_join:
        required_tables = min(num_joins + 1, 3)  # JOIN通常需要2-3张表
    elif num_placeholders <= 3:
        required_tables = 1
    else:
        required_tables = min(2, num_placeholders // 3)
    
    return {
        'has_join': has_join,
        'has_subquery': has_subquery,
        'has_aggregate': has_aggregate,
        'required_tables': required_tables,
        'num_joins': num_joins
    }

def find_tables_with_common_columns(schema_info, min_common_cols=1):
    """
    查找有公共列的表对，可用于JOIN
    
    返回: [(table1, table2, common_columns), ...]
    """
    table_column_sets = {}
    for table_name, columns in schema_info['columns'].items():
        # 提取列名（去掉表名前缀）
        column_names = set()
        for col in columns:
            if '.' in col:
                column_names.add(col.split('.', 1)[1])
            else:
                column_names.add(col)
        table_column_sets[table_name] = column_names
    
    common_column_pairs = []
    table_list = list(table_column_sets.keys())
    
    for i, table1 in enumerate(table_list):
        for table2 in table_list[i+1:]:
            common_cols = table_column_sets[table1] & table_column_sets[table2]
            if len(common_cols) >= min_common_cols:
                common_column_pairs.append((table1, table2, list(common_cols)))
    
    return common_column_pairs

def find_common_columns_for_tables(selected_tables, selected_columns):
    """
    查找已选表之间的公共列，用于JOIN提示
    
    返回: [(table1, table2, common_columns), ...]
    """
    # 提取每张表的列名（去掉表名前缀）
    table_column_sets = {}
    for table in selected_tables:
        if table in selected_columns:
            column_names = set()
            for col in selected_columns[table]:
                if '.' in col:
                    column_names.add(col.split('.', 1)[1])
                else:
                    column_names.add(col)
            table_column_sets[table] = column_names
    
    common_column_pairs = []
    for i, table1 in enumerate(selected_tables):
        for table2 in selected_tables[i+1:]:
            if table1 in table_column_sets and table2 in table_column_sets:
                common_cols = table_column_sets[table1] & table_column_sets[table2]
                if common_cols:
                    common_column_pairs.append((table1, table2, list(common_cols)))
    
    return common_column_pairs

def select_tables_using_graph(G, metadata, sql_analysis, schema_info):
    """
    利用元数据智能选择表（不再需要图结构，G参数保留用于向后兼容）。
    
    策略：
    1. 如果有JOIN，优先选择有外键关系的表对
    2. 如果没有外键关系但需要JOIN，查找有公共列的表对
    3. 如果都不满足，随机选择表（保留原有逻辑）
    4. 考虑表的列数量，选择列数适中的表
    """
    if metadata is None:
        # 如果没有元数据，回退到随机选择
        return select_random_tables(schema_info, sql_analysis['required_tables'])
    
    all_tables = list(metadata['table_info'].keys())
    required_tables = sql_analysis['required_tables']
    
    if required_tables == 1:
        # 单表查询，随机选择
        selected_table = random.choice(all_tables)
        return [selected_table], {selected_table: metadata['table_info'][selected_table]['columns']}
    
    # 多表查询，优先选择有外键关系的表
    fk_relations = metadata['foreign_key_relations']
    
    if sql_analysis['has_join']:
        # 需要JOIN操作
        if fk_relations:
            # 有外键关系，优先使用外键关系
            fk_relation = random.choice(fk_relations)
            source_table = fk_relation['source_table']
            target_table = fk_relation['target_table']
            
            selected_tables = [source_table, target_table]
            
            # 如果还需要更多表，随机添加
            if required_tables > 2:
                remaining_tables = [t for t in all_tables if t not in selected_tables]
                if remaining_tables:
                    additional = random.sample(remaining_tables, min(required_tables - 2, len(remaining_tables)))
                    selected_tables.extend(additional)
        else:
            # 没有外键关系，查找有公共列的表对
            common_column_pairs = find_tables_with_common_columns(schema_info, min_common_cols=1)
            
            if common_column_pairs:
                # 随机选择一个有公共列的表对
                table1, table2, common_cols = random.choice(common_column_pairs)
                selected_tables = [table1, table2]
                
                # 如果还需要更多表，随机添加
                if required_tables > 2:
                    remaining_tables = [t for t in all_tables if t not in selected_tables]
                    if remaining_tables:
                        additional = random.sample(remaining_tables, min(required_tables - 2, len(remaining_tables)))
                        selected_tables.extend(additional)
            else:
                # 没有公共列，随机选择（这种情况可能无法正确JOIN，但至少能生成SQL）
                selected_tables = random.sample(all_tables, min(required_tables, len(all_tables)))
    else:
        # 不需要JOIN，随机选择
        selected_tables = random.sample(all_tables, min(required_tables, len(all_tables)))
    
    # 构建选中的表和列信息
    selected_columns = {}
    for table in selected_tables:
        if table in metadata['table_info']:
            selected_columns[table] = metadata['table_info'][table]['columns']
        else:
            # 回退：从schema_info中获取
            selected_columns[table] = schema_info['columns'].get(table, [])
    
    return selected_tables, selected_columns

def select_random_tables(schema_info, num_tables=2):
    """随机选择表（保留原有逻辑）"""
    all_tables = schema_info['tables']
    selected_tables = random.sample(all_tables, min(num_tables, len(all_tables)))
    selected_columns = {}
    for table in selected_tables:
        selected_columns[table] = schema_info['columns'].get(table, [])
    return selected_tables, selected_columns

def extract_schema_info(schema):
    """从schema中提取表名和列名信息"""
    schema_info = {
        'tables': [],
        'columns': {}
    }
    for table in schema['tables']:
        table_name = table['table_name']
        schema_info['tables'].append(table_name)
        columns = []
        for column in table['columns']:
            column_name = column['column_name']
            full_column_name = f"{table_name}.{column_name}"
            columns.append(full_column_name)
        schema_info['columns'][table_name] = columns
    return schema_info

def extract_graph_metadata_from_loaded(metadata_dict):
    """
    从加载的元数据字典中提取信息（使用原始名称）。
    支持两种格式：
    1. 旧格式：包含node_id_map、table_info、column_info
    2. 新格式（优化版）：直接包含tables和foreign_keys
    """
    # 检查是否是新格式（优化版）
    if 'tables' in metadata_dict and isinstance(metadata_dict['tables'], dict):
        # 新格式：直接使用tables和foreign_keys
        table_info = {}
        column_info = {}
        foreign_key_relations = []
        
        # 处理表信息
        for table_name, table_data in metadata_dict['tables'].items():
            table_info[table_name] = {
                'name': table_data.get('name', table_name),
                'comment': table_data.get('comment', ''),
                'description': table_data.get('description', 'No description available.'),
                'columns': []
            }
            
            # 处理列信息
            for col in table_data.get('columns', []):
                col_name = col.get('name', '')
                full_column_name = f"{table_name}.{col_name}"
                column_info[full_column_name] = {
                    'full_name': full_column_name,
                    'table': table_name,
                    'column': col_name,
                    'data_type': col.get('data_type', 'TEXT')
                }
                table_info[table_name]['columns'].append(full_column_name)
        
        # 处理外键关系
        for fk in metadata_dict.get('foreign_keys', []):
            source_table = fk.get('source_table', '')
            source_column = fk.get('source_column', '')
            target_table = fk.get('target_table', '')
            target_column = fk.get('target_column', '')
            
            if source_table and source_column and target_table and target_column:
                source_full = f"{source_table}.{source_column}"
                target_full = f"{target_table}.{target_column}"
                foreign_key_relations.append({
                    'source': source_full,
                    'target': target_full,
                    'source_table': source_table,
                    'target_table': target_table
                })
        
        return {
            'foreign_key_relations': foreign_key_relations,
            'table_info': table_info,
            'column_info': column_info
        }
    
    # 旧格式：使用node_id_map恢复原始名称
    foreign_key_relations = metadata_dict.get('foreign_key_relations', [])
    table_info = {}
    column_info = {}
    node_id_map = metadata_dict.get('node_id_map', {})
    
    # 处理表信息（使用原始名称）
    for cleaned_id, original_name in node_id_map.items():
        if cleaned_id in metadata_dict.get('table_info', {}):
            table_meta = metadata_dict['table_info'][cleaned_id]
            table_info[original_name] = {
                'name': original_name,
                'comment': table_meta.get('comment', ''),
                'description': table_meta.get('description', 'No description available.'),
                'columns': []
            }
    
    # 处理列信息
    for cleaned_id, original_name in node_id_map.items():
        if cleaned_id in metadata_dict.get('column_info', {}):
            col_meta = metadata_dict['column_info'][cleaned_id]
            if '.' in original_name:
                table_name = original_name.split('.')[0]
            else:
                table_name = col_meta.get('table', '')
            
            column_info[original_name] = {
                'full_name': original_name,
                'table': table_name,
                'column': col_meta.get('column', ''),
                'data_type': col_meta.get('data_type', 'TEXT')
            }
            if table_name in table_info:
                table_info[table_name]['columns'].append(original_name)
    
    # 处理外键关系（使用原始名称）
    fk_relations_original = []
    for fk in foreign_key_relations:
        source_cleaned = fk.get('source', '')
        target_cleaned = fk.get('target', '')
        source_original = node_id_map.get(source_cleaned, source_cleaned)
        target_original = node_id_map.get(target_cleaned, target_cleaned)
        fk_relations_original.append({
            'source': source_original,
            'target': target_original,
            'source_table': fk.get('source_table', ''),
            'target_table': fk.get('target_table', '')
        })
    
    return {
        'foreign_key_relations': fk_relations_original,
        'table_info': table_info,
        'column_info': column_info
    }

def format_foreign_key_relations(metadata, selected_tables):
    """格式化外键关系信息，用于Prompt"""
    if not metadata or 'foreign_key_relations' not in metadata:
        return ""
    
    fk_relations = metadata['foreign_key_relations']
    relevant_fks = [
        fk for fk in fk_relations 
        if fk['source_table'] in selected_tables and fk['target_table'] in selected_tables
    ]
    
    if not relevant_fks:
        return ""
    
    fk_text = "\n外键关系：\n"
    for fk in relevant_fks:
        fk_text += f"- {fk['source']} 引用 {fk['target']}\n"
        fk_text += f"  ({fk['source_table']} 表可以通过 {fk['source'].split('.')[1]} 列与 {fk['target_table']} 表的 {fk['target'].split('.')[1]} 列进行JOIN)\n"
    
    return fk_text

def format_table_info(metadata, selected_tables, schema):
    """格式化表信息，用于Prompt"""
    table_info_text = "\n表详细信息：\n"
    
    for table_name in selected_tables:
        table_info_text += f"\n表名：{table_name}\n"
        
        # 从metadata获取表描述
        if metadata and table_name in metadata.get('table_info', {}):
            table_meta = metadata['table_info'][table_name]
            if table_meta.get('description') and table_meta['description'] != 'No description available.':
                table_info_text += f"描述：{table_meta['description']}\n"
            if table_meta.get('comment'):
                table_info_text += f"注释：{table_meta['comment']}\n"
        
        # 从schema获取列信息
        table_info_text += "列信息：\n"
        for table in schema['tables']:
            if table['table_name'] == table_name:
                for column in table['columns']:
                    column_name = column['column_name']
                    data_type = column.get('data_type', 'TEXT')
                    full_column_name = f"{table_name}.{column_name}"
                    table_info_text += f"  - {full_column_name} (类型: {data_type})\n"
                break
    
    return table_info_text

def construct_enhanced_prompt(sql_skeleton, selected_tables, selected_columns, 
                              metadata, schema, sql_analysis, cross_database=False):
    """
    构建增强的Prompt，包含：
    1. SQL骨架
    2. 表详细信息（描述、注释）
    3. 列信息（数据类型）
    4. 外键关系
    5. SQL骨架分析结果
    """
    def quote_identifier(identifier):
        """使用双引号包裹标识符，确保SQLite可以正确处理中文和特殊字符"""
        # 转义双引号
        escaped = str(identifier).replace('"', '""')
        return f'"{escaped}"'
    
    # 格式化表名（所有表名都用双引号包裹）
    tables = ', '.join([quote_identifier(table) for table in selected_tables])
    
    # 格式化列名（所有列名都用双引号包裹）
    columns = []
    for table in selected_tables:
        if table in selected_columns:
            # 列名格式：表名.列名，需要分别包裹
            for col in selected_columns[table]:
                if '.' in col:
                    # 如果已经是"表名.列名"格式，需要分别处理
                    parts = col.split('.', 1)
                    if len(parts) == 2:
                        table_part, col_part = parts
                        quoted_col = f'{quote_identifier(table_part)}.{quote_identifier(col_part)}'
                    else:
                        quoted_col = quote_identifier(col)
                else:
                    quoted_col = quote_identifier(col)
                columns.append(quoted_col)
    columns_str = ', '.join(columns)
    
    # 格式化表详细信息
    table_info_text = format_table_info(metadata, selected_tables, schema)
    
    # 格式化外键关系
    fk_text = format_foreign_key_relations(metadata, selected_tables)
    
    # SQL骨架分析提示
    analysis_hints = ""
    if sql_analysis['has_join']:
        analysis_hints += "\n提示：此SQL骨架包含JOIN操作。\n"
        if fk_text:
            analysis_hints += "  - 优先使用外键关系来连接表（见下方外键关系部分）。\n"
        # 检查是否有公共列可用于JOIN
        if len(selected_tables) >= 2:
            common_cols_info = find_common_columns_for_tables(selected_tables, selected_columns)
            if common_cols_info:
                analysis_hints += "  - 如果没有外键关系，可以使用公共列进行JOIN。\n"
                analysis_hints += "  - 以下表对有公共列，可用于JOIN条件：\n"
                for table1, table2, common_cols in common_cols_info:
                    common_cols_str = ', '.join([f'"{col}"' for col in common_cols[:3]])  # 只显示前3个
                    analysis_hints += f"    * {table1[:50]}... 和 {table2[:50]}... 的公共列: {common_cols_str}\n"
    if sql_analysis['has_aggregate']:
        analysis_hints += "提示：此SQL骨架包含聚合函数，请确保GROUP BY子句正确。\n"
    if sql_analysis['has_subquery']:
        analysis_hints += "提示：此SQL骨架包含子查询，请确保子查询语法正确。\n"
    
    if cross_database:
        databases = ', '.join(list(set([table.split('.')[0] for table in selected_tables if '.' in table])))
        prompt = f"""请根据以下 SQL 框架和可用的表名、列名，填充占位符"_"，生成完整且可在 SQLite 上正确执行的 SQL 语句。

严格要求：
- **仅输出最终生成的完整 SQL 语句，不要重复提示内容。**
- **生成的 SQL 要保证语法正确，可以直接在 SQLite 上运行得到结果。**
- **不要添加任何额外的解释、注释或输出格式（代码块，空格等）。**
- **添加的表名、列名、WHERE 条件等内容必须在给定的表和列中。**
- **所有表名和列名都必须用双引号包裹（包括中文和特殊字符），例如："表名" 或 "表名"."列名"**
- **SQLite支持中文表名和列名，只要用双引号正确包裹即可。**
- **可以对给定的SQL框架做调整，最后生成更合理的SQL语句。**
- **如果SQL骨架包含JOIN，请优先使用外键关系来连接表。如果没有外键关系，可以使用公共列进行JOIN（见下方提示）。**

SQL 框架：
{sql_skeleton}

可用的数据库：
{databases}

可用的表名：
{tables}

可用的列名（格式：表名.列名）：
{columns_str}
{table_info_text}
{fk_text}
{analysis_hints}

请仅输出生成的完整 SQL 语句：
"""
    else:
        prompt = f"""请根据以下 SQL 框架和可用的表名、列名，填充占位符"_"，生成完整且可在 SQLite 上正确执行的 SQL 语句。

严格要求：
- **仅输出最终生成的完整 SQL 语句，不要重复提示内容。**
- **生成的 SQL 要保证语法正确，可以直接在 SQLite 上运行得到结果。**
- **不要添加任何额外的解释、注释或输出格式（代码块，空格等）。**
- **添加的表名、列名、WHERE 条件等内容必须在给定的表和列中。**
- **所有表名和列名都必须用双引号包裹（包括中文和特殊字符），例如："表名" 或 "表名"."列名"**
- **SQLite支持中文表名和列名，只要用双引号正确包裹即可。**
- **可以对给定的SQL框架做调整，最后生成更合理的SQL语句。**
- **如果SQL骨架包含JOIN，请优先使用外键关系来连接表。如果没有外键关系，可以使用公共列进行JOIN（见下方提示）。**

SQL 框架：
{sql_skeleton}

可用的表名：
{tables}

可用的列名（格式：表名.列名）：
{columns_str}
{table_info_text}
{fk_text}
{analysis_hints}

请仅输出生成的完整 SQL 语句：
"""
    
    return prompt.strip()

def construct_compact_prompt(sql_framework, extracted_info, schema):
    """
    使用从图文件中提取的精简信息构建prompt
    这个版本只包含与SQL骨架相关的表和列，大幅减小prompt大小
    """
    def quote_identifier(identifier):
        """使用双引号包裹标识符，确保SQLite可以正确处理中文和特殊字符"""
        escaped = str(identifier).replace('"', '""')
        return f'"{escaped}"'
    
    tables = extracted_info.get('tables', [])
    table_info = extracted_info.get('table_info', {})
    columns = extracted_info.get('columns', {})
    column_info = extracted_info.get('column_info', {})
    foreign_keys = extracted_info.get('foreign_keys', [])
    sql_analysis = extracted_info.get('sql_analysis', {})
    
    # 格式化表名
    tables_str = ', '.join([quote_identifier(table) for table in tables])
    
    # 格式化列名
    columns_list = []
    for table in tables:
        if table in columns:
            for col_name in columns[table]:
                if '.' in col_name:
                    parts = col_name.split('.', 1)
                    if len(parts) == 2:
                        table_part, col_part = parts
                        quoted_col = f'{quote_identifier(table_part)}.{quote_identifier(col_part)}'
                    else:
                        quoted_col = quote_identifier(col_name)
                else:
                    quoted_col = f'{quote_identifier(table)}.{quote_identifier(col_name)}'
                columns_list.append(quoted_col)
    columns_str = ', '.join(columns_list)
    
    # 格式化表详细信息
    table_info_text = "\n表详细信息：\n"
    for table_name in tables:
        table_info_text += f"\n表名：{table_name}\n"
        
        if table_name in table_info:
            info = table_info[table_name]
            if info.get('description') and info['description'] != 'No description available.':
                table_info_text += f"描述：{info['description']}\n"
            if info.get('comment'):
                table_info_text += f"注释：{info['comment']}\n"
        
        # 列信息
        if table_name in columns:
            table_info_text += "列信息：\n"
            for col_name in columns[table_name]:
                if col_name in column_info:
                    col_info = column_info[col_name]
                    data_type = col_info.get('data_type', 'TEXT')
                    table_info_text += f"  - {col_name} (类型: {data_type})\n"
    
    # 格式化外键关系
    fk_text = ""
    if foreign_keys:
        fk_text = "\n外键关系（可用于JOIN）：\n"
        for fk in foreign_keys:
            source_full = f"{fk['source_table']}.{fk['source_column']}"
            target_full = f"{fk['target_table']}.{fk['target_column']}"
            fk_text += f"- {source_full} 引用 {target_full}\n"
            fk_text += f"  ({fk['source_table']} 表可以通过 {fk['source_column']} 列与 {fk['target_table']} 表的 {fk['target_column']} 列进行JOIN)\n"
    
    # SQL骨架分析提示
    analysis_hints = ""
    if sql_analysis.get('has_join'):
        analysis_hints += "\n提示：此SQL骨架包含JOIN操作，请使用外键关系来连接表。\n"
    if sql_analysis.get('has_aggregate'):
        analysis_hints += "提示：此SQL骨架包含聚合函数，请确保GROUP BY子句正确。\n"
    if sql_analysis.get('has_subquery'):
        analysis_hints += "提示：此SQL骨架包含子查询，请确保子查询语法正确。\n"
    
    prompt = f"""请根据以下 SQL 框架和可用的表名、列名，填充占位符"_"，生成完整且可在 SQLite 上正确执行的 SQL 语句。

严格要求：
- **仅输出最终生成的完整 SQL 语句，不要重复提示内容。**
- **生成的 SQL 要保证语法正确，可以直接在 SQLite 上运行得到结果。**
- **不要添加任何额外的解释、注释或输出格式（代码块，空格等）。**
- **添加的表名、列名、WHERE 条件等内容必须在给定的表和列中。**
- **所有表名和列名都必须用双引号包裹（包括中文和特殊字符），例如："表名" 或 "表名"."列名"**
- **SQLite支持中文表名和列名，只要用双引号正确包裹即可。**
- **可以对给定的SQL框架做调整，最后生成更合理的SQL语句。**
- **如果SQL骨架包含JOIN，请优先使用外键关系来连接表。如果没有外键关系，可以使用公共列进行JOIN（见下方提示）。**

SQL 框架：
{sql_framework}

可用的表名（已根据SQL骨架筛选，共{len(tables)}个）：
{tables_str}

可用的列名（格式：表名.列名，已根据SQL骨架筛选）：
{columns_str}
{table_info_text}
{fk_text}
{analysis_hints}

请仅输出生成的完整 SQL 语句：
"""
    
    return prompt.strip()

def generate_text(prompt):
    """调用大模型生成SQL"""
    try:
        client = get_client()
        response = client.chat.completions.create(
            model=API_CONFIG["model"],
            temperature=API_CONFIG["temperature"],
            max_tokens=API_CONFIG["max_tokens"],
            messages=[
                {"role": "system", "content": "You are a helpful assistant specialized in SQL generation."},
                {"role": "user", "content": prompt},
            ],
        )
        assistant_reply = response.choices[0].message.content
        return assistant_reply.strip()
    except Exception as e:
        print(f"生成SQL时出错: {e}")
        return None

def extract_sql_from_response(response):
    """提取模型输出中的SQL语句"""
    sql_statement = response.strip()
    # 移除可能的代码块标记
    sql_statement = re.sub(r'```sql\s*', '', sql_statement, flags=re.IGNORECASE)
    sql_statement = re.sub(r'```\s*', '', sql_statement)
    # 确保以 "SELECT" 开头
    if not sql_statement.upper().startswith('SELECT'):
        match = re.search(r'(SELECT\s.*)', sql_statement, re.IGNORECASE | re.DOTALL)
        if match:
            sql_statement = match.group(1).strip()
    return sql_statement

def is_valid_sql(sql_statement):
    """验证SQL语句的语法"""
    try:
        parsed = sqlparse.parse(sql_statement)
        if parsed and len(parsed) > 0:
            return True
        else:
            return False
    except Exception:
        return False

def execute_single_db_sql(sql, db_path):
    """执行单一数据库的SQL语句"""
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        cursor.execute(sql)
        results = cursor.fetchall()
        conn.close()
        if results:
            return results, True
        else:
            return [], False  # 空结果也算成功
    except sqlite3.Error as e:
        # 记录更详细的错误信息
        error_msg = str(e)
        # 不打印每个错误，避免输出过多
        return None, False
    except Exception as e:
        # 记录其他类型的错误
        error_msg = str(e)
        return None, False

def process_single_sql_skeleton(args):
    """处理单个SQL骨架（用于并发处理）"""
    idx, sql_skeleton, database_name, schema, schema_info, graph_dir, single_output_path, schema_file, max_retries = args
    
    # 提取SQL骨架字符串
    if isinstance(sql_skeleton, dict):
        sql_framework = sql_skeleton.get('sql_framework', '')
    else:
        sql_framework = sql_skeleton
    
    if not sql_framework:
        return idx, False, "SQL骨架为空"
    
    # 检查输出文件是否存在
    output_file = os.path.join(single_output_path, f'generated_sql_{idx}.json')
    if os.path.exists(output_file):
        return idx, True, "已存在"
    
    # 分析SQL骨架
    sql_analysis = analyze_sql_skeleton(sql_framework)
    
    # 加载元数据（优先使用metadata文件）
    metadata_file = os.path.join(graph_dir, database_name, f"{database_name}_metadata_{idx}.json")
    metadata_dict = load_graph_metadata(metadata_file)
    
    # 从元数据中提取信息（使用原始名称）
    if metadata_dict:
        metadata = extract_graph_metadata_from_loaded(metadata_dict)
    else:
        metadata = None
    
    # 尝试从图文件中提取关键信息（优先使用图文件提取，更精准）
    graph_file = os.path.join(graph_dir, database_name, f"{database_name}_graph_{idx}.graphml")
    extracted_info = None
    use_extracted_info = False
    
    if os.path.exists(graph_file):
        try:
            # 从图文件中提取与SQL骨架相关的关键信息
            G = load_graph(graph_file)
            extracted_info = extract_relevant_nodes_from_graph(G, sql_framework, max_tables=5, max_columns_per_table=10)
            
            if extracted_info and len(extracted_info.get('tables', [])) > 0:
                use_extracted_info = True
                # 将提取的信息转换为metadata格式
                metadata = {
                    'foreign_key_relations': [
                        {
                            'source': f"{fk['source_table']}.{fk['source_column']}",
                            'target': f"{fk['target_table']}.{fk['target_column']}",
                            'source_table': fk['source_table'],
                            'target_table': fk['target_table']
                        }
                        for fk in extracted_info.get('foreign_keys', [])
                    ],
                    'table_info': extracted_info.get('table_info', {}),
                    'column_info': extracted_info.get('column_info', {})
                }
                
                # 使用提取的表和列
                selected_tables = extracted_info.get('tables', [])
                selected_columns = extracted_info.get('columns', {})
                
                # 如果提取的表不够，补充随机选择
                if len(selected_tables) < sql_analysis['required_tables']:
                    remaining_tables = [t for t in schema_info['tables'] if t not in selected_tables]
                    if remaining_tables:
                        additional = random.sample(
                            remaining_tables, 
                            min(sql_analysis['required_tables'] - len(selected_tables), len(remaining_tables))
                        )
                        selected_tables.extend(additional)
                        for table in additional:
                            if table in schema_info['columns']:
                                selected_columns[table] = schema_info['columns'][table]
        except Exception as e:
            # 加载图文件失败，回退到metadata或随机选择
            use_extracted_info = False
    
    # 如果没有从图文件提取到信息，使用metadata或随机选择
    if not use_extracted_info:
        # 加载元数据（优先使用metadata文件）
        if not metadata_dict:
            metadata_dict = load_graph_metadata(metadata_file)
        
        # 从元数据中提取信息（使用原始名称）
        if metadata_dict:
            metadata = extract_graph_metadata_from_loaded(metadata_dict)
        else:
            metadata = None
        
        # 利用元数据选择表（不再需要图结构）
        selected_tables, selected_columns = select_tables_using_graph(
            None, metadata, sql_analysis, schema_info
        )
    
    # 构建Prompt（如果从图文件提取了信息，使用精简版prompt）
    if use_extracted_info and extracted_info:
        prompt = construct_compact_prompt(sql_framework, extracted_info, schema)
    else:
        prompt = construct_enhanced_prompt(
            sql_framework, selected_tables, selected_columns,
            metadata, schema, sql_analysis, cross_database=False
        )
    
    # 尝试生成SQL（带重试）
    sql_statement = None
    error_info = None
    for attempt in range(1, max_retries + 1):
        try:
            sql_statement = generate_text(prompt)
            if not sql_statement:
                error_info = "LLM生成失败"
                if attempt < max_retries:
                    time.sleep(1)  # 重试前等待
                continue
            
            sql_statement = extract_sql_from_response(sql_statement)
            
            if is_valid_sql(sql_statement) and sql_statement.upper().startswith('SELECT'):
                # 获取数据库路径
                db_path = schema_file.replace('.json', '.db')
                if not os.path.exists(db_path):
                    error_info = f"数据库文件不存在: {db_path}"
                    break
                
                # 执行SQL
                results, success = execute_single_db_sql(sql_statement, db_path)
                if success:
                    # 保存结果
                    save_data = {
                        'sql': sql_statement,
                        'results': results[:10] if results else [],  # 只保存前10条结果
                        'sql_skeleton': sql_framework,
                        'database': database_name,
                        'tables': {table: selected_columns[table] for table in selected_tables},
                        'metadata': {
                            'has_join': sql_analysis['has_join'],
                            'has_subquery': sql_analysis['has_subquery'],
                            'has_aggregate': sql_analysis['has_aggregate']
                        }
                    }
                    
                    with open(output_file, 'w', encoding='utf-8') as f:
                        json.dump(save_data, f, ensure_ascii=False, indent=2)
                    
                    return idx, True, "成功"
                else:
                    error_info = "SQL执行失败"
                    if attempt < max_retries:
                        time.sleep(1)  # 重试前等待
            else:
                error_info = "SQL语法验证失败"
                if attempt < max_retries:
                    time.sleep(1)  # 重试前等待
        except Exception as e:
            error_info = f"处理异常: {str(e)}"
            if attempt < max_retries:
                time.sleep(1)  # 重试前等待
    
    # 保存失败信息
    if not sql_statement or not is_valid_sql(sql_statement):
        error_file = output_file.replace('.json', '_error.json')
        with open(error_file, 'w', encoding='utf-8') as f:
            json.dump({
                'sql_skeleton': sql_framework,
                'database': database_name,
                'error': error_info,
                'generated_sql': sql_statement if sql_statement else None
            }, f, ensure_ascii=False, indent=2)
        return idx, False, error_info
    
    return idx, True, "成功"

def process_single_database(database_name, skeleton_file, schema_file, graph_dir, output_dir, max_retries=3, max_workers=None):
    """处理单个数据库的SQL骨架填充（支持并发）"""
    # 加载schema
    schema = load_schema(schema_file)
    schema_info = extract_schema_info(schema)
    
    # 加载SQL skeletons
    with open(skeleton_file, 'r', encoding='utf-8') as f:
        sql_skeletons = json.load(f)
    
    # 创建输出目录
    single_output_path = os.path.join(output_dir, 'single', database_name)
    os.makedirs(single_output_path, exist_ok=True)
    
    # 获取并发数
    if max_workers is None:
        max_workers = API_CONFIG.get('max_workers', 20) if API_CONFIG else 20
    
    print(f"正在处理数据库 '{database_name}'，共 {len(sql_skeletons)} 个SQL骨架...")
    print(f"并发数: {max_workers}, 最大重试次数: {max_retries}")
    
    success_count = 0
    fail_count = 0
    
    # 准备任务参数
    tasks = []
    for idx, sql_skeleton in enumerate(sql_skeletons):
        tasks.append((
            idx, sql_skeleton, database_name, schema, schema_info, 
            graph_dir, single_output_path, schema_file, max_retries
        ))
    
    # 使用线程池并发处理
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # 提交所有任务
        future_to_idx = {executor.submit(process_single_sql_skeleton, task): task[0] for task in tasks}
        
        # 使用tqdm显示进度
        with tqdm(total=len(tasks), desc=f"{database_name} 处理进度") as pbar:
            for future in as_completed(future_to_idx):
                idx = future_to_idx[future]
                try:
                    result_idx, success, message = future.result()
                    if success:
                        if message != "已存在":
                            success_count += 1
                    else:
                        fail_count += 1
                except Exception as e:
                    fail_count += 1
                    print(f"处理索引 {idx} 时发生异常: {e}")
                finally:
                    pbar.update(1)
    
    print(f"数据库 '{database_name}' 处理完成：成功 {success_count}，失败 {fail_count}")
    return success_count, fail_count

def main():
    global API_CONFIG
    
    parser = argparse.ArgumentParser(description='填充SQL骨架占位符（改进版）')
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(script_dir)))
    
    parser.add_argument('--skeleton_dir', type=str, default=None,
                       help='SQL骨架目录（默认：../../data/beijing/output/sql_skeleton）')
    parser.add_argument('--database_dir', type=str, default=None,
                       help='数据库目录（默认：../../data/beijing/database）')
    parser.add_argument('--graph_dir', type=str, default=None,
                       help='图目录（默认：../../data/beijing/output/graph）')
    parser.add_argument('--output_dir', type=str, default=None,
                       help='输出目录（默认：../../data/beijing/output）')
    parser.add_argument('--max_retries', type=int, default=3,
                       help='最大重试次数（默认3）')
    parser.add_argument('--config', type=str, default=None,
                       help='配置文件路径（默认：./config.yaml）')
    
    args = parser.parse_args()
    
    # 加载配置
    API_CONFIG = load_config(args.config)
    
    # 设置默认路径
    if args.skeleton_dir is None:
        args.skeleton_dir = os.path.join(project_root, 'benchmark', 'data', 'beijing', 'output', 'sql_skeleton')
    if args.database_dir is None:
        args.database_dir = os.path.join(project_root, 'benchmark', 'data', 'beijing', 'database')
    if args.graph_dir is None:
        args.graph_dir = os.path.join(project_root, 'benchmark', 'data', 'beijing', 'output', 'graph')
    if args.output_dir is None:
        args.output_dir = os.path.join(project_root, 'benchmark', 'data', 'beijing', 'output')
    
    # 转换为绝对路径
    args.skeleton_dir = os.path.abspath(args.skeleton_dir)
    args.database_dir = os.path.abspath(args.database_dir)
    args.graph_dir = os.path.abspath(args.graph_dir)
    args.output_dir = os.path.abspath(args.output_dir)
    
    # 获取所有SQL skeleton文件
    skeleton_files = [f for f in os.listdir(args.skeleton_dir) if f.endswith('_sql_skeleton.json')]
    
    print(f"找到 {len(skeleton_files)} 个数据库的SQL骨架文件")
    
    total_success = 0
    total_fail = 0
    
    for skeleton_file in tqdm(skeleton_files, desc="总体进度"):
        database_name = skeleton_file.replace('_sql_skeleton.json', '')
        
        skeleton_path = os.path.join(args.skeleton_dir, skeleton_file)
        schema_path = os.path.join(args.database_dir, database_name, f"{database_name}.json")
        
        success, fail = process_single_database(
            database_name, skeleton_path, schema_path,
            args.graph_dir, args.output_dir, args.max_retries
        )
        
        total_success += success
        total_fail += fail
    
    print(f"\n{'='*60}")
    print(f"✓ 所有数据库处理完成！")
    print(f"总计：成功 {total_success}，失败 {total_fail}")
    print(f"{'='*60}")

if __name__ == '__main__':
    main()

