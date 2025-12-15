#!/usr/bin/env python3
"""
跨数据库SQL骨架填充脚本

基于单数据库的SQL填充脚本，扩展支持跨数据库场景：
1. 加载多个数据库的schema和图文件
2. 在prompt中明确告知大模型这是跨数据库查询
3. 让大模型生成带数据库前缀的SQL（如：数据库名.表名）
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

# 导入单数据库的函数
import sys
import importlib.util
sql_filling_dir = os.path.join(os.path.dirname(__file__), '..', 'sql_filling')
sys.path.insert(0, sql_filling_dir)

# 动态导入
spec = importlib.util.spec_from_file_location(
    "fill_sql_placeholders_improved",
    os.path.join(sql_filling_dir, "2fill_sql_placeholders_improved.py")
)
fill_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(fill_module)

load_config = fill_module.load_config
get_client = fill_module.get_client
load_schema = fill_module.load_schema
analyze_sql_skeleton = fill_module.analyze_sql_skeleton
construct_enhanced_prompt = fill_module.construct_enhanced_prompt
# execute_sql 可能不存在，跨数据库SQL不需要直接执行验证
load_graph_metadata = getattr(fill_module, 'load_graph_metadata', None)

def convert_to_single_database_sql(cross_db_sql, table_database_mapping):
    """将跨数据库SQL转换为单数据库SQL（移除数据库前缀）"""
    single_db_sql = cross_db_sql
    # 替换 "数据库名"."表名" 为 "表名"
    for table, db in table_database_mapping.items():
        # 处理带引号的情况
        pattern1 = rf'"{re.escape(db)}"\."{re.escape(table)}"'
        replacement1 = f'"{table}"'
        single_db_sql = re.sub(pattern1, replacement1, single_db_sql)
        
        # 处理不带引号的情况
        pattern2 = rf'{re.escape(db)}\.{re.escape(table)}'
        replacement2 = table
        single_db_sql = re.sub(pattern2, replacement2, single_db_sql)
    
    # 替换 "数据库名"."表名"."列名" 为 "表名"."列名"
    for table, db in table_database_mapping.items():
        pattern = rf'"{re.escape(db)}"\."{re.escape(table)}"\."([^"]+)"'
        replacement = rf'"{table}"."\1"'
        single_db_sql = re.sub(pattern, replacement, single_db_sql)
    
    return single_db_sql

def execute_sql_on_database(sql, db_path):
    """在单个数据库上执行SQL"""
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        cursor.execute(sql)
        results = cursor.fetchall()
        conn.close()
        # 空结果也算成功（与单数据库保持一致）
        if results:
            return results, True
        else:
            return [], True  # 空结果也算成功
    except sqlite3.Error as e:
        return None, False
    except Exception as e:
        return None, False

def get_tables_in_database(db_path, alias=None):
    """获取数据库中的所有表名"""
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
        tables = [row[0] for row in cursor.fetchall()]
        conn.close()
        return set(tables)
    except:
        return set()

def validate_tables_in_sql(sql, databases, database_dir, table_database_mapping, db_aliases):
    """验证SQL中使用的表是否在对应数据库中存在"""
    # 提取SQL中使用的表名
    table_pattern = r'"(?:db\d+|[\u4e00-\u9fa5]+)"\."([^"]+)"'
    tables_in_sql = set(re.findall(table_pattern, sql))
    
    # 检查每个表是否在对应数据库中
    missing_tables = []
    for table_name in tables_in_sql:
        # 找到表对应的数据库
        db_name = table_database_mapping.get(table_name)
        if db_name:
            db_path = os.path.join(database_dir, db_name, f"{db_name}.db")
            if os.path.exists(db_path):
                tables_in_db = get_tables_in_database(db_path)
                if table_name not in tables_in_db:
                    missing_tables.append((table_name, db_name))
    
    return missing_tables

def execute_cross_database_sql_with_attach(cross_db_sql, databases, database_dir, table_database_mapping):
    """
    使用SQLite的ATTACH DATABASE功能执行跨数据库SQL
    添加表名验证，确保表在对应数据库中存在
    """
    if len(databases) < 2:
        # 如果只有一个数据库，直接执行
        db_path = os.path.join(database_dir, databases[0], f"{databases[0]}.db")
        if os.path.exists(db_path):
            return execute_sql_on_database(cross_db_sql, db_path)
        return None, False
    
    # 创建临时数据库作为主数据库
    import tempfile
    temp_db = tempfile.NamedTemporaryFile(suffix='.db', delete=False)
    temp_db.close()
    temp_db_path = temp_db.name
    
    try:
        conn = sqlite3.connect(temp_db_path)
        cursor = conn.cursor()
        
        # ATTACH所有涉及的数据库
        db_aliases = {}
        db_tables_cache = {}  # 缓存每个数据库的表名
        for i, db_name in enumerate(databases):
            db_path = os.path.join(database_dir, db_name, f"{db_name}.db")
            if os.path.exists(db_path):
                # 使用数据库名作为别名（但需要处理特殊字符）
                alias = f"db{i}"
                db_aliases[db_name] = alias
                cursor.execute(f'ATTACH DATABASE "{db_path}" AS {alias}')
                # 缓存表名
                db_tables_cache[alias] = get_tables_in_database(db_path)
        
        # 转换SQL：将"数据库名"."表名"转换为"别名"."表名"
        converted_sql = cross_db_sql
        for db_name, alias in db_aliases.items():
            # 先处理 "数据库名"."表名"."列名" 格式（避免被后面的替换影响）
            pattern2 = rf'"{re.escape(db_name)}"\."([^"]+)"\."([^"]+)"'
            replacement2 = rf'"{alias}"."\1"."\2"'
            converted_sql = re.sub(pattern2, replacement2, converted_sql)
            
            # 再处理 "数据库名"."表名" 格式
            pattern = rf'"{re.escape(db_name)}"\."([^"]+)"'
            replacement = rf'"{alias}"."\1"'
            converted_sql = re.sub(pattern, replacement, converted_sql)
            
            # 处理 "数据库名.表名" 格式（不带引号的点分隔）
            pattern3 = rf'"{re.escape(db_name)}\.([^"]+)"'
            replacement3 = rf'"{alias}"."\1"'
            converted_sql = re.sub(pattern3, replacement3, converted_sql)
            
            # 处理 "数据库名.表名"."列名" 格式
            pattern4 = rf'"{re.escape(db_name)}\.([^"]+)"\."([^"]+)"'
            replacement4 = rf'"{alias}"."\1"."\2"'
            converted_sql = re.sub(pattern4, replacement4, converted_sql)
        
        # 验证表是否存在（在转换后验证）
        table_pattern = rf'"(db\d+)"\."([^"]+)"'
        tables_in_sql = re.findall(table_pattern, converted_sql)
        missing_tables = []
        for alias, table_name in tables_in_sql:
            if alias in db_tables_cache:
                if table_name not in db_tables_cache[alias]:
                    missing_tables.append(f"{alias}.{table_name}")
        
        if missing_tables:
            # 表不存在，返回失败
            conn.close()
            os.unlink(temp_db_path)
            return None, False
        
        # 执行SQL
        cursor.execute(converted_sql)
        results = cursor.fetchall()
        conn.close()
        
        # 清理临时文件
        os.unlink(temp_db_path)
        
        return results, True
        
    except Exception as e:
        # 清理临时文件
        if os.path.exists(temp_db_path):
            os.unlink(temp_db_path)
        return None, False

def load_multiple_schemas(database_names, database_dir):
    """加载多个数据库的schema信息"""
    schemas = {}
    for db_name in database_names:
        schema_file = os.path.join(database_dir, db_name, f"{db_name}.json")
        if os.path.exists(schema_file):
            schemas[db_name] = load_schema(schema_file)
        else:
            print(f"警告: 找不到schema文件 {schema_file}")
    return schemas

def load_cross_database_graph(graph_file):
    """加载跨数据库图文件"""
    with open(graph_file, 'r', encoding='utf-8') as f:
        graph_data = json.load(f)
    return graph_data

def extract_tables_from_cross_database_graph(graph_data, table_database_mapping):
    """从跨数据库图中提取相关表"""
    tables = set()
    for node in graph_data.get('nodes', []):
        node_type = node.get('node_type')
        if node_type == 'table':
            table_name = node.get('table_name', '')
            if table_name in table_database_mapping:
                # 格式：数据库名.表名
                db_name = table_database_mapping[table_name]
                tables.add(f"{db_name}.{table_name}")
    return list(tables)

def extract_columns_from_cross_database_graph(graph_data, table_database_mapping):
    """从跨数据库图中提取相关列"""
    columns_by_table = defaultdict(list)
    for node in graph_data.get('nodes', []):
        node_type = node.get('node_type')
        if node_type == 'column':
            table_name = node.get('table_name', '')
            column_name = node.get('column_name', '')
            if table_name in table_database_mapping:
                db_name = table_database_mapping[table_name]
                full_table_name = f"{db_name}.{table_name}"
                columns_by_table[full_table_name].append(column_name)
    return dict(columns_by_table)

def validate_tables_exist_in_databases(selected_tables, schemas, database_dir):
    """验证表是否在对应数据库中存在，只返回存在的表"""
    valid_tables = []
    for table_full_name in selected_tables:
        # 解析表名：格式为"数据库名.表名"
        parts = table_full_name.split('.', 1)
        if len(parts) == 2:
            db_name, table_name = parts
            # 检查表是否在schema中
            if db_name in schemas:
                schema = schemas[db_name]
                # 检查表是否在schema的tables列表中
                table_exists = False
                for table_info in schema.get('tables', []):
                    if table_info.get('table_name') == table_name:
                        table_exists = True
                        break
                
                # 如果schema中有，再验证数据库中是否真的存在
                if table_exists:
                    db_path = os.path.join(database_dir, db_name, f"{db_name}.db")
                    if os.path.exists(db_path):
                        tables_in_db = get_tables_in_database(db_path)
                        if table_name in tables_in_db:
                            valid_tables.append(table_full_name)
    return valid_tables

def get_all_tables_from_databases(schemas, database_dir, max_tables_per_db=50):
    """
    获取所有数据库中真实存在的表（限制数量以避免prompt过长）
    关键改进：直接从数据库文件查询真实表名，而不是从schema读取
    """
    all_tables = {}
    for db_name, schema in schemas.items():
        db_path = os.path.join(database_dir, db_name, f"{db_name}.db")
        if os.path.exists(db_path):
            # 直接从数据库文件查询真实表名（这是关键！）
            tables_in_db = get_tables_in_database(db_path)
            
            all_tables[db_name] = []
            count = 0
            
            # 只使用数据库中真实存在的表
            for table_name in tables_in_db:
                if count >= max_tables_per_db:
                    break
                
                # 从schema中查找对应的表信息（用于获取描述和列信息）
                table_info_from_schema = None
                for table_info in schema.get('tables', []):
                    if table_info.get('table_name') == table_name:
                        table_info_from_schema = table_info
                        break
                
                # 如果schema中没有，尝试查找相似的表名（可能表名有后缀数字）
                if table_info_from_schema is None:
                    # 尝试去掉后缀数字匹配
                    base_name = table_name.rsplit('-', 1)[0] if '-' in table_name else table_name
                    for table_info in schema.get('tables', []):
                        schema_table_name = table_info.get('table_name', '')
                        schema_base_name = schema_table_name.rsplit('-', 1)[0] if '-' in schema_table_name else schema_table_name
                        if base_name == schema_base_name:
                            table_info_from_schema = table_info
                            break
                
                # 获取列信息（从数据库直接查询，确保准确性）
                columns_in_db = []
                try:
                    conn = sqlite3.connect(db_path)
                    cursor = conn.cursor()
                    cursor.execute(f'PRAGMA table_info("{table_name}")')
                    columns_in_db = [row[1] for row in cursor.fetchall()]
                    conn.close()
                except:
                    # 如果查询失败，使用schema中的列信息
                    if table_info_from_schema:
                        columns_in_db = [col.get('column_name', '') for col in table_info_from_schema.get('columns', [])]
                
                all_tables[db_name].append({
                    'name': table_name,  # 使用数据库中真实存在的表名
                    'description': table_info_from_schema.get('table_description', '') if table_info_from_schema else '',
                    'comment': table_info_from_schema.get('table_comment', '') if table_info_from_schema else '',
                    'columns': columns_in_db[:15]  # 只显示前15个列，确保准确性
                })
                count += 1
    return all_tables

def extract_compact_graph_info(graph_data, table_database_mapping, schemas, max_tables=20, max_columns_per_table=10):
    """
    从图文件中提取压缩的关键信息，用于构建prompt
    
    只提取：
    1. 占位符相关的表和列（最相关的）
    2. 外键关系（用于判断是否可以JOIN）
    3. 表的简要信息（限制数量和列数）
    
    Args:
        graph_data: 完整的图数据
        table_database_mapping: 表到数据库的映射
        schemas: 数据库schema
        max_tables: 最多提取的表数量
        max_columns_per_table: 每个表最多显示的列数
    
    Returns:
        dict: {
            'suggested_tables': [...],  # 建议的表列表
            'foreign_keys': [...],      # 外键关系列表
            'table_info': {...}         # 表的简要信息
        }
    """
    # 1. 提取占位符相关的表（从table_database_mapping中提取）
    suggested_tables = []
    for table_name, db_name in table_database_mapping.items():
        table_full_name = f"{db_name}.{table_name}"
        suggested_tables.append(table_full_name)
    
    # 限制表数量
    suggested_tables = suggested_tables[:max_tables]
    
    # 2. 提取外键关系（只提取涉及建议表的外键）
    foreign_keys = []
    suggested_table_set = set(suggested_tables)
    
    for edge in graph_data.get('edges', []):
        if edge.get('edge_type') == 'foreign_key':
            source = edge.get('source', '')
            target = edge.get('target', '')
            
            # 检查是否涉及建议的表
            source_table = '.'.join(source.split('.')[:2]) if '.' in source else None
            target_table = '.'.join(target.split('.')[:2]) if '.' in target else None
            
            if source_table in suggested_table_set or target_table in suggested_table_set:
                foreign_keys.append({
                    'source': source,
                    'target': target
                })
    
    # 3. 提取表的简要信息（只提取建议的表，限制列数）
    table_info = {}
    for table_full_name in suggested_tables:
        parts = table_full_name.split('.', 1)
        if len(parts) == 2:
            db_name, table_name = parts
            if db_name in schemas:
                schema = schemas[db_name]
                for table_info_item in schema.get('tables', []):
                    if table_info_item.get('table_name') == table_name:
                        # 只提取前N个列
                        columns = table_info_item.get('columns', [])[:max_columns_per_table]
                        table_info[table_full_name] = {
                            'description': table_info_item.get('table_description', ''),
                            'comment': table_info_item.get('table_comment', ''),
                            'columns': [
                                {
                                    'name': col.get('column_name', ''),
                                    'type': col.get('data_type', 'TEXT')
                                }
                                for col in columns
                            ],
                            'total_columns': len(table_info_item.get('columns', []))
                        }
                        break
    
    return {
        'suggested_tables': suggested_tables,
        'foreign_keys': foreign_keys,
        'table_info': table_info
    }

def construct_cross_database_prompt(sql_skeleton, schemas, table_database_mapping, 
                                   graph_data, sql_analysis, database_dir):
    """构建跨数据库SQL填充的prompt - 优化版：使用压缩的图信息"""
    
    # 使用压缩的图信息提取方法（只提取关键信息）
    compact_info = extract_compact_graph_info(
        graph_data, 
        table_database_mapping, 
        schemas,
        max_tables=20,  # 最多20个建议的表
        max_columns_per_table=10  # 每个表最多10个列
    )
    
    suggested_tables = compact_info['suggested_tables']
    foreign_keys = compact_info['foreign_keys']
    table_info = compact_info['table_info']
    
    # 验证建议的表是否在对应数据库中存在
    valid_suggested_tables = validate_tables_exist_in_databases(suggested_tables, schemas, database_dir)
    
    # 获取所有数据库中真实存在的表（给大模型更多选择，但限制数量）
    all_available_tables = get_all_tables_from_databases(schemas, database_dir, max_tables_per_db=20)
    
    # 构建建议表的详细信息（只显示建议的表，压缩信息）
    suggested_tables_info = ""
    if valid_suggested_tables:
        suggested_tables_info = "\n建议使用的表（从SQL骨架分析得出，仅供参考，你可以选择其他更合适的表）：\n"
        for table_full_name in valid_suggested_tables[:15]:  # 最多显示15个
            if table_full_name in table_info:
                info = table_info[table_full_name]
                suggested_tables_info += f"\n  - {table_full_name}\n"
                if info.get('description'):
                    suggested_tables_info += f"    描述: {info['description'][:100]}...\n"  # 限制描述长度
                if info.get('comment'):
                    suggested_tables_info += f"    注释: {info['comment'][:100]}...\n"
                suggested_tables_info += f"    列（前{len(info['columns'])}个，共{info['total_columns']}个）:\n"
                for col in info['columns']:
                    suggested_tables_info += f"      - {col['name']} ({col['type']})\n"
    
    # 构建所有可用表的简要信息（只显示表名和关键信息，大幅压缩）
    # 重要：这里显示的表名都是从数据库文件直接查询的真实表名，确保准确性
    all_tables_info = ""
    for db_name, tables_list in all_available_tables.items():
        all_tables_info += f"\n数据库：{db_name}（共{len(tables_list)}个表，显示前15个，**这些表名都是从数据库直接查询的真实表名，确保存在**）\n"
        for table in tables_list[:15]:  # 每个数据库最多显示15个表
            table_name = table['name']
            # 显示表名和列数（让大模型知道表的结构）
            column_count = len(table.get('columns', []))
            all_tables_info += f"  - {db_name}.{table_name}（{column_count}个列）\n"
            # 显示前5个列名（帮助大模型选择正确的列）
            if table.get('columns'):
                all_tables_info += f"    列（前5个）: {', '.join(table['columns'][:5])}\n"
    
    # 构建外键关系信息（只显示涉及建议表的外键）
    fk_text = ""
    fk_count = 0
    for fk in foreign_keys[:20]:  # 最多显示20个外键关系
        source = fk.get('source', '')
        target = fk.get('target', '')
        if source and target:
            fk_text += f"  - {source} -> {target}\n"
            fk_count += 1
    
    # 检查是否有外键关系（用于决定是否建议UNION）
    has_foreign_keys = fk_count > 0
    
    # SQL骨架分析提示（强调可以简化）
    analysis_hints = ""
    analysis_hints += "\n**重要：你可以大幅简化SQL骨架，优先保证可执行性！**\n"
    
    if sql_analysis['has_join']:
        analysis_hints += "\n⚠️ **此SQL骨架包含JOIN操作，建议优先考虑改为UNION：**\n"
        analysis_hints += "  - 跨数据库JOIN通常难以执行成功（缺少外键关系）\n"
        analysis_hints += "  - **强烈建议改为UNION**：从不同数据库选择语义相关的表，分别查询后合并\n"
        analysis_hints += "  - UNION示例（简单易执行）：\n"
        analysis_hints += "    SELECT \"数据库1\".\"表1\".\"列1\", \"数据库1\".\"表1\".\"列2\" FROM \"数据库1\".\"表1\" WHERE \"数据库1\".\"表1\".\"列1\" IS NOT NULL\n"
        analysis_hints += "    UNION\n"
        analysis_hints += "    SELECT \"数据库2\".\"表2\".\"列1\", \"数据库2\".\"表2\".\"列2\" FROM \"数据库2\".\"表2\" WHERE \"数据库2\".\"表2\".\"列1\" IS NOT NULL\n"
        analysis_hints += "  - 如果必须使用JOIN，确保JOIN条件简单（如：ON 表1.列1 = 表2.列2），且列名真实存在\n"
    
    if sql_analysis['has_aggregate']:
        analysis_hints += "\n⚠️ **此SQL骨架包含聚合函数，建议简化：**\n"
        analysis_hints += "  - 如果GROUP BY/COUNT等聚合函数复杂，可以简化为简单的SELECT\n"
        analysis_hints += "  - 优先生成能执行成功的简单查询，而不是复杂的聚合查询\n"
    
    if sql_analysis['has_subquery']:
        analysis_hints += "\n⚠️ **此SQL骨架包含子查询，建议简化：**\n"
        analysis_hints += "  - 子查询在跨数据库场景下难以执行\n"
        analysis_hints += "  - **建议改为简单的SELECT或UNION**，避免子查询\n"
    
    # 通用建议
    analysis_hints += "\n**通用建议：**\n"
    analysis_hints += "  - 优先使用UNION而不是JOIN\n"
    analysis_hints += "  - 使用简单的WHERE条件（IS NOT NULL, = '值'等）\n"
    analysis_hints += "  - 选择有数据的表，确保查询能返回结果\n"
    analysis_hints += "  - 如果骨架太复杂，可以完全简化为：SELECT 列 FROM 表 WHERE 条件\n"
    
    # 构建完整prompt
    databases_str = ', '.join(schemas.keys())
    
    prompt = f"""请根据以下 SQL 框架和数据库信息，生成完整且可在 SQLite 上正确执行的跨数据库 SQL 语句。

**重要：这是一个跨数据库查询，涉及以下数据库：{databases_str}**

**核心原则：优先生成简单、可执行的SQL，可以大幅简化SQL骨架！**

**重要提示：你可以根据实际情况大幅修改和简化SQL骨架，使其更容易执行成功：**
1. **可以完全改变SQL结构**：
   - 如果骨架是复杂的JOIN，可以改为简单的UNION
   - 如果骨架有子查询，可以简化为单层查询
   - 如果骨架有聚合函数，可以改为简单的SELECT
   - **目标是生成能成功执行的SQL，而不是完全遵循骨架结构**

2. **优先使用UNION方式**（推荐）：
   - UNION比JOIN更容易执行成功
   - 从不同数据库选择语义相关的表，分别查询后合并
   - 确保每个SELECT的列数和类型兼容
   - 示例：SELECT "数据库1"."表1"."列1" FROM "数据库1"."表1" WHERE ... UNION SELECT "数据库2"."表2"."列1" FROM "数据库2"."表2" WHERE ...

3. **如果必须使用JOIN**：
   - 确保JOIN条件使用真实存在的列名
   - 优先使用简单的等值JOIN（如：ON 表1.列1 = 表2.列2）
   - 如果JOIN困难，立即改为UNION

4. **简化查询条件**：
   - 使用简单的WHERE条件（如：IS NOT NULL, = '值'）
   - 避免复杂的子查询
   - 避免复杂的聚合函数

5. **表的选择**：
   - 优先选择有数据的表（从"所有可用表"中选择）
   - 如果建议的表不合适，可以自由选择其他表
   - 确保选择的表在对应数据库中真实存在

**严格要求：**
- **仅输出最终生成的完整 SQL 语句，不要重复提示内容。**
- **生成的 SQL 要保证语法正确，可以直接在 SQLite 上运行得到结果。**
- **不要添加任何额外的解释、注释或输出格式（代码块，空格等）。**
- **⚠️ 关键：必须使用下面"所有可用表"中列出的真实表名和列名，这些表名和列名都是从数据库直接查询的，确保存在！**
- **⚠️ 绝对不要使用schema中可能存在但数据库中不存在的表名或列名！**
- **所有表名必须使用格式："数据库名"."表名"，例如："企业服务"."市市场监管局-市场主体注册情况-1820"**
- **所有列名必须使用格式："数据库名"."表名"."列名"，例如："企业服务"."市市场监管局-市场主体注册情况-1820"."市场主体名称"**
- **所有表名和列名都必须用双引号包裹（包括中文和特殊字符）。**
- **⚠️ 表名必须完全匹配"所有可用表"中列出的表名（包括后缀数字），不能省略或修改！**

**SQL 框架（仅供参考，你可以大幅简化或完全改变结构，优先保证可执行性）：**
{sql_skeleton}

{suggested_tables_info}

所有可用表（你可以从中选择最合适的表）：
{all_tables_info}

外键关系：
{fk_text if fk_text else "⚠️ 无外键关系 - 跨数据库查询可能没有外键关系。如果JOIN困难，建议改用UNION方式合并不同数据库的查询结果。"}

{analysis_hints}

请生成完整的跨数据库SQL语句（可以根据实际情况优化和调整SQL骨架）："""

    # 返回所有可用表（用于后续验证）
    all_tables_list = []
    for db_name, tables_list in all_available_tables.items():
        for table in tables_list:
            all_tables_list.append(f"{db_name}.{table['name']}")
    
    return prompt, all_tables_list, {}

def process_cross_database_skeleton(skeleton_data, schemas, graph_dir, output_dir, 
                                    database_dir, max_retries=3):
    """处理单个跨数据库SQL骨架，填充生成完整SQL"""
    
    sql_skeleton = skeleton_data['sql_skeleton']
    table_database_mapping = skeleton_data['table_database_mapping']
    databases = skeleton_data.get('databases', [])
    
    # 确定输出文件名
    original_file = skeleton_data.get('original_file', 'unknown')
    match = re.search(r'(\d+)', original_file)
    if match:
        idx = match.group(1)
        output_file = os.path.join(output_dir, f"cross_db_generated_sql_{idx}.json")
    else:
        import hashlib
        hash_id = hashlib.md5(sql_skeleton.encode()).hexdigest()[:8]
        output_file = os.path.join(output_dir, f"cross_db_generated_sql_{hash_id}.json")
    
    # 检查是否已存在（修改后重新生成，所以不跳过）
    # if os.path.exists(output_file):
    #     return idx if match else hash_id, True, "已存在"
    
    # 加载图文件
    graph_file = os.path.join(graph_dir, f"cross_db_graph_{idx if match else hash_id}.json")
    if not os.path.exists(graph_file):
        return idx if match else hash_id, False, "图文件不存在"
    
    graph_data = load_cross_database_graph(graph_file)
    
    # 分析SQL骨架
    sql_analysis = analyze_sql_skeleton(sql_skeleton)
    
    # 构建prompt
    prompt, selected_tables, selected_columns = construct_cross_database_prompt(
        sql_skeleton, schemas, table_database_mapping, graph_data, sql_analysis, database_dir
    )
    
    # 不再跳过，即使没有建议的表，也提供所有可用表给大模型选择
    if prompt is None:
        # 如果构建失败，使用备用方案：提供所有可用表
        all_available_tables = get_all_tables_from_databases(schemas, database_dir)
        if not any(all_available_tables.values()):
            return None, False, "数据库中没有可用的表"
        # 重新构建prompt，使用所有可用表
        prompt, _, _ = construct_cross_database_prompt(
            sql_skeleton, schemas, table_database_mapping, graph_data, sql_analysis, database_dir
        )
    
    # 调用大模型生成SQL
    client = get_client()
    API_CONFIG = load_config()
    
    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model=API_CONFIG.get("model", "gpt-4o"),
                messages=[
                    {"role": "system", "content": "你是一个SQL专家，擅长生成跨数据库SQL查询。"},
                    {"role": "user", "content": prompt}
                ],
                temperature=API_CONFIG.get("temperature", 0.1),
                max_tokens=API_CONFIG.get("max_tokens", 8000)
            )
            
            generated_sql = response.choices[0].message.content.strip()
            
            # 清理SQL（移除代码块标记等）
            generated_sql = re.sub(r'^```sql\s*', '', generated_sql, flags=re.IGNORECASE)
            generated_sql = re.sub(r'^```\s*', '', generated_sql)
            generated_sql = re.sub(r'```\s*$', '', generated_sql)
            generated_sql = generated_sql.strip()
            
            # 验证SQL语法
            try:
                sqlparse.parse(generated_sql)
            except:
                if attempt < max_retries - 1:
                    continue
                return idx if match else hash_id, False, "SQL语法错误"
            
            # 执行SQL并获取结果
            # 使用ATTACH DATABASE功能执行跨数据库SQL
            results = None
            execution_error = None
            
            try:
                # 方法1：使用ATTACH DATABASE执行真正的跨数据库查询
                results, success = execute_cross_database_sql_with_attach(
                    generated_sql, databases, database_dir, table_database_mapping
                )
                
                if not success:
                    # 方法2：如果ATTACH失败，尝试转换为单数据库格式执行（降级方案）
                    single_db_sql = convert_to_single_database_sql(generated_sql, table_database_mapping)
                    
                    # 尝试在涉及的数据库上执行（优先第一个数据库）
                    for db_name in databases:
                        db_path = os.path.join(database_dir, db_name, f"{db_name}.db")
                        if os.path.exists(db_path):
                            results, success = execute_sql_on_database(single_db_sql, db_path)
                            if success:
                                break  # 成功执行，退出循环
                    
                    # 如果所有方法都失败，记录错误
                    if not success:
                        execution_error = "无法在任意数据库上执行SQL（ATTACH和单数据库格式都失败）"
                        results = None
                else:
                    # ATTACH成功，results可能是空列表（空结果也算成功）
                    if results is None:
                        results = []
            except Exception as e:
                execution_error = f"执行异常: {str(e)}"
            
            # 保存结果（限制结果数量，与单数据库保持一致）
            saved_results = []
            if results is not None:
                # 只保存前10条结果（与单数据库保持一致）
                saved_results = results[:10] if len(results) > 10 else results
                # 转换为列表格式（确保可以JSON序列化）
                saved_results = [list(row) for row in saved_results]
            
            # 保存结果
            result = {
                'sql': generated_sql,
                'results': saved_results,
                'sql_skeleton': sql_skeleton,
                'databases': databases,
                'table_database_mapping': table_database_mapping,
                'tables': selected_tables,
                'columns': selected_columns,
                'metadata': {
                    'has_join': sql_analysis['has_join'],
                    'has_subquery': sql_analysis['has_subquery'],
                    'has_aggregate': sql_analysis['has_aggregate'],
                    'is_cross_database': True,
                    'num_databases': len(databases)
                },
                'generation_info': {
                    'model': API_CONFIG.get("model", "gpt-4o"),
                    'attempt': attempt + 1
                }
            }
            
            # 如果有执行错误，记录到metadata中
            if execution_error:
                result['metadata']['execution_error'] = execution_error
            
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(result, f, ensure_ascii=False, indent=2)
            
            return idx if match else hash_id, True, "成功"
            
        except Exception as e:
            if attempt < max_retries - 1:
                time.sleep(1)
                continue
            return idx if match else hash_id, False, f"生成失败: {str(e)}"
    
    return idx if match else hash_id, False, "达到最大重试次数"

def main():
    parser = argparse.ArgumentParser(description='填充跨数据库SQL骨架')
    parser.add_argument('--skeleton_file', type=str, required=True,
                       help='跨数据库SQL骨架文件')
    parser.add_argument('--graph_dir', type=str,
                       default='benchmark/data/beijing/output/cross_db_graph',
                       help='图文件目录')
    parser.add_argument('--database_dir', type=str,
                       default='benchmark/data/beijing/database_chinese',
                       help='数据库目录')
    parser.add_argument('--output_dir', type=str,
                       default='benchmark/data/beijing/output/cross_db_single',
                       help='输出目录')
    parser.add_argument('--max_retries', type=int, default=3,
                       help='最大重试次数')
    parser.add_argument('--max_workers', type=int, default=5,
                       help='并发线程数')
    
    args = parser.parse_args()
    
    # 加载跨数据库SQL骨架
    print(f"加载跨数据库SQL骨架: {args.skeleton_file}")
    with open(args.skeleton_file, 'r', encoding='utf-8') as f:
        skeletons = json.load(f)
    
    print(f"共 {len(skeletons)} 个SQL骨架")
    
    # 获取所有涉及的数据库
    all_databases = set()
    for skeleton in skeletons:
        all_databases.update(skeleton.get('databases', []))
    
    print(f"涉及的数据库: {sorted(all_databases)}")
    
    # 加载所有数据库的schema
    print("\n加载数据库schema...")
    schemas = load_multiple_schemas(all_databases, args.database_dir)
    print(f"成功加载 {len(schemas)} 个数据库的schema")
    
    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 处理每个SQL骨架
    print(f"\n填充SQL骨架...")
    success_count = 0
    failed_count = 0
    
    # 使用线程池并发处理
    with ThreadPoolExecutor(max_workers=args.max_workers) as executor:
        futures = []
        for skeleton in skeletons:
            future = executor.submit(
                process_cross_database_skeleton,
                skeleton, schemas, args.graph_dir, args.output_dir,
                args.database_dir, args.max_retries
            )
            futures.append(future)
        
        # 收集结果
        for future in tqdm(as_completed(futures), total=len(futures), desc="填充进度"):
            idx, success, message = future.result()
            if success:
                success_count += 1
            else:
                failed_count += 1
                if failed_count <= 10:  # 只显示前10个错误
                    print(f"\n失败 (idx={idx}): {message}")
    
    print(f"\n完成！")
    print(f"成功: {success_count}/{len(skeletons)}")
    print(f"失败: {failed_count}/{len(skeletons)}")
    print(f"输出目录: {args.output_dir}")

if __name__ == '__main__':
    main()

