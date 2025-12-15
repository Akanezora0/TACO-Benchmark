"""
Schema格式化工具

根据实验设置格式化Schema（完整Schema或过滤后的Schema）
"""

from typing import Dict, List, Tuple, Optional


def format_schema_simple(
    schema: Dict, 
    max_tables: Optional[int] = None, 
    max_columns_per_table: Optional[int] = None
) -> Tuple[str, Dict]:
    """
    格式化完整Schema（用于Origin和QR设置）
    
    Args:
        schema: Schema字典
        max_tables: 最大表数（None表示包含所有表）
        max_columns_per_table: 每表最大列数（None表示包含所有列）
        
    Returns:
        (格式化的Schema文本, 配置信息字典)
    """
    all_tables = schema.get('tables', [])
    
    # 如果未指定max_tables，则包含所有表
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


def format_schema_filtered(
    schema: Dict, 
    relevant_tables: List[str],
    max_columns_per_table: Optional[int] = None
) -> Tuple[str, Dict]:
    """
    格式化过滤后的Schema（用于QR+TL和QR+TL+QP设置）
    
    Args:
        schema: Schema字典
        relevant_tables: 相关表列表（来自Table Linking）
        max_columns_per_table: 每表最大列数
        
    Returns:
        (格式化的Schema文本, 配置信息字典)
    """
    all_tables = schema.get('tables', [])
    
    # 筛选相关表
    selected_tables = [
        table for table in all_tables 
        if table.get('table_name', '') in relevant_tables
    ]
    
    # 格式化Schema文本
    text = "相关表Schema信息：\n\n"
    
    total_columns = 0
    
    for table in selected_tables:
        table_name = table.get('table_name', '')
        columns = table.get('columns', [])
        
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
        'relevant_tables_count': len(relevant_tables),
        'included_tables_count': len(selected_tables),
        'included_columns_count': total_columns,
        'max_columns_per_table': max_columns_per_table,
        'schema_text_length': len(text),
        'estimated_tokens': len(text) // 4
    }
    
    return text, config_info


def format_schema_for_planning(
    schema: Dict,
    tables: List[str]
) -> Dict:
    """
    为Query Planning格式化Schema信息
    
    Args:
        schema: Schema字典
        tables: 表列表
        
    Returns:
        格式化的Schema字典（用于JSON输出）
    """
    all_tables = schema.get('tables', [])
    
    planning_schema = {}
    
    for table in all_tables:
        table_name = table.get('table_name', '')
        if table_name in tables:
            planning_schema[table_name] = {
                'columns': [
                    {
                        'name': col.get('column_name', ''),
                        'type': col.get('data_type', 'TEXT')
                    }
                    for col in table.get('columns', [])
                ],
                'description': table.get('description', '')
            }
    
    return planning_schema


# 示例使用
if __name__ == "__main__":
    # 示例Schema
    example_schema = {
        'tables': [
            {
                'table_name': '企业注册表',
                'columns': [
                    {'column_name': '企业名称', 'data_type': 'TEXT'},
                    {'column_name': '注册时间', 'data_type': 'DATE'},
                    {'column_name': '注册资本', 'data_type': 'INTEGER'}
                ]
            },
            {
                'table_name': '企业信息表',
                'columns': [
                    {'column_name': '企业名称', 'data_type': 'TEXT'},
                    {'column_name': '行业类型', 'data_type': 'TEXT'}
                ]
            }
        ]
    }
    
    # 格式化完整Schema
    full_schema_text, full_config = format_schema_simple(example_schema)
    print("完整Schema：")
    print(full_schema_text)
    print(f"配置信息：{full_config}")
    print("\n" + "="*80 + "\n")
    
    # 格式化过滤后的Schema
    relevant_tables = ['企业注册表']
    filtered_schema_text, filtered_config = format_schema_filtered(
        example_schema, 
        relevant_tables
    )
    print("过滤后的Schema：")
    print(filtered_schema_text)
    print(f"配置信息：{filtered_config}")

