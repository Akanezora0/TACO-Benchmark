"""
Schema formatting utilities

Format schema based on experiment setting (full schema or filtered schema)
"""

from typing import Dict, List, Tuple, Optional


def format_schema_simple(
    schema: Dict, 
    max_tables: Optional[int] = None, 
    max_columns_per_table: Optional[int] = None
) -> Tuple[str, Dict]:
    """
    Format full schema (for Origin and QR settings)
    
    Args:
        schema: Schema dictionary
        max_tables: Maximum number of tables (None means include all tables)
        max_columns_per_table: Maximum columns per table (None means include all columns)
        
    Returns:
        (Formatted schema text, configuration info dictionary)
    """
    all_tables = schema.get('tables', [])
    
    # If max_tables is not specified, include all tables
    if max_tables is None:
        selected_tables = all_tables
    else:
        selected_tables = all_tables[:max_tables]
    
    # Format schema text
    text = "Database Schema Information:\n\n"
    
    total_tables = len(selected_tables)
    total_columns = 0
    
    for table in selected_tables:
        table_name = table.get('table_name', '')
        columns = table.get('columns', [])
        
        # If max_columns_per_table is not specified, include all columns
        if max_columns_per_table is not None:
            columns = columns[:max_columns_per_table]
        total_columns += len(columns)
        
        text += f"Table: {table_name}\n"
        text += "  Columns:\n"
        for col in columns:
            col_name = col.get('column_name', '')
            col_type = col.get('data_type', 'TEXT')
            text += f"    - {col_name} ({col_type})\n"
        text += "\n"
    
    # Record configuration info
    config_info = {
        'total_tables_in_schema': len(all_tables),
        'included_tables_count': total_tables,
        'included_columns_count': total_columns,
        'max_tables': max_tables,
        'max_columns_per_table': max_columns_per_table,
        'schema_text_length': len(text),
        'estimated_tokens': len(text) // 4  # Rough estimate
    }
    
    return text, config_info


def format_schema_filtered(
    schema: Dict, 
    relevant_tables: List[str],
    max_columns_per_table: Optional[int] = None
) -> Tuple[str, Dict]:
    """
    Format filtered schema (for QR+TL and QR+TL+QP settings)
    
    Args:
        schema: Schema dictionary
        relevant_tables: List of relevant tables (from Table Linking)
        max_columns_per_table: Maximum columns per table
        
    Returns:
        (Formatted schema text, configuration info dictionary)
    """
    all_tables = schema.get('tables', [])
    
    # Filter relevant tables
    selected_tables = [
        table for table in all_tables 
        if table.get('table_name', '') in relevant_tables
    ]
    
    # Format schema text
    text = "Relevant Tables Schema Information:\n\n"
    
    total_columns = 0
    
    for table in selected_tables:
        table_name = table.get('table_name', '')
        columns = table.get('columns', [])
        
        if max_columns_per_table is not None:
            columns = columns[:max_columns_per_table]
        total_columns += len(columns)
        
        text += f"Table: {table_name}\n"
        text += "  Columns:\n"
        for col in columns:
            col_name = col.get('column_name', '')
            col_type = col.get('data_type', 'TEXT')
            text += f"    - {col_name} ({col_type})\n"
        text += "\n"
    
    # Record configuration info
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
    Format schema information for Query Planning
    
    Args:
        schema: Schema dictionary
        tables: Table list
        
    Returns:
        Formatted schema dictionary (for JSON output)
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


# Example usage
if __name__ == "__main__":
    # Example schema
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
    
    # Format full schema
    full_schema_text, full_config = format_schema_simple(example_schema)
    print("Full schema:")
    print(full_schema_text)
    print(f"Configuration info: {full_config}")
    print("\n" + "="*80 + "\n")
    
    # Format filtered schema
    relevant_tables = ['企业注册表']
    filtered_schema_text, filtered_config = format_schema_filtered(
        example_schema, 
        relevant_tables
    )
    print("Filtered schema:")
    print(filtered_schema_text)
    print(f"Configuration info: {filtered_config}")
