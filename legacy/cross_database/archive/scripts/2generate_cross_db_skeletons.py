#!/usr/bin/env python3
"""
Convert single-database SQL into cross-database SQL skeletons.
"""

import os
import json
import re
import argparse
import random
from collections import defaultdict

def extract_tables_from_sql(sql):
    """Extract table names from SQL."""
    # Simple table name extraction (from FROM and JOIN clauses)
    tables = []
    
    # Extract table after FROM
    from_match = re.search(r'FROM\s+["\']?(\w+)["\']?', sql, re.IGNORECASE)
    if from_match:
        tables.append(from_match.group(1))
    
    # Extract tables after JOIN
    join_matches = re.finditer(r'JOIN\s+["\']?(\w+)["\']?', sql, re.IGNORECASE)
    for match in join_matches:
        tables.append(match.group(1))
    
    return list(set(tables))  # Deduplicate

def assign_tables_to_databases(tables, target_databases, strategy='round_robin'):
    """Assign tables to different databases."""
    table_db_mapping = {}
    
    if strategy == 'round_robin':
        # Round-robin assignment, ensuring at least 2 databases are used
        for i, table in enumerate(tables):
            db_idx = i % min(len(target_databases), len(tables))
            table_db_mapping[table] = target_databases[db_idx]
    elif strategy == 'random':
        # Random assignment, ensuring at least 2 databases are used
        for i, table in enumerate(tables):
            if i < 2:
                # Assign first 2 tables to different databases
                db_idx = i % len(target_databases)
            else:
                # Randomly assign subsequent tables
                db_idx = random.randint(0, len(target_databases) - 1)
            table_db_mapping[table] = target_databases[db_idx]
    
    return table_db_mapping

def convert_to_cross_database_sql(original_sql, table_db_mapping):
    """Convert single-database SQL to cross-database SQL (add database prefixes)."""
    # Note: this step only records the cross-database SQL format; actual filling is done by the LLM later
    converted_sql = original_sql
    
    # Sort by table name length descending to avoid short names being matched inside long names
    sorted_tables = sorted(table_db_mapping.keys(), key=len, reverse=True)
    
    for table in sorted_tables:
        db = table_db_mapping[table]
        # Replace table name: table -> database.table
        # Handle quoted table names
        patterns = [
            rf'\b{table}\b',  # Plain table name
            rf'"{table}"',    # Double quotes
            rf"'{table}'",    # Single quotes
        ]
        
        for pattern in patterns:
            replacement = f'{db}.{table}'
            converted_sql = re.sub(pattern, replacement, converted_sql, flags=re.IGNORECASE)
    
    return converted_sql

def convert_to_skeleton(sql):
    """Convert SQL to a skeleton (using the original simple logic, without database prefixes)."""
    # Use the original simple skeleton generation logic
    sql_skeleton = re.sub(r"'[^']*'", '_', sql)
    sql_skeleton = re.sub(r'"[^"]*"', '_', sql_skeleton)
    sql_skeleton = re.sub(r'\b\d+\b', '_', sql_skeleton)
    sql_keywords = set(['SELECT', 'FROM', 'WHERE', 'GROUP', 'BY', 'HAVING', 'ORDER', 'LIMIT', 'JOIN', 'ON', 'AS', 'AND', 'OR', 'IN', 'NOT', 'NULL', 'IS', 'DISTINCT', 'COUNT', 'SUM', 'AVG', 'MIN', 'MAX', 'INNER', 'LEFT', 'RIGHT', 'FULL', 'OUTER', 'UNION', 'ALL'])
    def replace_identifier(match):
        word = match.group(0)
        if word.upper() in sql_keywords or word == '*':
            return word
        else:
            return '_'
    sql_skeleton = re.sub(r'\b\w+\b', replace_identifier, sql_skeleton)
    sql_skeleton = re.sub(r'(_\s*)+', '_ ', sql_skeleton)
    sql_skeleton = ' '.join(sql_skeleton.strip().split())
    return sql_skeleton

def generate_cross_database_skeletons(candidates_file, target_databases, output_file):
    """Generate cross-database SQL skeletons (simplified: skeletons omit database prefixes; filling handles them later)."""
    # Load candidate SQL
    with open(candidates_file, 'r', encoding='utf-8') as f:
        candidates = json.load(f)
    
    cross_db_skeletons = []
    
    for candidate in candidates:
        original_sql = candidate['sql']
        tables = candidate['tables']
        
        # Assign tables to databases
        table_db_mapping = assign_tables_to_databases(tables, target_databases)
        
        # Generate skeleton from original SQL (without database prefixes)
        skeleton = convert_to_skeleton(original_sql)
        
        # Record cross-database metadata (for use during later filling)
        cross_db_skeletons.append({
            'original_sql': original_sql,
            'original_database': candidate['database'],
            'original_file': candidate['file'],
            'sql_skeleton': skeleton,  # Plain skeleton without database prefixes
            'databases': list(set(table_db_mapping.values())),  # List of involved databases
            'table_database_mapping': table_db_mapping,  # Table-to-database mapping
            'tables': tables,
            'table_count': len(tables),
            'is_cross_database': True,  # Mark as cross-database query
            'num_databases': len(set(table_db_mapping.values()))  # Number of involved databases
        })
    
    # Save
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(cross_db_skeletons, f, ensure_ascii=False, indent=2)
    
    print(f"Generated {len(cross_db_skeletons)} cross-database SQL skeletons")
    
    # Statistics
    db_count_dist = defaultdict(int)
    for skeleton in cross_db_skeletons:
        db_count = skeleton['num_databases']
        db_count_dist[db_count] += 1
    
    print(f"\nCross-database count distribution:")
    for db_count in sorted(db_count_dist.keys()):
        print(f"  {db_count} databases: {db_count_dist[db_count]}")

def main():
    parser = argparse.ArgumentParser(description='Generate cross-database SQL skeletons')
    parser.add_argument('--candidates_file', type=str,
                       default='benchmark/generation/cross_database/candidates_2db.json',
                       help='Candidate SQL file')
    parser.add_argument('--target_databases', type=str, nargs='+',
                       required=True,
                       help='Target database list (e.g.: enterprise services social security)')
    parser.add_argument('--output_file', type=str,
                       default='benchmark/generation/cross_database/cross_db_skeletons_2db.json',
                       help='Output file')
    parser.add_argument('--strategy', type=str, default='round_robin',
                       choices=['round_robin', 'random'],
                       help='Table assignment strategy')
    
    args = parser.parse_args()
    
    print("=" * 70)
    print("Generating cross-database SQL skeletons")
    print("=" * 70)
    print(f"\nCandidates file: {args.candidates_file}")
    print(f"Target databases: {args.target_databases}")
    print(f"Output file: {args.output_file}")
    print(f"Assignment strategy: {args.strategy}")
    print()
    
    generate_cross_database_skeletons(
        args.candidates_file,
        args.target_databases,
        args.output_file
    )
    
    print("\n" + "=" * 70)
    print("Done!")
    print("=" * 70)

if __name__ == '__main__':
    main()
