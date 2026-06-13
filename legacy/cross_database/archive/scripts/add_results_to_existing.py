#!/usr/bin/env python3
"""
Add execution results to existing cross-database SQL result files.
"""

import json
import os
import sqlite3
import re
from tqdm import tqdm
import sys

# Import conversion and execution functions
sys.path.insert(0, os.path.dirname(__file__))
from cross_db_2fill_sql_placeholders import convert_to_single_database_sql, execute_sql_on_database

def add_results_to_file(file_path, database_dir):
    """Add execution results to a single file."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Skip if results field already exists and is non-empty
        if 'results' in data and data['results']:
            return True, "Results already present"
        
        sql = data.get('sql', '')
        databases = data.get('databases', [])
        table_database_mapping = data.get('table_database_mapping', {})
        
        if not sql or not databases:
            return False, "Missing required information"
        
        # Convert to single-database format
        single_db_sql = convert_to_single_database_sql(sql, table_database_mapping)
        
        # Try executing on each involved database
        results = None
        execution_error = None
        
        for db_name in databases:
            db_path = os.path.join(database_dir, db_name, f"{db_name}.db")
            if os.path.exists(db_path):
                results, success = execute_sql_on_database(single_db_sql, db_path)
                if success and results is not None:
                    break
        
        # Save results (limit count)
        saved_results = []
        if results is not None:
            saved_results = results[:10] if len(results) > 10 else results
            saved_results = [list(row) for row in saved_results]
        
        # Update data
        data['results'] = saved_results
        if execution_error:
            if 'metadata' not in data:
                data['metadata'] = {}
            data['metadata']['execution_error'] = execution_error
        
        # Save
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        
        return True, f"Success, result count: {len(saved_results)}"
        
    except Exception as e:
        return False, f"Error: {str(e)}"

def main():
    sql_dir = "benchmark/data/beijing/output/cross_db_single"
    database_dir = "benchmark/data/beijing/database_chinese"
    
    if not os.path.exists(sql_dir):
        print(f"Directory does not exist: {sql_dir}")
        return
    
    sql_files = sorted([f for f in os.listdir(sql_dir) 
                       if f.startswith('cross_db_generated_sql_') and f.endswith('.json')])
    
    print(f"Found {len(sql_files)} SQL result files")
    print(f"Adding execution results...\n")
    
    success_count = 0
    failed_count = 0
    
    for sql_file in tqdm(sql_files, desc="Processing"):
        file_path = os.path.join(sql_dir, sql_file)
        success, message = add_results_to_file(file_path, database_dir)
        
        if success:
            success_count += 1
        else:
            failed_count += 1
            if failed_count <= 5:  # Show only first 5 errors
                print(f"\nFailed: {sql_file} - {message}")
    
    print(f"\nDone!")
    print(f"Success: {success_count}/{len(sql_files)}")
    print(f"Failed: {failed_count}/{len(sql_files)}")

if __name__ == '__main__':
    main()
