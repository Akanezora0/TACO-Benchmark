#!/usr/bin/env python3
"""
为已生成的跨数据库SQL结果文件添加执行结果
"""

import json
import os
import sqlite3
import re
from tqdm import tqdm
import sys

# 导入转换和执行函数
sys.path.insert(0, os.path.dirname(__file__))
from cross_db_2fill_sql_placeholders import convert_to_single_database_sql, execute_sql_on_database

def add_results_to_file(file_path, database_dir):
    """为单个文件添加执行结果"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # 如果已经有results字段且不为空，跳过
        if 'results' in data and data['results']:
            return True, "已有结果"
        
        sql = data.get('sql', '')
        databases = data.get('databases', [])
        table_database_mapping = data.get('table_database_mapping', {})
        
        if not sql or not databases:
            return False, "缺少必要信息"
        
        # 转换为单数据库格式
        single_db_sql = convert_to_single_database_sql(sql, table_database_mapping)
        
        # 尝试在涉及的数据库上执行
        results = None
        execution_error = None
        
        for db_name in databases:
            db_path = os.path.join(database_dir, db_name, f"{db_name}.db")
            if os.path.exists(db_path):
                results, success = execute_sql_on_database(single_db_sql, db_path)
                if success and results is not None:
                    break
        
        # 保存结果（限制数量）
        saved_results = []
        if results is not None:
            saved_results = results[:10] if len(results) > 10 else results
            saved_results = [list(row) for row in saved_results]
        
        # 更新数据
        data['results'] = saved_results
        if execution_error:
            if 'metadata' not in data:
                data['metadata'] = {}
            data['metadata']['execution_error'] = execution_error
        
        # 保存
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        
        return True, f"成功，结果数量: {len(saved_results)}"
        
    except Exception as e:
        return False, f"错误: {str(e)}"

def main():
    sql_dir = "benchmark/data/beijing/output/cross_db_single"
    database_dir = "benchmark/data/beijing/database_chinese"
    
    if not os.path.exists(sql_dir):
        print(f"目录不存在: {sql_dir}")
        return
    
    sql_files = sorted([f for f in os.listdir(sql_dir) 
                       if f.startswith('cross_db_generated_sql_') and f.endswith('.json')])
    
    print(f"找到 {len(sql_files)} 个SQL结果文件")
    print(f"开始添加执行结果...\n")
    
    success_count = 0
    failed_count = 0
    
    for sql_file in tqdm(sql_files, desc="处理进度"):
        file_path = os.path.join(sql_dir, sql_file)
        success, message = add_results_to_file(file_path, database_dir)
        
        if success:
            success_count += 1
        else:
            failed_count += 1
            if failed_count <= 5:  # 只显示前5个错误
                print(f"\n失败: {sql_file} - {message}")
    
    print(f"\n完成！")
    print(f"成功: {success_count}/{len(sql_files)}")
    print(f"失败: {failed_count}/{len(sql_files)}")

if __name__ == '__main__':
    main()


