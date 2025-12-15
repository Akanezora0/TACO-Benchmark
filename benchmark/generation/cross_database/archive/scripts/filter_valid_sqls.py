#!/usr/bin/env python3
"""
过滤掉无法执行的跨数据库SQL，只保留有执行结果的
"""

import json
import os
import shutil
from collections import defaultdict

def filter_valid_sqls(sql_dir, output_dir=None):
    """
    过滤有效的SQL文件（有执行结果的）
    
    Args:
        sql_dir: SQL结果目录
        output_dir: 输出目录（如果为None，则覆盖原文件）
    """
    if not os.path.exists(sql_dir):
        print(f"目录不存在: {sql_dir}")
        return
    
    sql_files = sorted([f for f in os.listdir(sql_dir) 
                       if f.startswith('cross_db_generated_sql_') and f.endswith('.json')])
    
    print(f"=" * 70)
    print(f"过滤跨数据库SQL结果")
    print(f"=" * 70)
    print(f"\n总文件数: {len(sql_files)}")
    
    valid_files = []
    invalid_files = []
    
    for sql_file in sql_files:
        file_path = os.path.join(sql_dir, sql_file)
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            results = data.get('results', [])
            error = data.get('metadata', {}).get('execution_error')
            
            # 判断是否有效：有结果且无错误
            if len(results) > 0 and not error:
                valid_files.append(sql_file)
            else:
                invalid_files.append(sql_file)
        except Exception as e:
            print(f"读取文件失败 {sql_file}: {e}")
            invalid_files.append(sql_file)
    
    print(f"\n有效文件（有执行结果）: {len(valid_files)} ({len(valid_files)/len(sql_files)*100:.1f}%)")
    print(f"无效文件（无执行结果）: {len(invalid_files)} ({len(invalid_files)/len(sql_files)*100:.1f}%)")
    
    if output_dir:
        # 复制有效文件到新目录
        os.makedirs(output_dir, exist_ok=True)
        for sql_file in valid_files:
            src = os.path.join(sql_dir, sql_file)
            dst = os.path.join(output_dir, sql_file)
            shutil.copy2(src, dst)
        print(f"\n有效文件已复制到: {output_dir}")
    else:
        # 删除无效文件
        print(f"\n删除无效文件...")
        for sql_file in invalid_files:
            file_path = os.path.join(sql_dir, sql_file)
            os.remove(file_path)
        print(f"已删除 {len(invalid_files)} 个无效文件")
        print(f"保留 {len(valid_files)} 个有效文件")
    
    # 重新编号（可选）
    if output_dir or len(invalid_files) > 0:
        print(f"\n重新编号文件...")
        valid_files_sorted = sorted(valid_files, key=lambda x: int(re.search(r'(\d+)', x).group(1)) if re.search(r'(\d+)', x) else 0)
        
        for i, sql_file in enumerate(valid_files_sorted):
            old_path = os.path.join(sql_dir, sql_file)
            new_name = f"cross_db_generated_sql_{i}.json"
            new_path = os.path.join(sql_dir, new_name)
            
            if old_path != new_path:
                os.rename(old_path, new_path)
        
        print(f"已重新编号 {len(valid_files_sorted)} 个文件")
    
    return len(valid_files), len(invalid_files)

def main():
    import argparse
    parser = argparse.ArgumentParser(description='过滤有效的跨数据库SQL')
    parser.add_argument('--sql_dir', type=str,
                       default='benchmark/data/beijing/output/cross_db_single',
                       help='SQL结果目录')
    parser.add_argument('--output_dir', type=str, default=None,
                       help='输出目录（如果指定，则复制有效文件；否则删除无效文件）')
    
    args = parser.parse_args()
    
    valid_count, invalid_count = filter_valid_sqls(args.sql_dir, args.output_dir)
    
    print(f"\n" + "=" * 70)
    print(f"完成！")
    print(f"有效文件: {valid_count}")
    print(f"无效文件: {invalid_count}")
    print(f"=" * 70)

if __name__ == '__main__':
    import re
    main()


