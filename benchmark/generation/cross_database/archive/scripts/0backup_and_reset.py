#!/usr/bin/env python3
"""
备份已有结果并清空目录，准备重新生成
"""

import os
import json
import shutil
import argparse
from pathlib import Path

def backup_existing_results(sql_dir, backup_dir, start_index=0):
    """备份已有的有结果的SQL文件，重命名为连续编号"""
    
    # 创建备份目录
    os.makedirs(backup_dir, exist_ok=True)
    
    # 收集所有有结果的SQL文件
    valid_files = []
    for f in os.listdir(sql_dir):
        if f.startswith('cross_db_generated_sql_') and f.endswith('.json'):
            file_path = os.path.join(sql_dir, f)
            try:
                with open(file_path, 'r', encoding='utf-8') as file:
                    data = json.load(file)
                    results = data.get('results', [])
                    # 有结果的文件
                    if results is not None and len(results) > 0:
                        valid_files.append((file_path, f))
            except:
                pass
    
    print(f"找到 {len(valid_files)} 个有结果的SQL文件")
    
    # 按文件名排序，确保顺序一致
    valid_files.sort(key=lambda x: x[1])
    
    # 备份并重命名
    for idx, (file_path, original_name) in enumerate(valid_files):
        new_name = f"cross_db_generated_sql_{start_index + idx}.json"
        backup_path = os.path.join(backup_dir, new_name)
        shutil.copy2(file_path, backup_path)
        print(f"  备份: {original_name} -> {new_name}")
    
    return len(valid_files)

def clear_sql_directory(sql_dir):
    """清空SQL目录"""
    print(f"\n清空目录: {sql_dir}")
    for f in os.listdir(sql_dir):
        if f.startswith('cross_db_generated_sql_') and f.endswith('.json'):
            file_path = os.path.join(sql_dir, f)
            os.remove(file_path)
            print(f"  删除: {f}")
    print("清空完成")

def main():
    parser = argparse.ArgumentParser(description='备份已有结果并清空目录')
    parser.add_argument('--sql_dir', type=str,
                       default='benchmark/data/beijing/output/cross_db_single_join',
                       help='SQL文件目录')
    parser.add_argument('--backup_dir', type=str,
                       default='benchmark/data/beijing/output/cross_db_single_join_backup_51',
                       help='备份目录')
    parser.add_argument('--start_index', type=int, default=0,
                       help='起始编号')
    parser.add_argument('--clear', action='store_true',
                       help='是否清空原目录')
    
    args = parser.parse_args()
    
    print("=" * 70)
    print("备份已有结果")
    print("=" * 70)
    
    # 备份
    count = backup_existing_results(args.sql_dir, args.backup_dir, args.start_index)
    print(f"\n✅ 已备份 {count} 个文件到 {args.backup_dir}")
    
    # 清空
    if args.clear:
        clear_sql_directory(args.sql_dir)
    
    print("\n" + "=" * 70)
    print("完成！")
    print("=" * 70)

if __name__ == '__main__':
    main()

