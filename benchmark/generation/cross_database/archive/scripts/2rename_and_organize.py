#!/usr/bin/env python3
"""
重命名和组织有结果的SQL文件

按数据库数量分类（2、3、4个数据库），然后按顺序重命名
例如：cross_db_generated_sql_0.json, cross_db_generated_sql_1.json, ...
"""

import os
import json
import argparse
from collections import defaultdict

def rename_and_organize(sql_dir):
    """重命名和组织SQL文件"""
    
    # 1. 收集所有有结果的SQL文件，按数据库数量分类
    sqls_by_db_count = defaultdict(list)
    
    print("收集有结果的SQL文件...")
    for f in sorted(os.listdir(sql_dir)):
        if f.startswith('cross_db_generated_sql_') and f.endswith('.json'):
            file_path = os.path.join(sql_dir, f)
            try:
                with open(file_path, 'r', encoding='utf-8') as file:
                    data = json.load(file)
                    results = data.get('results', [])
                    
                    # 只处理有结果的
                    if results is not None and len(results) > 0:
                        num_databases = data.get('metadata', {}).get('num_databases', 
                                                                    len(data.get('databases', [])))
                        sqls_by_db_count[num_databases].append((file_path, data))
            except Exception as e:
                print(f"处理文件 {f} 时出错: {e}")
    
    print(f"\n按数据库数量分类:")
    for db_count in sorted(sqls_by_db_count.keys()):
        print(f"  {db_count}个数据库: {len(sqls_by_db_count[db_count])} 个文件")
    
    # 2. 为每个数据库数量类别重命名
    total_renamed = 0
    rename_map = {}
    
    for db_count in sorted(sqls_by_db_count.keys()):
        files = sqls_by_db_count[db_count]
        
        # 按某种顺序排序（可以按文件修改时间或SQL内容）
        # 这里简单按文件路径排序
        files.sort(key=lambda x: x[0])
        
        # 计算起始索引（基于已有的文件）
        existing_indices = []
        for f in os.listdir(sql_dir):
            if f.startswith('cross_db_generated_sql_') and f.endswith('.json'):
                try:
                    idx = int(f.replace('cross_db_generated_sql_', '').replace('.json', ''))
                    existing_indices.append(idx)
                except:
                    pass
        
        start_idx = max(existing_indices) + 1 if existing_indices else 0
        
        # 重命名
        for i, (file_path, data) in enumerate(files):
            new_name = f"cross_db_generated_sql_{start_idx + i}.json"
            new_path = os.path.join(sql_dir, new_name)
            
            # 如果新文件名已存在，跳过（避免覆盖）
            if os.path.exists(new_path) and new_path != file_path:
                continue
            
            # 如果文件已经是正确的名称，跳过
            if os.path.basename(file_path) == new_name:
                continue
            
            rename_map[file_path] = new_path
            total_renamed += 1
    
    # 3. 执行重命名（先重命名到临时名称，再重命名到最终名称，避免冲突）
    print(f"\n重命名 {total_renamed} 个文件...")
    
    # 先重命名到临时名称
    temp_map = {}
    for old_path, new_path in rename_map.items():
        temp_name = os.path.join(os.path.dirname(old_path), f"__temp_{os.path.basename(new_path)}")
        os.rename(old_path, temp_name)
        temp_map[temp_name] = new_path
    
    # 再重命名到最终名称
    for temp_path, final_path in temp_map.items():
        os.rename(temp_path, final_path)
    
    print(f"重命名完成: {total_renamed} 个文件")
    
    return total_renamed

def main():
    parser = argparse.ArgumentParser(description='重命名和组织有结果的SQL文件')
    parser.add_argument('--sql_dir', type=str,
                       default='benchmark/data/beijing/output/cross_db_single_join',
                       help='SQL文件目录')
    
    args = parser.parse_args()
    
    print("=" * 70)
    print("重命名和组织SQL文件")
    print("=" * 70)
    print()
    
    rename_and_organize(args.sql_dir)
    
    print("\n" + "=" * 70)

if __name__ == '__main__':
    main()


