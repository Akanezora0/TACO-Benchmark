#!/usr/bin/env python3
"""
补全NL查询到200条
"""

import json
import os
import re
import sys
import argparse
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed

# 导入主生成脚本的函数
script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, script_dir)
import importlib.util
spec = importlib.util.spec_from_file_location("generate_nl_queries_improved", os.path.join(script_dir, "4generate_nl_queries_improved.py"))
gen_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(gen_module)
process_single_sql = gen_module.process_single_sql

def main():
    parser = argparse.ArgumentParser(description='补全NL查询到200条')
    parser.add_argument('--sql_dir', type=str, required=True, help='SQL文件目录')
    parser.add_argument('--schema_dir', type=str, required=True, help='Schema文件目录')
    parser.add_argument('--output_dir', type=str, required=True, help='输出目录')
    parser.add_argument('--database', type=str, required=True, help='数据库名称')
    parser.add_argument('--target_count', type=int, default=200, help='目标数量')
    parser.add_argument('--max_workers', type=int, default=5, help='并发线程数')
    
    args = parser.parse_args()
    
    sql_db_dir = os.path.join(args.sql_dir, args.database)
    schema_file = os.path.join(args.schema_dir, args.database, f"{args.database}.json")
    output_db_dir = os.path.join(args.output_dir, args.database)
    
    # 获取所有SQL文件
    sql_files = sorted([f for f in os.listdir(sql_db_dir) if f.startswith('generated_sql_') and f.endswith('.json') and '_error' not in f],
                       key=lambda x: int(re.findall(r'\d+', x)[0]) if re.findall(r'\d+', x) else 0)
    
    # 获取所有NL查询文件
    nl_files = [f for f in os.listdir(output_db_dir) if f.startswith('generated_nl_query_') and f.endswith('.json')]
    nl_indices = set()
    for f in nl_files:
        match = re.search(r'generated_nl_query_(\d+)', f)
        if match:
            nl_indices.add(int(match.group(1)))
    
    # 找出缺失的索引
    target_indices = set(range(0, args.target_count))
    missing_indices = sorted(target_indices - nl_indices)
    
    print(f"数据库: {args.database}")
    print(f"SQL文件数: {len(sql_files)}")
    print(f"当前NL查询数: {len(nl_indices)}")
    print(f"缺失数量: {len(missing_indices)}")
    
    if not missing_indices:
        print("已满足目标数量，无需补全")
        return
    
    # 准备任务：为缺失的索引生成NL查询
    tasks = []
    sql_count = len(sql_files)
    
    for missing_idx in missing_indices:
        # 确定使用哪个SQL文件和variant
        # 根据索引计算：variant 0使用base_idx，variant 1使用sql_count*1+base_idx，variant 2使用sql_count*2+base_idx
        if missing_idx < sql_count:
            # 使用原始SQL（variant 0）
            base_idx = missing_idx
            variant = 0
            sql_file = sql_files[base_idx]
        else:
            # 计算是第几个变体
            # new_idx = sql_count * variant + base_idx
            # 所以：variant = (missing_idx - base_idx) // sql_count
            # 但我们需要找到对应的base_idx
            # 尝试不同的variant
            found = False
            for v in range(1, 4):  # variant 1, 2, 3
                base_idx = missing_idx - sql_count * v
                if 0 <= base_idx < sql_count:
                    variant = v
                    sql_file = sql_files[base_idx]
                    found = True
                    break
            
            if not found:
                # 如果找不到对应的base_idx，使用第一个SQL的变体
                base_idx = 0
                variant = (missing_idx // sql_count) + 1
                sql_file = sql_files[0]
        
        sql_file_path = os.path.join(sql_db_dir, sql_file)
        output_file = os.path.join(output_db_dir, f'generated_nl_query_{missing_idx}.json')
        
        if os.path.exists(output_file):
            continue
        
        tasks.append((sql_file_path, schema_file, output_file, variant))
    
    print(f"准备生成 {len(tasks)} 个NL查询...")
    
    # 并发处理
    if tasks:
        total_processed = 0
        total_success = 0
        
        with ThreadPoolExecutor(max_workers=args.max_workers) as executor:
            futures = {executor.submit(process_single_sql, sql_path, schema_file, out_file, variant): (sql_path, out_file) 
                      for sql_path, _, out_file, variant in tasks}
            
            for future in tqdm(as_completed(futures), total=len(futures), desc=f"补全 {args.database}"):
                sql_path, out_file = futures[future]
                total_processed += 1
                try:
                    if future.result():
                        total_success += 1
                except Exception as e:
                    print(f"处理失败 {sql_path}: {e}")
        
        print(f"\n完成！处理: {total_processed}, 成功: {total_success}")
        
        # 再次检查最终数量
        nl_files_final = [f for f in os.listdir(output_db_dir) if f.startswith('generated_nl_query_') and f.endswith('.json')]
        print(f"最终NL查询数量: {len(nl_files_final)}")

if __name__ == '__main__':
    main()

