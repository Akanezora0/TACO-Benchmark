#!/usr/bin/env python3
"""
分析跨数据库SQL执行结果，并过滤有效SQL
"""

import json
import os
import re
from collections import defaultdict

def analyze_execution_results(sql_dir):
    """分析执行结果"""
    sql_files = sorted([f for f in os.listdir(sql_dir) 
                       if f.startswith('cross_db_generated_sql_') and f.endswith('.json')])
    
    print("=" * 70)
    print("跨数据库SQL执行结果分析")
    print("=" * 70)
    print(f"\n总文件数: {len(sql_files)}")
    
    has_results = 0
    no_results = 0
    error_types = defaultdict(int)
    
    for sql_file in sql_files:
        file_path = os.path.join(sql_dir, sql_file)
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            results = data.get('results', [])
            error = data.get('metadata', {}).get('execution_error')
            
            if len(results) > 0:
                has_results += 1
            else:
                no_results += 1
                if error:
                    # 提取错误类型
                    if 'ATTACH' in error:
                        error_types['ATTACH失败'] += 1
                    elif '单数据库格式' in error:
                        error_types['单数据库格式失败'] += 1
                    else:
                        error_types['其他错误'] += 1
                else:
                    error_types['无错误信息'] += 1
        except Exception as e:
            no_results += 1
            error_types['文件读取失败'] += 1
    
    print(f"\n有执行结果: {has_results} ({has_results/len(sql_files)*100:.1f}%)")
    print(f"无执行结果: {no_results} ({no_results/len(sql_files)*100:.1f}%)")
    
    print(f"\n错误类型分布:")
    for error_type, count in sorted(error_types.items(), key=lambda x: x[1], reverse=True):
        print(f"  {error_type}: {count}")
    
    return has_results, no_results

def filter_valid_sqls(sql_dir):
    """过滤有效SQL，删除无效的"""
    sql_files = sorted([f for f in os.listdir(sql_dir) 
                       if f.startswith('cross_db_generated_sql_') and f.endswith('.json')])
    
    valid_files = []
    invalid_files = []
    
    for sql_file in sql_files:
        file_path = os.path.join(sql_dir, sql_file)
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            results = data.get('results', [])
            error = data.get('metadata', {}).get('execution_error')
            
            # 有效：有结果且无错误
            if len(results) > 0 and not error:
                valid_files.append(sql_file)
            else:
                invalid_files.append(sql_file)
        except:
            invalid_files.append(sql_file)
    
    print(f"\n" + "=" * 70)
    print(f"过滤结果")
    print(f"=" * 70)
    print(f"有效文件: {len(valid_files)}")
    print(f"无效文件: {len(invalid_files)}")
    
    # 删除无效文件
    print(f"\n删除无效文件...")
    for sql_file in invalid_files:
        file_path = os.path.join(sql_dir, sql_file)
        os.remove(file_path)
    
    # 重新编号
    print(f"重新编号有效文件...")
    valid_files_sorted = sorted(valid_files, 
                                key=lambda x: int(re.search(r'(\d+)', x).group(1)) 
                                if re.search(r'(\d+)', x) else 0)
    
    for i, sql_file in enumerate(valid_files_sorted):
        old_path = os.path.join(sql_dir, sql_file)
        new_name = f"cross_db_generated_sql_{i}.json"
        new_path = os.path.join(sql_dir, new_name)
        
        if old_path != new_path:
            os.rename(old_path, new_path)
    
    print(f"完成！保留 {len(valid_files_sorted)} 个有效文件")
    
    return len(valid_files_sorted)

def main():
    import argparse
    parser = argparse.ArgumentParser(description='分析和过滤跨数据库SQL')
    parser.add_argument('--sql_dir', type=str,
                       default='benchmark/data/beijing/output/cross_db_single',
                       help='SQL结果目录')
    parser.add_argument('--filter', action='store_true',
                       help='是否过滤无效SQL')
    
    args = parser.parse_args()
    
    # 分析
    has_results, no_results = analyze_execution_results(args.sql_dir)
    
    # 如果需要，过滤
    if args.filter:
        valid_count = filter_valid_sqls(args.sql_dir)
        print(f"\n最终有效SQL数量: {valid_count}")
        print(f"目标数量: 359条（跨2个数据库）")
        if valid_count < 359:
            print(f"⚠️  有效SQL数量不足，需要生成更多SQL骨架")
            print(f"   如果成功率{has_results/(has_results+no_results)*100:.1f}%，需要生成约 {int(359/(has_results/(has_results+no_results)))} 个SQL骨架")

if __name__ == '__main__':
    main()


