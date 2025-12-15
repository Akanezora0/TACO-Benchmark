#!/usr/bin/env python3
"""
备份新生成的有结果的SQL到备份目录，继续编号
"""

import os
import json
import shutil
import argparse

def get_next_index(backup_dir):
    """获取备份目录中下一个可用的编号（从最大连续编号+1开始）"""
    indices = []
    if os.path.exists(backup_dir):
        for f in os.listdir(backup_dir):
            if f.startswith('cross_db_generated_sql_') and f.endswith('.json'):
                try:
                    idx = int(f.replace('cross_db_generated_sql_', '').replace('.json', ''))
                    indices.append(idx)
                except:
                    pass
    
    if not indices:
        # 如果备份目录为空或不存在，从0开始
        return 0
    
    # 找到最大连续编号（从0开始连续的最大编号）
    indices_sorted = sorted(indices)
    max_continuous = -1
    
    # 从0开始检查连续编号
    expected = 0
    for idx in indices_sorted:
        if idx == expected:
            max_continuous = idx
            expected += 1
        elif idx > expected:
            # 出现断点，停止
            break
    
    # 返回最大连续编号+1
    return max_continuous + 1

def backup_new_results(sql_dir, backup_dir):
    """备份新生成的有结果的SQL文件"""
    
    # 创建备份目录
    os.makedirs(backup_dir, exist_ok=True)
    
    # 获取下一个编号
    next_index = get_next_index(backup_dir)
    print(f"下一个编号: {next_index}")
    
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
    
    # 按文件名排序
    valid_files.sort(key=lambda x: x[1])
    
    # 备份并重命名
    backed_up = 0
    for idx, (file_path, original_name) in enumerate(valid_files):
        new_name = f"cross_db_generated_sql_{next_index + idx}.json"
        backup_path = os.path.join(backup_dir, new_name)
        
        # 检查是否已存在（避免重复备份）
        if not os.path.exists(backup_path):
            shutil.copy2(file_path, backup_path)
            backed_up += 1
            print(f"  备份: {original_name} -> {new_name}")
    
    return backed_up

def main():
    # 获取脚本所在目录和项目根目录
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(script_dir)))
    
    parser = argparse.ArgumentParser(description='备份新生成的有结果的SQL')
    parser.add_argument('--sql_dir', type=str,
                       default=os.path.join(project_root, 'benchmark', 'data', 'beijing', 'output', 'cross_db_single_join'),
                       help='SQL文件目录')
    parser.add_argument('--backup_dir', type=str,
                       default=os.path.join(project_root, 'benchmark', 'data', 'beijing', 'output', 'cross_db_single_join_backup_51'),
                       help='备份目录')
    
    args = parser.parse_args()
    
    # 转换为绝对路径（如果用户提供了相对路径）
    if not os.path.isabs(args.sql_dir):
        args.sql_dir = os.path.join(project_root, args.sql_dir) if not os.path.isabs(args.sql_dir) else args.sql_dir
    if not os.path.isabs(args.backup_dir):
        args.backup_dir = os.path.join(project_root, args.backup_dir) if not os.path.isabs(args.backup_dir) else args.backup_dir
    
    print("=" * 70)
    print("备份新生成的有结果的SQL")
    print("=" * 70)
    
    count = backup_new_results(args.sql_dir, args.backup_dir)
    print(f"\n✅ 已备份 {count} 个新文件到 {args.backup_dir}")
    
    print("\n" + "=" * 70)
    print("完成！")
    print("=" * 70)

if __name__ == '__main__':
    main()

