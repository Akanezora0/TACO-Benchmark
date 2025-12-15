#!/usr/bin/env python3
"""
清理备份目录中错误编号的文件（编号大于50且小于342的文件）
这些是之前错误备份的文件，应该删除
"""

import os
import argparse

def cleanup_wrong_backup(backup_dir, max_correct_index=50):
    """清理错误编号的文件（删除所有编号大于max_correct_index的文件，因为应该从max_correct_index+1开始连续编号）"""
    
    if not os.path.exists(backup_dir):
        print(f"备份目录不存在: {backup_dir}")
        return 0
    
    wrong_files = []
    for f in os.listdir(backup_dir):
        if f.startswith('cross_db_generated_sql_') and f.endswith('.json'):
            try:
                idx = int(f.replace('cross_db_generated_sql_', '').replace('.json', ''))
                # 编号大于max_correct_index的文件都是错误备份的（应该从max_correct_index+1开始连续编号）
                if idx > max_correct_index:
                    wrong_files.append((f, idx))
            except:
                pass
    
    if not wrong_files:
        print("没有找到需要清理的文件")
        return 0
    
    wrong_files.sort(key=lambda x: x[1])
    print(f"找到 {len(wrong_files)} 个错误编号的文件（编号在 {max_correct_index+1} 到 341 之间）")
    print(f"这些文件将被删除...")
    
    deleted = 0
    for f, idx in wrong_files:
        file_path = os.path.join(backup_dir, f)
        try:
            os.remove(file_path)
            deleted += 1
            print(f"  删除: {f} (编号: {idx})")
        except Exception as e:
            print(f"  删除失败: {f} - {e}")
    
    return deleted

def main():
    parser = argparse.ArgumentParser(description='清理备份目录中错误编号的文件')
    parser.add_argument('--backup_dir', type=str,
                       default='benchmark/data/beijing/output/cross_db_single_join_backup_51',
                       help='备份目录')
    parser.add_argument('--max_correct_index', type=int, default=50,
                       help='最大正确编号（默认50，即0-50是正确的）')
    
    args = parser.parse_args()
    
    print("=" * 70)
    print("清理备份目录中错误编号的文件")
    print("=" * 70)
    print(f"备份目录: {args.backup_dir}")
    print(f"最大正确编号: {args.max_correct_index}")
    print()
    
    deleted = cleanup_wrong_backup(args.backup_dir, args.max_correct_index)
    
    print(f"\n✅ 已删除 {deleted} 个错误编号的文件")
    print("\n" + "=" * 70)
    print("完成！")
    print("=" * 70)

if __name__ == '__main__':
    main()

