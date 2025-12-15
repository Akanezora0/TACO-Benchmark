#!/usr/bin/env python3
"""
清理没有结果的SQL文件

删除所有没有results或results为空的SQL文件
"""

import os
import json
import argparse

def cleanup_failed_sqls(sql_dir):
    """清理没有结果的SQL文件"""
    deleted_count = 0
    kept_count = 0
    
    print(f"清理目录: {sql_dir}")
    
    for f in sorted(os.listdir(sql_dir)):
        if f.startswith('cross_db_generated_sql_') and f.endswith('.json'):
            file_path = os.path.join(sql_dir, f)
            try:
                with open(file_path, 'r', encoding='utf-8') as file:
                    data = json.load(file)
                    results = data.get('results', [])
                    
                    # 如果没有结果或结果为空，删除文件
                    if results is None or len(results) == 0:
                        os.remove(file_path)
                        deleted_count += 1
                    else:
                        kept_count += 1
            except Exception as e:
                print(f"处理文件 {f} 时出错: {e}")
                # 出错的文件也删除
                try:
                    os.remove(file_path)
                    deleted_count += 1
                except:
                    pass
    
    print(f"\n清理完成:")
    print(f"  保留: {kept_count} 个文件")
    print(f"  删除: {deleted_count} 个文件")
    
    return kept_count, deleted_count

def main():
    parser = argparse.ArgumentParser(description='清理没有结果的SQL文件')
    parser.add_argument('--sql_dir', type=str,
                       default='benchmark/data/beijing/output/cross_db_single_join',
                       help='SQL文件目录')
    
    args = parser.parse_args()
    
    print("=" * 70)
    print("清理没有结果的SQL文件")
    print("=" * 70)
    print()
    
    cleanup_failed_sqls(args.sql_dir)
    
    print("\n" + "=" * 70)

if __name__ == '__main__':
    main()


