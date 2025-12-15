#!/usr/bin/env python3
"""
清理3个和4个数据库的JOIN SQL中没有结果的文件
"""

import os
import json
import argparse
from collections import defaultdict

def cleanup_failed_sqls(sql_dir, db_counts=[3, 4], dry_run=False):
    """清理指定数据库数量类别中没有结果的SQL文件"""
    
    if not os.path.exists(sql_dir):
        print(f"目录不存在: {sql_dir}")
        return
    
    files_to_delete = []
    stats = defaultdict(lambda: {'total': 0, 'with_results': 0, 'no_results': 0})
    
    print("=" * 70)
    print("清理3个和4个数据库的JOIN SQL中没有结果的文件")
    print("=" * 70)
    print(f"SQL目录: {sql_dir}")
    print(f"目标数据库数量: {db_counts}")
    if dry_run:
        print("⚠️  这是预览模式（dry-run），不会实际删除文件")
    print()
    
    # 扫描所有SQL文件
    for f in os.listdir(sql_dir):
        if f.startswith('cross_db_generated_sql_') and f.endswith('.json'):
            file_path = os.path.join(sql_dir, f)
            try:
                with open(file_path, 'r', encoding='utf-8') as file:
                    data = json.load(file)
                    results = data.get('results', [])
                    num_databases = data.get('metadata', {}).get('num_databases', 
                                                                len(data.get('databases', [])))
                    
                    # 只处理指定数据库数量的文件
                    if num_databases in db_counts:
                        stats[num_databases]['total'] += 1
                        
                        if results is not None and len(results) > 0:
                            stats[num_databases]['with_results'] += 1
                        else:
                            stats[num_databases]['no_results'] += 1
                            files_to_delete.append((file_path, f, num_databases))
            except Exception as e:
                print(f"  警告: 读取文件失败 {f}: {e}")
    
    # 显示统计信息
    print("统计信息:")
    print("-" * 70)
    total_all = 0
    with_results_all = 0
    no_results_all = 0
    
    for db_count in sorted(stats.keys()):
        stat = stats[db_count]
        total_all += stat['total']
        with_results_all += stat['with_results']
        no_results_all += stat['no_results']
        
        print(f"  {db_count}个数据库:")
        print(f"    总数: {stat['total']}")
        print(f"    有结果: {stat['with_results']}")
        print(f"    无结果: {stat['no_results']} (将被删除)")
        print()
    
    print("-" * 70)
    print(f"总计:")
    print(f"  总数: {total_all}")
    print(f"  有结果: {with_results_all}")
    print(f"  无结果: {no_results_all} (将被删除)")
    print()
    
    if len(files_to_delete) == 0:
        print("✅ 没有需要删除的文件")
        return
    
    # 显示要删除的文件列表（前10个和后10个）
    print(f"将删除 {len(files_to_delete)} 个文件:")
    if len(files_to_delete) <= 20:
        for file_path, f, db_count in files_to_delete:
            print(f"  [{db_count}DB] {f}")
    else:
        for file_path, f, db_count in files_to_delete[:10]:
            print(f"  [{db_count}DB] {f}")
        print(f"  ... (省略 {len(files_to_delete) - 20} 个) ...")
        for file_path, f, db_count in files_to_delete[-10:]:
            print(f"  [{db_count}DB] {f}")
    print()
    
    # 执行删除
    if dry_run:
        print("⚠️  预览模式：以上文件将被删除（但实际未删除）")
    else:
        deleted_count = 0
        failed_count = 0
        
        for file_path, f, db_count in files_to_delete:
            try:
                os.remove(file_path)
                deleted_count += 1
            except Exception as e:
                print(f"  删除失败: {f} - {e}")
                failed_count += 1
        
        print(f"\n✅ 已删除 {deleted_count} 个文件")
        if failed_count > 0:
            print(f"⚠️  删除失败: {failed_count} 个文件")
    
    print("\n" + "=" * 70)
    print("完成！")
    print("=" * 70)

def main():
    parser = argparse.ArgumentParser(description='清理3个和4个数据库的JOIN SQL中没有结果的文件')
    parser.add_argument('--sql_dir', type=str,
                       default='benchmark/data/beijing/output/cross_db_single_join',
                       help='SQL文件目录')
    parser.add_argument('--only_3db', action='store_true',
                       help='只清理3个数据库的SQL')
    parser.add_argument('--only_4db', action='store_true',
                       help='只清理4个数据库的SQL')
    parser.add_argument('--dry_run', action='store_true',
                       help='预览模式，不实际删除文件')
    
    args = parser.parse_args()
    
    # 确定要清理的数据库数量
    if args.only_3db:
        db_counts = [3]
    elif args.only_4db:
        db_counts = [4]
    else:
        db_counts = [3, 4]
    
    # 转换为绝对路径
    if not os.path.isabs(args.sql_dir):
        # 获取项目根目录
        script_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.dirname(os.path.dirname(os.path.dirname(script_dir)))
        args.sql_dir = os.path.join(project_root, args.sql_dir)
    
    cleanup_failed_sqls(args.sql_dir, db_counts, args.dry_run)

if __name__ == '__main__':
    main()

