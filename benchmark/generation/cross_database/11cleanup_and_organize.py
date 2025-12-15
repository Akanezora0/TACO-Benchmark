#!/usr/bin/env python3
"""
清理和整理实验目录
- 将旧版本脚本移到archive目录
- 删除日志文件
- 删除临时文件
- 保留核心文件
"""

import os
import shutil
import argparse

# 获取脚本所在目录
script_dir = os.path.dirname(os.path.abspath(__file__))

# 要归档的文件（旧版本脚本）
ARCHIVE_FILES = [
    # 旧版本脚本
    '2generate_cross_db_skeletons.py',
    'cross_db_2fill_sql_placeholders.py',
    '4generate_more_join_sqls.py',
    'batch_fill_sql.py',
    'batch_generate_graphs.py',
    'generate_skeletons_batch.py',
    '1select_candidates.py',
    'add_results_to_existing.py',
    'analyze_and_filter.py',
    'filter_valid_sqls.py',
    'generate_all_combinations.py',
    'check_generation_status.py',
    
    # 测试脚本
    'test_api.py',
    'test_api_old_style.py',
    '8test_3db_4db_generation.py',
    
    # 临时脚本
    '0backup_and_reset.py',
    '6cleanup_wrong_backup.py',
    '2rename_and_organize.py',
]

# 要删除的文件（日志和临时文件）
DELETE_FILES = [
    # 日志文件
    'batch_fill_sql_final.log',
    'batch_fill_sql_fixed.log',
    'batch_fill_sql_improved.log',
    'batch_fill_sql_optimized.log',
    'batch_fill_sql_union.log',
    'batch_fill_sql.log',
    'batch_generate_graphs.log',
    'cross_db_fill_enhanced.log',
    'cross_db_fill_final.log',
    'cross_db_fill_flexible.log',
    'cross_db_fill_join.log',
    'cross_db_fill_optimized_prompt.log',
    'cross_db_fill_optimized.log',
    'cross_db_fill_rerun.log',
    'cross_db_fill.log',
    'generate_skeletons.log',
    'join_generation_background.log',
    
    # 临时数据文件（大文件）
    'candidates_2db.json',
    'cross_db_skeletons_2db_企业服务_社会保障.json',
    'joinable_columns.json',              # 42GB大文件
    'joinable_columns_report.txt',        # 367MB大文件
    'generation_plan.json',
    'database_combinations_distribution.json',
    '快速运行命令.txt',
]

# 要归档的文档（过时文档）
ARCHIVE_DOCS = [
    'API修复说明.md',
    'API问题报告.md',
    'JOIN版本生成使用说明.md',
    'JOIN版本生成结果报告.md',
    'JOIN版本说明.md',
    'SQL填充优化说明.md',
    'SQL生成统计报告.md',
    '修复说明_保存失败文件.md',
    '图文件位置说明.md',
    '图信息压缩方案.md',
    '运行说明_修复版.md',
    '运行说明_简化版.md',
    '生成3db4db说明.md',
    '骨架准备完成说明.md',
    '根因分析报告.md',
    '问题根因分析.md',
    '简单运行命令.md',
    '后台运行命令.md',
    '后台运行说明.md',
    '3db_4db生成说明.md',
    '使用说明_完整版.md',
    '使用说明.md',
    '运行和查看日志说明.md',
    '数据库组合策略.md',
    '优化总结.md',
    '组合方案总结.md',
    '批量生成总结.md',
    '灵活优化策略说明.md',
    '表选择逻辑说明.md',
    '优化效果总结.md',
    '生成说明.md',
    '等待完成并统计.md',
    '优化措施总结.md',
    '总结.md',
    '当前进度和下一步.md',
    '实验计划.md',
    '跨数据库查询实现方案.md',
    '跨数据库查询数量计算.md',
    '简化方案说明.md',
    '跨数据库Benchmark数据生成方案.md',
    '最终方案.md',
]

# Shell脚本（可选删除）
SHELL_SCRIPTS = [
    'batch_generate_skeletons.sh',
    'run_generation_pipeline.sh',
    'run_join_generation_background.sh',
    'run_join_generation_pipeline.sh',
    '快速测试运行.sh',
    '简单运行流程.sh',
]

def cleanup_directory(dry_run=False):
    """清理和整理目录"""
    
    archive_dir = os.path.join(script_dir, 'archive')
    
    print("=" * 70)
    print("清理和整理实验目录")
    print("=" * 70)
    if dry_run:
        print("⚠️  这是预览模式（dry-run），不会实际移动或删除文件")
    print()
    
    # 1. 创建archive目录
    if not dry_run:
        os.makedirs(archive_dir, exist_ok=True)
        os.makedirs(os.path.join(archive_dir, 'scripts'), exist_ok=True)
        os.makedirs(os.path.join(archive_dir, 'docs'), exist_ok=True)
        os.makedirs(os.path.join(archive_dir, 'shell_scripts'), exist_ok=True)
    
    # 2. 归档旧版本脚本
    print("归档旧版本脚本...")
    archived_scripts = 0
    for filename in ARCHIVE_FILES:
        filepath = os.path.join(script_dir, filename)
        if os.path.exists(filepath):
            dest = os.path.join(archive_dir, 'scripts', filename)
            if dry_run:
                print(f"  将移动: {filename} -> archive/scripts/")
            else:
                shutil.move(filepath, dest)
            archived_scripts += 1
    print(f"  归档了 {archived_scripts} 个脚本文件")
    print()
    
    # 3. 归档过时文档
    print("归档过时文档...")
    archived_docs = 0
    for filename in ARCHIVE_DOCS:
        filepath = os.path.join(script_dir, filename)
        if os.path.exists(filepath):
            dest = os.path.join(archive_dir, 'docs', filename)
            if dry_run:
                print(f"  将移动: {filename} -> archive/docs/")
            else:
                shutil.move(filepath, dest)
            archived_docs += 1
    print(f"  归档了 {archived_docs} 个文档文件")
    print()
    
    # 4. 归档Shell脚本
    print("归档Shell脚本...")
    archived_shell = 0
    for filename in SHELL_SCRIPTS:
        filepath = os.path.join(script_dir, filename)
        if os.path.exists(filepath):
            dest = os.path.join(archive_dir, 'shell_scripts', filename)
            if dry_run:
                print(f"  将移动: {filename} -> archive/shell_scripts/")
            else:
                shutil.move(filepath, dest)
            archived_shell += 1
    print(f"  归档了 {archived_shell} 个Shell脚本")
    print()
    
    # 5. 删除日志文件
    print("删除日志文件...")
    deleted_logs = 0
    for filename in DELETE_FILES:
        filepath = os.path.join(script_dir, filename)
        if os.path.exists(filepath):
            if dry_run:
                print(f"  将删除: {filename}")
            else:
                os.remove(filepath)
            deleted_logs += 1
    print(f"  删除了 {deleted_logs} 个日志/临时文件")
    print()
    
    # 6. 删除所有.log文件（额外）
    print("删除所有.log文件...")
    deleted_extra_logs = 0
    for filename in os.listdir(script_dir):
        if filename.endswith('.log'):
            filepath = os.path.join(script_dir, filename)
            if dry_run:
                print(f"  将删除: {filename}")
            else:
                os.remove(filepath)
            deleted_extra_logs += 1
    print(f"  删除了 {deleted_extra_logs} 个额外的.log文件")
    print()
    
    # 总结
    print("=" * 70)
    print("清理完成!")
    print(f"归档脚本: {archived_scripts} 个")
    print(f"归档文档: {archived_docs} 个")
    print(f"归档Shell脚本: {archived_shell} 个")
    print(f"删除日志/临时文件: {deleted_logs + deleted_extra_logs} 个")
    print("=" * 70)
    
    if dry_run:
        print("\n⚠️  这是预览模式，实际未执行任何操作")
        print("运行时不加 --dry_run 参数将实际执行清理")

def main():
    parser = argparse.ArgumentParser(description='清理和整理实验目录')
    parser.add_argument('--dry_run', action='store_true',
                       help='预览模式，不实际移动或删除文件')
    
    args = parser.parse_args()
    
    cleanup_directory(args.dry_run)

if __name__ == '__main__':
    main()

