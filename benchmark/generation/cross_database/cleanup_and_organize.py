#!/usr/bin/env python3
"""
Clean up and organize the experiment directory
- Move old script versions to the archive directory
- Delete log files
- Delete temporary files
- Keep core files
"""

import os
import shutil
import argparse

# Get script directory
script_dir = os.path.dirname(os.path.abspath(__file__))

# Files to archive (old script versions)
ARCHIVE_FILES = [
    # Old script versions
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
    
    # Test scripts
    'test_api.py',
    'test_api_old_style.py',
    '8test_3db_4db_generation.py',
    
    # Temporary scripts
    '0backup_and_reset.py',
    '6cleanup_wrong_backup.py',
    '2rename_and_organize.py',
]

# Files to delete (logs and temporary files)
DELETE_FILES = [
    # Log files
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
    
    # Temporary data files (large files)
    'candidates_2db.json',
    'cross_db_skeletons_2db_企业服务_社会保障.json',
    'joinable_columns.json',              # 42GB large file
    'joinable_columns_report.txt',        # 367MB large file
    'generation_plan.json',
    'database_combinations_distribution.json',
    '快速运行命令.txt',
]

# Documents to archive (outdated docs)
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

# Shell scripts (optional deletion)
SHELL_SCRIPTS = [
    'batch_generate_skeletons.sh',
    'run_generation_pipeline.sh',
    'run_join_generation_background.sh',
    'run_join_generation_pipeline.sh',
    '快速测试运行.sh',
    '简单运行流程.sh',
]

def cleanup_directory(dry_run=False):
    """Clean up and organize the directory."""
    
    archive_dir = os.path.join(script_dir, 'archive')
    
    print("=" * 70)
    print("Clean up and organize experiment directory")
    print("=" * 70)
    if dry_run:
        print("⚠️  Preview mode (dry-run); no files will be moved or deleted")
    print()
    
    # 1. Create archive directory
    if not dry_run:
        os.makedirs(archive_dir, exist_ok=True)
        os.makedirs(os.path.join(archive_dir, 'scripts'), exist_ok=True)
        os.makedirs(os.path.join(archive_dir, 'docs'), exist_ok=True)
        os.makedirs(os.path.join(archive_dir, 'shell_scripts'), exist_ok=True)
    
    # 2. Archive old script versions
    print("Archiving old script versions...")
    archived_scripts = 0
    for filename in ARCHIVE_FILES:
        filepath = os.path.join(script_dir, filename)
        if os.path.exists(filepath):
            dest = os.path.join(archive_dir, 'scripts', filename)
            if dry_run:
                print(f"  Will move: {filename} -> archive/scripts/")
            else:
                shutil.move(filepath, dest)
            archived_scripts += 1
    print(f"  Archived {archived_scripts} script files")
    print()
    
    # 3. Archive outdated documents
    print("Archiving outdated documents...")
    archived_docs = 0
    for filename in ARCHIVE_DOCS:
        filepath = os.path.join(script_dir, filename)
        if os.path.exists(filepath):
            dest = os.path.join(archive_dir, 'docs', filename)
            if dry_run:
                print(f"  Will move: {filename} -> archive/docs/")
            else:
                shutil.move(filepath, dest)
            archived_docs += 1
    print(f"  Archived {archived_docs} document files")
    print()
    
    # 4. Archive shell scripts
    print("Archiving shell scripts...")
    archived_shell = 0
    for filename in SHELL_SCRIPTS:
        filepath = os.path.join(script_dir, filename)
        if os.path.exists(filepath):
            dest = os.path.join(archive_dir, 'shell_scripts', filename)
            if dry_run:
                print(f"  Will move: {filename} -> archive/shell_scripts/")
            else:
                shutil.move(filepath, dest)
            archived_shell += 1
    print(f"  Archived {archived_shell} shell scripts")
    print()
    
    # 5. Delete log files
    print("Deleting log files...")
    deleted_logs = 0
    for filename in DELETE_FILES:
        filepath = os.path.join(script_dir, filename)
        if os.path.exists(filepath):
            if dry_run:
                print(f"  Will delete: {filename}")
            else:
                os.remove(filepath)
            deleted_logs += 1
    print(f"  Deleted {deleted_logs} log/temporary files")
    print()
    
    # 6. Delete all .log files (additional)
    print("Deleting all .log files...")
    deleted_extra_logs = 0
    for filename in os.listdir(script_dir):
        if filename.endswith('.log'):
            filepath = os.path.join(script_dir, filename)
            if dry_run:
                print(f"  Will delete: {filename}")
            else:
                os.remove(filepath)
            deleted_extra_logs += 1
    print(f"  Deleted {deleted_extra_logs} additional .log files")
    print()
    
    # Summary
    print("=" * 70)
    print("Cleanup complete!")
    print(f"Archived scripts: {archived_scripts}")
    print(f"Archived documents: {archived_docs}")
    print(f"Archived shell scripts: {archived_shell}")
    print(f"Deleted log/temporary files: {deleted_logs + deleted_extra_logs}")
    print("=" * 70)
    
    if dry_run:
        print("\n⚠️  Preview mode; no actions were performed")
        print("Run without --dry_run to execute cleanup")

def main():
    parser = argparse.ArgumentParser(description='Clean up and organize the experiment directory')
    parser.add_argument('--dry_run', action='store_true',
                       help='Preview mode; do not move or delete files')
    
    args = parser.parse_args()
    
    cleanup_directory(args.dry_run)

if __name__ == '__main__':
    main()
