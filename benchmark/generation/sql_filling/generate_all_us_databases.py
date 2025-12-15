#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
US数据集批量生成所有数据库的单数据库SQL

一键执行所有数据库的SQL生成，支持命令行参数控制
"""

import os
import sys
import json
import argparse
import subprocess
from pathlib import Path
from datetime import datetime
from collections import defaultdict

# 项目根目录
PROJECT_ROOT = Path(__file__).parent.parent.parent.parent

# 默认路径
DEFAULT_DATABASE_DIR = PROJECT_ROOT / "benchmark" / "data" / "us" / "database"
DEFAULT_SKELETON_DIR = PROJECT_ROOT / "benchmark" / "data" / "us" / "output" / "sql_skeleton"
DEFAULT_GRAPH_DIR = PROJECT_ROOT / "benchmark" / "data" / "us" / "output" / "graph"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "benchmark" / "data" / "us" / "output"
DEFAULT_TARGET_COUNT = 220  # 每个数据库目标220条

def get_all_databases(skeleton_dir):
    """获取所有数据库列表"""
    databases = []
    if not skeleton_dir.exists():
        return databases
    
    for skeleton_file in skeleton_dir.glob("*_sql_skeleton.json"):
        db_name = skeleton_file.name.replace("_sql_skeleton.json", "")
        databases.append(db_name)
    
    return sorted(databases)

def count_existing_sqls(output_dir, database_name):
    """统计当前已有的SQL数量"""
    sql_dir = output_dir / "single" / database_name
    if not sql_dir.exists():
        return 0
    
    count = 0
    for f in sql_dir.glob("generated_sql_*.json"):
        try:
            # 检查文件是否有效（有results字段）
            with open(f, 'r', encoding='utf-8') as file:
                data = json.load(file)
                if 'results' in data and data['results'] is not None:
                    count += 1
        except:
            pass
    
    return count

def get_database_status(database_name, skeleton_dir, output_dir, target_count):
    """获取数据库的生成状态"""
    skeleton_file = skeleton_dir / f"{database_name}_sql_skeleton.json"
    current_count = count_existing_sqls(output_dir, database_name)
    need_count = max(0, target_count - current_count)
    
    return {
        'database': database_name,
        'skeleton_exists': skeleton_file.exists(),
        'current_count': current_count,
        'target_count': target_count,
        'need_count': need_count,
        'completed': need_count == 0
    }

def print_status_table(databases_status):
    """打印状态表格"""
    print("=" * 100)
    print(f"{'数据库名称':<50} {'当前数量':<10} {'目标数量':<10} {'还需生成':<10} {'状态':<10}")
    print("=" * 100)
    
    total_current = 0
    total_target = 0
    total_need = 0
    completed_count = 0
    
    for status in databases_status:
        db_name = status['database']
        current = status['current_count']
        target = status['target_count']
        need = status['need_count']
        completed = status['completed']
        
        # 截断过长的数据库名称
        display_name = db_name[:47] + "..." if len(db_name) > 50 else db_name
        
        status_str = "✅ 完成" if completed else "⏳ 进行中"
        
        print(f"{display_name:<50} {current:<10} {target:<10} {need:<10} {status_str:<10}")
        
        total_current += current
        total_target += target
        total_need += need
        if completed:
            completed_count += 1
    
    print("=" * 100)
    print(f"{'总计':<50} {total_current:<10} {total_target:<10} {total_need:<10} {completed_count}/{len(databases_status)} 完成")
    print("=" * 100)

def generate_sql_for_database(database_name, script_path, database_dir, skeleton_dir, 
                             graph_dir, output_dir, target_count, max_retries, 
                             background=False, log_dir=None, max_workers=None):
    """为单个数据库生成SQL"""
    cmd = [
        sys.executable,
        str(script_path),
        "--database_name", database_name,
        "--database_dir", str(database_dir),
        "--skeleton_dir", str(skeleton_dir),
        "--graph_dir", str(graph_dir),
        "--output_dir", str(output_dir),
        "--target_count", str(target_count),
        "--max_retries", str(max_retries)
    ]
    
    # 如果指定了max_workers，添加到命令中
    if max_workers:
        cmd.extend(["--max_workers", str(max_workers)])
    
    if background:
        if log_dir is None:
            log_dir = PROJECT_ROOT / "benchmark" / "generation" / "sql_filling" / "logs"
        log_dir.mkdir(parents=True, exist_ok=True)
        
        log_file = log_dir / f"generate_us_sql_{database_name.replace(' ', '_').replace('-', '_').replace('/', '_')}.log"
        
        with open(log_file, 'w', encoding='utf-8') as f:
            process = subprocess.Popen(
                cmd,
                stdout=f,
                stderr=subprocess.STDOUT,
                cwd=str(script_path.parent)
            )
        
        return process, log_file
    else:
        try:
            process = subprocess.run(
                cmd,
                cwd=str(script_path.parent),
                timeout=None  # 不设置超时，让进程自然完成
            )
            return process, None
        except subprocess.TimeoutExpired:
            # 超时处理
            print(f"警告: {database_name} 生成超时")
            process = subprocess.CompletedProcess(cmd, -1, None, None)
            return process, None
        except KeyboardInterrupt:
            # 用户中断
            print(f"\n警告: {database_name} 生成被用户中断")
            process = subprocess.CompletedProcess(cmd, -15, None, None)
            return process, None
        except Exception as e:
            # 其他异常
            print(f"错误: {database_name} 生成时发生异常: {e}")
            process = subprocess.CompletedProcess(cmd, -1, None, None)
            return process, None

def main():
    parser = argparse.ArgumentParser(
        description='批量生成US数据集所有数据库的单数据库SQL',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 显示所有数据库状态
  python3 generate_all_us_databases.py --status
  
  # 生成所有需要生成的数据库（前台运行）
  python3 generate_all_us_databases.py
  
  # 生成所有需要生成的数据库（后台运行）
  python3 generate_all_us_databases.py --background
  
  # 只生成指定的数据库
  python3 generate_all_us_databases.py --databases "City of Austin - 1586" "City of Chicago - 854"
  
  # 跳过已完成的数据库
  python3 generate_all_us_databases.py --skip-completed
  
  # 设置自定义目标数量
  python3 generate_all_us_databases.py --target-count 250
        """
    )
    
    parser.add_argument('--database-dir', type=str, default=None,
                       help=f'数据库目录（默认: {DEFAULT_DATABASE_DIR}）')
    parser.add_argument('--skeleton-dir', type=str, default=None,
                       help=f'SQL骨架目录（默认: {DEFAULT_SKELETON_DIR}）')
    parser.add_argument('--graph-dir', type=str, default=None,
                       help=f'图文件目录（默认: {DEFAULT_GRAPH_DIR}）')
    parser.add_argument('--output-dir', type=str, default=None,
                       help=f'输出目录（默认: {DEFAULT_OUTPUT_DIR}）')
    parser.add_argument('--target-count', type=int, default=DEFAULT_TARGET_COUNT,
                       help=f'每个数据库的目标数量（默认: {DEFAULT_TARGET_COUNT}）')
    parser.add_argument('--max-retries', type=int, default=3,
                       help='最大重试次数（默认: 3）')
    parser.add_argument('--databases', type=str, nargs='+', default=None,
                       help='指定要生成的数据库列表（默认: 所有数据库）')
    parser.add_argument('--skip-completed', action='store_true',
                       help='跳过已完成的数据库')
    parser.add_argument('--background', action='store_true',
                       help='后台运行（每个数据库在后台生成）')
    parser.add_argument('--status', action='store_true',
                       help='只显示状态，不生成')
    parser.add_argument('--log-dir', type=str, default=None,
                       help='日志目录（后台运行时使用，默认: benchmark/generation/sql_filling/logs）')
    parser.add_argument('--max-workers', type=int, default=None,
                       help='每个数据库的最大并发数（默认: 从config.yaml读取，或20）')
    parser.add_argument('--retry-failed', action='store_true',
                       help='重试之前失败的数据库')
    parser.add_argument('--max-retry-attempts', type=int, default=2,
                       help='每个数据库的最大重试次数（默认: 2）')
    
    args = parser.parse_args()
    
    # 设置路径
    database_dir = Path(args.database_dir) if args.database_dir else DEFAULT_DATABASE_DIR
    skeleton_dir = Path(args.skeleton_dir) if args.skeleton_dir else DEFAULT_SKELETON_DIR
    graph_dir = Path(args.graph_dir) if args.graph_dir else DEFAULT_GRAPH_DIR
    output_dir = Path(args.output_dir) if args.output_dir else DEFAULT_OUTPUT_DIR
    script_path = Path(__file__).parent / "generate_us_single_db_sqls.py"
    
    # 检查脚本是否存在
    if not script_path.exists():
        print(f"错误: 生成脚本不存在: {script_path}")
        return 1
    
    # 获取所有数据库
    all_databases = get_all_databases(skeleton_dir)
    if not all_databases:
        print(f"错误: 未找到任何数据库骨架文件在 {skeleton_dir}")
        return 1
    
    # 如果指定了数据库列表，只处理这些数据库
    if args.databases:
        databases_to_process = [db for db in args.databases if db in all_databases]
        if not databases_to_process:
            print(f"错误: 指定的数据库都不存在")
            print(f"可用的数据库: {', '.join(all_databases[:5])}...")
            return 1
    else:
        databases_to_process = all_databases
    
    # 获取状态
    databases_status = []
    for db_name in databases_to_process:
        status = get_database_status(db_name, skeleton_dir, output_dir, args.target_count)
        databases_status.append(status)
    
    # 打印状态
    print_status_table(databases_status)
    
    # 如果只是显示状态，直接返回
    if args.status:
        return 0
    
    # 筛选需要生成的数据库
    databases_to_generate = []
    for status in databases_status:
        if args.skip_completed and status['completed']:
            continue
        if not status['skeleton_exists']:
            print(f"警告: {status['database']} 的骨架文件不存在，跳过")
            continue
        if status['need_count'] > 0:
            databases_to_generate.append(status)
    
    if not databases_to_generate:
        print("\n所有数据库都已完成，无需生成")
        return 0
    
    print(f"\n需要生成的数据库: {len(databases_to_generate)} 个")
    print(f"目标数量: {args.target_count} 条/数据库")
    print(f"最大重试次数: {args.max_retries}")
    print(f"运行模式: {'后台运行' if args.background else '前台运行'}")
    
    if not args.background:
        # 前台运行：逐个处理
        confirm = input("\n是否开始生成？(y/n): ")
        if confirm.lower() != 'y':
            print("已取消")
            return 0
        
        print("\n开始生成...\n")
        
        success_count = 0
        fail_count = 0
        
        failed_databases = []  # 记录失败的数据库，用于重试
        
        for i, status in enumerate(databases_to_generate, 1):
            db_name = status['database']
            print(f"\n[{i}/{len(databases_to_generate)}] 处理数据库: {db_name}")
            print(f"当前: {status['current_count']}, 目标: {status['target_count']}, 还需: {status['need_count']}")
            
            # 重试逻辑
            retry_count = 0
            max_retry_attempts = args.max_retry_attempts
            success = False
            
            while retry_count <= max_retry_attempts and not success:
                if retry_count > 0:
                    print(f"  重试第 {retry_count}/{max_retry_attempts} 次...")
                    # 重试前等待一段时间，避免资源竞争
                    import time
                    time.sleep(5)
                
                try:
                    process, _ = generate_sql_for_database(
                        db_name, script_path, database_dir, skeleton_dir,
                        graph_dir, output_dir, args.target_count, args.max_retries,
                        background=False, max_workers=args.max_workers
                    )
                    
                    if process.returncode == 0:
                        success = True
                        success_count += 1
                        print(f"✅ {db_name} 生成完成")
                    else:
                        # 检查返回码
                        if process.returncode == -15:
                            print(f"⚠️  {db_name} 进程被终止 (可能是资源不足或超时)")
                        elif process.returncode < 0:
                            print(f"⚠️  {db_name} 进程异常退出 (信号: {abs(process.returncode)})")
                        else:
                            print(f"⚠️  {db_name} 生成失败 (返回码: {process.returncode})")
                        
                        # 检查是否已达到目标数量（即使进程失败，可能已经生成了一些）
                        current_count = count_existing_sqls(output_dir, db_name)
                        if current_count >= args.target_count:
                            print(f"   但已达到目标数量 ({current_count}/{args.target_count})，视为成功")
                            success = True
                            success_count += 1
                        elif retry_count < max_retry_attempts:
                            retry_count += 1
                            continue
                        else:
                            failed_databases.append(db_name)
                            fail_count += 1
                            print(f"❌ {db_name} 生成失败，已重试 {max_retry_attempts} 次")
                
                except KeyboardInterrupt:
                    print(f"\n⚠️  用户中断，{db_name} 生成被取消")
                    failed_databases.append(db_name)
                    break
                except Exception as e:
                    print(f"❌ {db_name} 生成时发生异常: {e}")
                    if retry_count < max_retry_attempts:
                        retry_count += 1
                        continue
                    else:
                        failed_databases.append(db_name)
                        fail_count += 1
        
        print("\n" + "=" * 100)
        print("生成完成")
        print(f"成功: {success_count} 个")
        print(f"失败: {fail_count} 个")
        if failed_databases:
            print(f"\n失败的数据库: {', '.join(failed_databases)}")
            print("提示: 可以使用 --databases 参数单独重试失败的数据库")
        print("=" * 100)
        
    else:
        # 后台运行：同时启动所有任务
        confirm = input("\n是否开始后台生成？(y/n): ")
        if confirm.lower() != 'y':
            print("已取消")
            return 0
        
        print("\n启动后台任务...\n")
        
        processes = []
        log_files = []
        
        for status in databases_to_generate:
            db_name = status['database']
            print(f"启动: {db_name} (还需: {status['need_count']} 条)")
            
            process, log_file = generate_sql_for_database(
                db_name, script_path, database_dir, skeleton_dir,
                graph_dir, output_dir, args.target_count, args.max_retries,
                background=True, log_dir=args.log_dir, max_workers=args.max_workers
            )
            
            processes.append((db_name, process))
            if log_file:
                log_files.append((db_name, log_file))
                print(f"  进程ID: {process.pid}")
                print(f"  日志文件: {log_file}")
            print()
        
        print("=" * 100)
        print(f"已启动 {len(processes)} 个后台任务")
        print("=" * 100)
        print("\n查看任务状态:")
        print("  ps aux | grep 'generate_us_single_db_sqls.py' | grep -v grep")
        print("\n查看日志:")
        for db_name, log_file in log_files[:5]:  # 只显示前5个
            print(f"  tail -f {log_file}")
        if len(log_files) > 5:
            print(f"  ... 还有 {len(log_files) - 5} 个日志文件")
        print()
    
    return 0

if __name__ == '__main__':
    sys.exit(main())

