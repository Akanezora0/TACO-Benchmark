#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
US数据集跨数据库SQL生成 - 一键运行脚本

支持分步执行或完整流程
"""

import os
import sys
import argparse
import subprocess
from pathlib import Path

# 项目根目录
PROJECT_ROOT = Path(__file__).parent.parent.parent.parent

# 脚本目录
SCRIPT_DIR = Path(__file__).parent

def run_step(step_num, step_name, script_name, description=""):
    """运行单个步骤"""
    print("\n" + "=" * 80)
    print(f"步骤 {step_num}: {step_name}")
    if description:
        print(f"说明: {description}")
    print("=" * 80)
    
    script_path = SCRIPT_DIR / script_name
    if not script_path.exists():
        print(f"错误: 脚本不存在: {script_path}")
        return False
    
    try:
        result = subprocess.run(
            [sys.executable, str(script_path)],
            cwd=str(SCRIPT_DIR),
            check=False
        )
        return result.returncode == 0
    except Exception as e:
        print(f"错误: 运行脚本时发生异常: {e}")
        return False

def main():
    parser = argparse.ArgumentParser(
        description='US数据集跨数据库SQL生成 - 一键运行',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
步骤说明:
  1. analyze_joinable_tables - 分析可JOIN表对
  2. generate_skeletons - 生成SQL骨架
  3. build_graphs - 生成Schema图
  4. generate_2db - 生成2数据库SQL
  5. generate_3db_4db - 生成3和4数据库SQL

示例:
  # 查看状态
  python3 run_all.py --status
  
  # 运行所有步骤
  python3 run_all.py
  
  # 只运行步骤1
  python3 run_all.py --step 1
  
  # 从步骤3开始运行
  python3 run_all.py --from-step 3
        """
    )
    
    parser.add_argument('--status', action='store_true',
                       help='只显示状态，不执行')
    parser.add_argument('--step', type=int, default=None,
                       help='只执行指定步骤（1-5）')
    parser.add_argument('--from-step', type=int, default=1,
                       help='从指定步骤开始执行（默认: 1）')
    parser.add_argument('--to-step', type=int, default=5,
                       help='执行到指定步骤（默认: 5）')
    parser.add_argument('--skip-step', type=int, nargs='+', default=[],
                       help='跳过指定步骤')
    
    args = parser.parse_args()
    
    # 如果只是查看状态
    if args.status:
        status_script = SCRIPT_DIR / "0check_status.py"
        if status_script.exists():
            subprocess.run([sys.executable, str(status_script)], cwd=str(SCRIPT_DIR))
        else:
            print("错误: 状态检查脚本不存在")
        return
    
    # 定义步骤
    steps = [
        (1, "分析可JOIN表对", "1analyze_joinable_tables.py", "分析US数据集中的表，找出可以JOIN的表对"),
        (2, "生成SQL骨架", "2generate_cross_db_skeletons_join.py", "基于可JOIN表对生成JOIN SQL骨架"),
        (3, "生成Schema图", "3build_schema_graphs.py", "为SQL骨架生成Schema Linking Graph"),
        (4, "生成2数据库SQL", "4generate_2db_sqls.py", "批量生成2个数据库的JOIN SQL"),
        (5, "生成3和4数据库SQL", "5generate_3db_4db_sqls.py", "生成3个和4个数据库的JOIN SQL"),
    ]
    
    # 确定要执行的步骤
    if args.step:
        steps_to_run = [s for s in steps if s[0] == args.step]
    else:
        steps_to_run = [
            s for s in steps 
            if args.from_step <= s[0] <= args.to_step 
            and s[0] not in args.skip_step
        ]
    
    if not steps_to_run:
        print("没有要执行的步骤")
        return
    
    print("=" * 80)
    print("US数据集跨数据库SQL生成 - 一键运行")
    print("=" * 80)
    print(f"\n将执行以下步骤:")
    for step_num, step_name, _, _ in steps_to_run:
        print(f"  {step_num}. {step_name}")
    
    confirm = input("\n是否开始执行？(y/n): ")
    if confirm.lower() != 'y':
        print("已取消")
        return
    
    # 执行步骤
    success_count = 0
    fail_count = 0
    
    for step_num, step_name, script_name, description in steps_to_run:
        success = run_step(step_num, step_name, script_name, description)
        if success:
            success_count += 1
            print(f"✅ 步骤 {step_num} 完成")
        else:
            fail_count += 1
            print(f"❌ 步骤 {step_num} 失败")
            
            # 询问是否继续
            if step_num < steps_to_run[-1][0]:
                continue_choice = input(f"\n步骤 {step_num} 失败，是否继续执行后续步骤？(y/n): ")
                if continue_choice.lower() != 'y':
                    break
    
    # 总结
    print("\n" + "=" * 80)
    print("执行完成")
    print(f"成功: {success_count} 个步骤")
    print(f"失败: {fail_count} 个步骤")
    print("=" * 80)
    
    # 显示最终状态
    print("\n查看最终状态:")
    status_script = SCRIPT_DIR / "0check_status.py"
    if status_script.exists():
        subprocess.run([sys.executable, str(status_script)], cwd=str(SCRIPT_DIR))

if __name__ == '__main__':
    main()

