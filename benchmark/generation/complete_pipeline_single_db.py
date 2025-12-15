#!/usr/bin/env python3
"""
单数据库完整流程脚本
流程：SQL骨架生成 → 生成图 → 填充SQL → 生成NL查询

策略：
- 生成3倍于目标数量的SQL骨架
- 填充SQL直到达到目标数量就停止
- 一次性生成所有NL查询
"""

import os
import sys
import json
import subprocess
import shutil
from pathlib import Path
from typing import List, Dict
import argparse

# 项目根目录
PROJECT_ROOT = Path(__file__).parent.parent.parent

# 添加模块路径
sys.path.insert(0, str(PROJECT_ROOT / "benchmark/generation/sql_skeleton_generation"))
sys.path.insert(0, str(PROJECT_ROOT / "benchmark/generation/sql_filling"))
sys.path.insert(0, str(PROJECT_ROOT / "benchmark/generation/nl_query"))

def run_command(cmd: List[str], cwd: str = None, check: bool = True) -> bool:
    """运行命令"""
    try:
        result = subprocess.run(
            cmd,
            cwd=cwd or PROJECT_ROOT,
            check=check,
            capture_output=True,
            text=True
        )
        if result.returncode != 0:
            print(f"命令执行失败: {' '.join(cmd)}")
            print(f"错误输出: {result.stderr}")
            return False
        return True
    except Exception as e:
        print(f"执行命令时出错: {e}")
        return False

def count_successful_sqls(sql_dir: str, database: str) -> int:
    """统计已成功生成的SQL数量"""
    db_dir = os.path.join(sql_dir, database)
    if not os.path.exists(db_dir):
        return 0
    
    count = 0
    for file in os.listdir(db_dir):
        if file.startswith('generated_sql_') and file.endswith('.json') and '_error' not in file:
            file_path = os.path.join(db_dir, file)
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    # 检查是否有有效的SQL和结果
                    if data.get('sql') and data.get('results') is not None:
                        count += 1
            except:
                pass
    return count

def step1_generate_skeletons(database: str, target_count: int, output_dir: str, 
                             database_dir: str, expert_file: str, old_cfg_file: str = None,
                             old_data_file: str = None, new_logs_file: str = None) -> bool:
    """步骤1: 生成SQL骨架（生成4倍数量，确保多样性）"""
    print(f"\n{'='*80}")
    print(f"步骤1: 生成SQL骨架 - {database}")
    print(f"{'='*80}")
    
    # 检查当前已生成的SQL数量
    current_sql_count = count_successful_sqls(os.path.join(output_dir, "single"), database)
    need_sql = max(0, target_count - current_sql_count)
    
    if need_sql == 0:
        print(f"✅ 当前已有 {current_sql_count} 条SQL，已达到目标 {target_count} 条，跳过SQL骨架生成")
        return True
    
    # 计算骨架数量（4倍策略，确保足够多）
    skeleton_count = need_sql * 4
    # 为了确保多样性，结构数量应该足够多，但不要超过合理范围
    # 结构数量应该是骨架数量的1.2-1.5倍，但最多不超过5000
    structure_count = min(int(skeleton_count * 1.2), 5000)
    # 确保至少是骨架数量
    structure_count = max(structure_count, skeleton_count)
    
    print(f"目标SQL数量: {target_count}条")
    print(f"当前SQL数量: {current_sql_count}条")
    print(f"还需生成SQL: {need_sql}条")
    print(f"生成SQL骨架数量: {skeleton_count}条（4倍策略）")
    print(f"生成SQL结构数量: {structure_count}条（确保多样性，上限5000）")
    
    try:
        # 导入SQL骨架生成模块
        from generate_for_databases_improved import (
            generate_cfg_for_database,
            generate_structures_for_database,
            generate_skeletons_for_database
        )
        
        # 创建输出目录
        cfg_dir = os.path.join(output_dir, 'ast_cfg')
        structure_dir = os.path.join(output_dir, 'sql_structure')
        skeleton_dir = os.path.join(output_dir, 'sql_skeleton')
        
        os.makedirs(cfg_dir, exist_ok=True)
        os.makedirs(structure_dir, exist_ok=True)
        os.makedirs(skeleton_dir, exist_ok=True)
        
        # 步骤1.1: 生成CFG
        cfg_file = os.path.join(cfg_dir, f"{database}_ast_cfg.json")
        print(f"  生成CFG文件...")
        try:
            count = generate_cfg_for_database(expert_file, old_cfg_file, cfg_file, database)
            print(f"  ✓ CFG生成成功: {count}条")
        except Exception as e:
            print(f"  ✗ CFG生成失败: {e}")
            return False
        
        # 步骤1.2: 生成SQL结构（确保多样性）
        structure_file = os.path.join(structure_dir, f"{database}_structure.json")
        print(f"  生成SQL结构（数量: {structure_count}，确保多样性）...")
        try:
            structures = generate_structures_for_database(cfg_file, old_cfg_file, structure_file, database, structure_count)
            print(f"  ✓ SQL结构生成成功: {len(structures)}个")
        except Exception as e:
            print(f"  ✗ SQL结构生成失败: {e}")
            return False
        
        # 步骤1.3: 生成SQL骨架（确保多样性，使用更多结构）
        skeleton_file = os.path.join(skeleton_dir, f"{database}_sql_skeleton.json")
        print(f"  生成SQL骨架（目标: {skeleton_count}条，确保多样性）...")
        try:
            # 使用更平衡的难度比例，确保多样性
            count = generate_skeletons_for_database(
                structure_file, skeleton_file, old_data_file, new_logs_file, 
                skeleton_count, database, 0.4, 0.4, 0.2  # 调整比例，增加中等和复杂查询
            )
            print(f"  ✓ SQL骨架生成成功: {count}个")
        except Exception as e:
            print(f"  ✗ SQL骨架生成失败: {e}")
            return False
        
        # 检查输出
        if os.path.exists(skeleton_file):
            with open(skeleton_file, 'r', encoding='utf-8') as f:
                skeletons = json.load(f)
            print(f"✅ SQL骨架生成成功: {len(skeletons)}条")
            return True
        else:
            print(f"❌ SQL骨架文件不存在: {skeleton_file}")
            return False
            
    except ImportError as e:
        print(f"❌ 导入模块失败: {e}")
        print("尝试使用命令行方式...")
        # Fallback到命令行方式
        script_path = PROJECT_ROOT / "benchmark/generation/sql_skeleton_generation/generate_for_databases_improved.py"
        cmd = [
            "python3", str(script_path),
            "--total_skeletons", str(skeleton_count),
            "--num_samples", str(structure_count),
            "--simple_ratio", "0.4",
            "--medium_ratio", "0.4",
            "--complex_ratio", "0.2",
        ]
        # 注意：命令行方式会为所有数据库生成，这里只是fallback
        if not run_command(cmd):
            return False
        
        skeleton_dir = os.path.join(output_dir, "sql_skeleton")
        skeleton_file = os.path.join(skeleton_dir, f"{database}_sql_skeleton.json")
        
        if os.path.exists(skeleton_file):
            with open(skeleton_file, 'r', encoding='utf-8') as f:
                skeletons = json.load(f)
            print(f"✅ SQL骨架生成成功: {len(skeletons)}条")
            return True
        else:
            print(f"❌ SQL骨架文件不存在: {skeleton_file}")
            return False

def step2_build_graphs(database: str, output_dir: str, schema_dir: str) -> bool:
    """步骤2: 生成Schema Linking Graph"""
    print(f"\n{'='*80}")
    print(f"步骤2: 生成Schema Linking Graph - {database}")
    print(f"{'='*80}")
    
    skeleton_dir = os.path.join(output_dir, "sql_skeleton")
    skeleton_file = os.path.join(skeleton_dir, f"{database}_sql_skeleton.json")
    graph_dir = os.path.join(output_dir, "graph")
    schema_file = os.path.join(schema_dir, database, f"{database}.json")
    
    if not os.path.exists(skeleton_file):
        print(f"❌ SQL骨架文件不存在: {skeleton_file}")
        return False
    
    if not os.path.exists(schema_file):
        print(f"❌ Schema文件不存在: {schema_file}")
        return False
    
    try:
        # 导入Graph生成模块
        from importlib import import_module
        graph_module = import_module('1build_schema_graphs_improved')
        process_database = graph_module.process_database
        
        # 调用处理函数
        process_database(database, skeleton_file, schema_file, graph_dir)
        
        # 检查输出（Graph文件是.graphml格式，保存在子目录中）
        db_graph_dir = os.path.join(graph_dir, database)
        if os.path.exists(db_graph_dir):
            graph_files = [f for f in os.listdir(db_graph_dir) if f.endswith('.graphml')]
            print(f"✅ Graph生成成功: {len(graph_files)}个")
            return True
        else:
            print(f"❌ Graph目录不存在: {db_graph_dir}")
            return False
            
    except Exception as e:
        print(f"❌ Graph生成失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def step3_fill_sqls(database: str, target_count: int, output_dir: str, schema_dir: str) -> bool:
    """步骤3: 填充SQL（直到达到目标数量）"""
    print(f"\n{'='*80}")
    print(f"步骤3: 填充SQL - {database}")
    print(f"{'='*80}")
    
    skeleton_dir = os.path.join(output_dir, "sql_skeleton")
    skeleton_file = os.path.join(skeleton_dir, f"{database}_sql_skeleton.json")
    graph_dir = os.path.join(output_dir, "graph")
    schema_file = os.path.join(schema_dir, database, f"{database}.json")
    
    if not os.path.exists(skeleton_file):
        print(f"❌ SQL骨架文件不存在: {skeleton_file}")
        return False
    
    if not os.path.exists(schema_file):
        print(f"❌ Schema文件不存在: {schema_file}")
        return False
    
    # 检查当前已生成的SQL数量
    current_count = count_successful_sqls(os.path.join(output_dir, "single"), database)
    print(f"当前已生成SQL数量: {current_count}条")
    
    if current_count >= target_count:
        print(f"✅ 已达到目标数量，跳过填充步骤")
        return True
    
    needed = target_count - current_count
    print(f"需要生成: {needed}条")
    
    try:
        # 导入SQL填充模块
        from importlib import import_module
        fill_module = import_module('2fill_sql_placeholders_improved')
        process_single_sql_skeleton = fill_module.process_single_sql_skeleton
        load_schema = fill_module.load_schema
        extract_schema_info = fill_module.extract_schema_info
        
        # 加载schema
        schema = load_schema(schema_file)
        schema_info = extract_schema_info(schema)
        
        # 加载SQL骨架，只处理前needed*3个（考虑成功率）
        with open(skeleton_file, 'r', encoding='utf-8') as f:
            all_skeletons = json.load(f)
        
        skeletons_to_process = all_skeletons[:needed * 4]  # 处理4倍数量以确保成功率
        print(f"处理 {len(skeletons_to_process)} 个SQL骨架（目标: {needed}条成功SQL）")
        
        # 创建输出目录
        single_output_path = os.path.join(output_dir, 'single', database)
        os.makedirs(single_output_path, exist_ok=True)
        
        # 逐个处理，达到目标数量就停止
        from concurrent.futures import ThreadPoolExecutor, as_completed
        from tqdm import tqdm
        
        success_count = 0
        fail_count = 0
        max_workers = 5
        max_retries = 3
        
        # 准备任务
        tasks = []
        for idx, sql_skeleton in enumerate(skeletons_to_process):
            tasks.append((
                idx, sql_skeleton, database, schema, schema_info,
                graph_dir, single_output_path, schema_file, max_retries
            ))
        
        # 并发处理，但监控成功数量
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_idx = {executor.submit(process_single_sql_skeleton, task): task[0] for task in tasks}
            
            with tqdm(total=len(tasks), desc=f"{database} 填充进度") as pbar:
                for future in as_completed(future_to_idx):
                    idx = future_to_idx[future]
                    try:
                        result_idx, success, message = future.result()
                        if success:
                            if message != "已存在":
                                success_count += 1
                                # 检查是否达到目标
                                current = count_successful_sqls(os.path.join(output_dir, "single"), database)
                                if current >= target_count:
                                    print(f"\n✅ 已达到目标数量 {target_count}条，停止处理")
                                    # 取消剩余任务
                                    for f in future_to_idx:
                                        f.cancel()
                                    break
                        else:
                            fail_count += 1
                    except Exception as e:
                        fail_count += 1
                        print(f"处理索引 {idx} 时发生异常: {e}")
                    finally:
                        pbar.update(1)
        
        # 检查最终结果
        final_count = count_successful_sqls(os.path.join(output_dir, "single"), database)
        print(f"填充后SQL数量: {final_count}条 (本次成功: {success_count}, 失败: {fail_count})")
        
        if final_count >= target_count:
            print(f"✅ 已达到目标数量")
            return True
        else:
            print(f"⚠️  未完全达到目标，当前: {final_count}条，目标: {target_count}条")
            return final_count > 0  # 至少有一些成功
            
    except Exception as e:
        print(f"❌ SQL填充失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def step4_generate_nl_queries(database: str, target_count: int, output_dir: str, schema_dir: str) -> bool:
    """步骤4: 生成NL查询"""
    print(f"\n{'='*80}")
    print(f"步骤4: 生成NL查询 - {database}")
    print(f"{'='*80}")
    
    script_path = PROJECT_ROOT / "benchmark/generation/nl_query/4generate_nl_queries_improved.py"
    sql_dir = os.path.join(output_dir, "single")
    nl_output_dir = os.path.join(output_dir, "nl_query")
    
    # 检查当前已生成的SQL数量
    current_sql_count = count_successful_sqls(sql_dir, database)
    print(f"可用SQL数量: {current_sql_count}条")
    
    if current_sql_count == 0:
        print(f"❌ 没有可用的SQL，跳过NL查询生成")
        return False
    
    # 检查已生成的NL查询数量
    nl_db_dir = os.path.join(nl_output_dir, database)
    existing_nl_count = 0
    if os.path.exists(nl_db_dir):
        existing_nl_count = len([f for f in os.listdir(nl_db_dir) 
                                 if f.startswith('generated_nl_query_') and f.endswith('.json')])
    
    # 计算需要生成的数量（目标 - 当前）
    needed = max(0, target_count - existing_nl_count)
    
    if needed == 0:
        print(f"✅ 已达到目标数量（{existing_nl_count}/{target_count}条），跳过NL查询生成")
        return True
    
    print(f"目标NL查询数量: {target_count}条")
    print(f"当前NL查询数量: {existing_nl_count}条")
    print(f"还需生成: {needed}条")
    
    cmd = [
        "python3", str(script_path),
        "--sql_dir", sql_dir,
        "--schema_dir", schema_dir,
        "--output_dir", nl_output_dir,
        "--database", database,
        "--limit", str(needed),
        "--max_workers", "3"
    ]
    
    print(f"执行命令: {' '.join(cmd)}")
    
    if not run_command(cmd):
        return False
    
    # 检查结果
    final_nl_count = 0
    if os.path.exists(nl_db_dir):
        final_nl_count = len([f for f in os.listdir(nl_db_dir) 
                             if f.startswith('generated_nl_query_') and f.endswith('.json')])
    
    print(f"生成后NL查询数量: {final_nl_count}条")
    
    if final_nl_count >= target_count:
        print(f"✅ 已达到目标数量")
        return True
    else:
        print(f"⚠️  未完全达到目标，当前: {final_nl_count}条，目标: {target_count}条")
        return final_nl_count > 0

def main():
    parser = argparse.ArgumentParser(description='单数据库完整流程脚本')
    parser.add_argument('--database', type=str, required=True, help='数据库名称')
    parser.add_argument('--target_count', type=int, required=True, help='目标NL查询数量')
    parser.add_argument('--output_dir', type=str, 
                       default=str(PROJECT_ROOT / "benchmark/data/beijing/output"),
                       help='输出目录')
    parser.add_argument('--schema_dir', type=str,
                       default=str(PROJECT_ROOT / "benchmark/data/beijing/database_chinese"),
                       help='Schema目录')
    parser.add_argument('--database_dir', type=str,
                       default=str(PROJECT_ROOT / "benchmark/data/beijing/database"),
                       help='数据库目录（用于SQL骨架生成）')
    parser.add_argument('--expert_file', type=str,
                       default=str(PROJECT_ROOT / "benchmark/data/target/expert_skeletons_beijing.json"),
                       help='专家例子文件')
    parser.add_argument('--old_cfg_file', type=str, default=None,
                       help='旧数据库CFG文件（可选）')
    parser.add_argument('--old_data_file', type=str, default=None,
                       help='旧数据文件（可选）')
    parser.add_argument('--new_logs_file', type=str, default=None,
                       help='新日志文件（可选）')
    parser.add_argument('--skip_skeleton', action='store_true', help='跳过SQL骨架生成')
    parser.add_argument('--skip_graph', action='store_true', help='跳过Graph生成')
    parser.add_argument('--skip_fill', action='store_true', help='跳过SQL填充')
    parser.add_argument('--skip_nl', action='store_true', help='跳过NL查询生成')
    
    args = parser.parse_args()
    
    print("=" * 80)
    print("单数据库完整流程")
    print("=" * 80)
    print(f"数据库: {args.database}")
    print(f"目标NL查询数量: {args.target_count}条")
    print(f"输出目录: {args.output_dir}")
    print(f"Schema目录: {args.schema_dir}")
    print()
    
    success = True
    
    # 步骤1: 生成SQL骨架
    if not args.skip_skeleton:
        if not step1_generate_skeletons(
            args.database, args.target_count, args.output_dir,
            args.database_dir, args.expert_file, args.old_cfg_file,
            args.old_data_file, args.new_logs_file
        ):
            print("❌ SQL骨架生成失败")
            success = False
    else:
        print("⏭️  跳过SQL骨架生成")
    
    # 步骤2: 生成Graph
    if success and not args.skip_graph:
        if not step2_build_graphs(args.database, args.output_dir, args.schema_dir):
            print("❌ Graph生成失败")
            success = False
    else:
        print("⏭️  跳过Graph生成")
    
    # 步骤3: 填充SQL
    if success and not args.skip_fill:
        if not step3_fill_sqls(args.database, args.target_count, args.output_dir, args.schema_dir):
            print("❌ SQL填充失败")
            success = False
    else:
        print("⏭️  跳过SQL填充")
    
    # 步骤4: 生成NL查询
    if success and not args.skip_nl:
        if not step4_generate_nl_queries(args.database, args.target_count, args.output_dir, args.schema_dir):
            print("❌ NL查询生成失败")
            success = False
    else:
        print("⏭️  跳过NL查询生成")
    
    print("\n" + "=" * 80)
    if success:
        print("✅ 流程完成")
    else:
        print("❌ 流程失败")
    print("=" * 80)
    
    return 0 if success else 1

if __name__ == '__main__':
    sys.exit(main())

