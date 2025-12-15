#!/usr/bin/env python3
"""
生成更多JOIN版本的SQL（简化版，直接调用函数，不阻塞）

基于已有结果，只生成新的SQL，避免重复
"""

import os
import json
import argparse
import random
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
import sys

# 导入SQL填充模块
sys.path.insert(0, os.path.dirname(__file__))
from cross_db_2fill_sql_placeholders_join import (
    load_multiple_schemas,
    process_cross_database_skeleton
)

# 目标数量
TARGET_COUNTS = {
    2: 359,  # 跨2个数据库
    3: 105,  # 跨3个数据库
    4: 2     # 跨4个数据库
}

def get_existing_skeletons(sql_dir):
    """获取已使用的骨架标识（基于已有结果的SQL文件）"""
    existing_skeleton_signatures = set()
    
    for f in os.listdir(sql_dir):
        if f.startswith('cross_db_generated_sql_') and f.endswith('.json'):
            file_path = os.path.join(sql_dir, f)
            try:
                with open(file_path, 'r', encoding='utf-8') as file:
                    data = json.load(file)
                    results = data.get('results', [])
                    
                    if results is not None and len(results) > 0:
                        databases = sorted(data.get('databases', []))
                        table_db_mapping = data.get('table_database_mapping', {})
                        tables = sorted(table_db_mapping.keys())
                        
                        if len(databases) >= 2 and len(tables) >= 2:
                            table1 = tables[0]
                            table2 = tables[1] if len(tables) > 1 else None
                            if table2:
                                signature = tuple(sorted(databases[:2]) + sorted([table1, table2]))
                                existing_skeleton_signatures.add(signature)
            except:
                pass
    
    return existing_skeleton_signatures

def count_needed_by_db_count(sql_dir):
    """统计每个数据库数量类别还需要多少个"""
    needed = {}
    
    for db_count in [2, 3, 4]:
        target = TARGET_COUNTS[db_count]
        current = 0
        
        for f in os.listdir(sql_dir):
            if f.startswith('cross_db_generated_sql_') and f.endswith('.json'):
                try:
                    file_path = os.path.join(sql_dir, f)
                    with open(file_path, 'r', encoding='utf-8') as file:
                        data = json.load(file)
                        results = data.get('results', [])
                        num_databases = data.get('metadata', {}).get('num_databases', 
                                                                    len(data.get('databases', [])))
                        
                        if results is not None and len(results) > 0 and num_databases == db_count:
                            current += 1
                except:
                    pass
        
        needed[db_count] = max(0, target - current)
    
    return needed

def filter_skeletons_by_db_count(skeletons, db_count):
    """过滤出指定数据库数量的骨架"""
    return [s for s in skeletons if s.get('num_databases', len(s.get('databases', []))) == db_count]

def generate_more_sqls(skeleton_file, sql_dir, graph_dir, output_dir, database_dir, 
                       max_workers=5, max_retries=3, ignore_existing=False):
    """生成更多SQL（直接调用函数，不阻塞）
    
    Args:
        ignore_existing: 如果为True，忽略已有结果，重新生成（用于清空后重新生成）
    """
    
    if ignore_existing:
        # 忽略已有结果，直接生成一批新的
        print("=" * 70)
        print("重新生成JOIN版本SQL（忽略已有结果）")
        print("=" * 70)
        print("\n将生成一批新的SQL，使用不同的骨架...")
        needed = {2: 100}  # 每次生成约100个，成功率约22.5%，预计得到20-25个有结果的
    else:
        # 1. 统计还需要多少个
        needed = count_needed_by_db_count(sql_dir)
        
        print("=" * 70)
        print("生成更多JOIN版本SQL")
        print("=" * 70)
        print("\n目标数量:")
        for db_count in sorted(needed.keys()):
            target = TARGET_COUNTS[db_count]
            current = target - needed[db_count]
            print(f"  跨{db_count}个数据库: {current} / {target} (还需要 {needed[db_count]})")
        
        total_needed = sum(needed.values())
        if total_needed == 0:
            print("\n✅ 所有目标数量都已达到，无需生成更多SQL")
            return
        
        print(f"\n总计还需要: {total_needed} 个SQL")
    
    # 2. 加载骨架文件
    print("\n加载SQL骨架...")
    with open(skeleton_file, 'r', encoding='utf-8') as f:
        all_skeletons = json.load(f)
    
    print(f"  总骨架数: {len(all_skeletons)}")
    
    # 3. 获取已使用的骨架标识（如果ignore_existing为True，则跳过）
    if ignore_existing:
        existing_skeleton_signatures = set()
        print(f"  忽略已有结果，将使用所有可用骨架")
    else:
        existing_skeleton_signatures = get_existing_skeletons(sql_dir)
        print(f"  已使用的骨架数: {len(existing_skeleton_signatures)}")
    
    # 4. 按数据库数量分类骨架
    skeletons_by_db_count = {}
    for db_count in [2, 3, 4]:
        skeletons_by_db_count[db_count] = filter_skeletons_by_db_count(all_skeletons, db_count)
        print(f"  {db_count}个数据库的骨架: {len(skeletons_by_db_count[db_count])} 个")
    
    # 5. 为每个数据库数量类别选择未使用的骨架
    selected_skeletons = []
    
    for db_count in sorted(needed.keys()):
        if needed[db_count] == 0:
            continue
        
        available_skeletons = skeletons_by_db_count[db_count]
        
        if ignore_existing:
            # 忽略已有结果，直接随机选择
            unused_skeletons = available_skeletons
        else:
            # 过滤出未使用的骨架
            unused_skeletons = []
            for skeleton in available_skeletons:
                databases = sorted(skeleton.get('databases', []))
                table_db_mapping = skeleton.get('table_database_mapping', {})
                tables = sorted(table_db_mapping.keys())
                
                if len(databases) >= 2 and len(tables) >= 2:
                    table1 = tables[0]
                    table2 = tables[1] if len(tables) > 1 else None
                    if table2:
                        signature = tuple(sorted(databases[:2]) + sorted([table1, table2]))
                        if signature not in existing_skeleton_signatures:
                            unused_skeletons.append(skeleton)
        
        print(f"\n跨{db_count}个数据库:")
        print(f"  可用骨架: {len(available_skeletons)}")
        print(f"  未使用骨架: {len(unused_skeletons)}")
        print(f"  需要生成: {needed[db_count]}")
        
        # 选择需要的数量（生成4倍，因为成功率约22.5%）
        to_generate = min(needed[db_count] * 4, len(unused_skeletons))  # 4倍更保险
        if to_generate > 0:
            selected = random.sample(unused_skeletons, to_generate) if len(unused_skeletons) >= to_generate else unused_skeletons
            print(f"  选择生成: {len(selected)} 个骨架")
            selected_skeletons.extend(selected)
    
    if not selected_skeletons:
        print("\n⚠️  没有可用的未使用骨架，无法生成更多SQL")
        return
    
    print(f"\n总计选择: {len(selected_skeletons)} 个骨架用于生成")
    
    # 6. 加载所有数据库的schema
    print("\n加载数据库schema...")
    all_databases = set()
    for skeleton in selected_skeletons:
        all_databases.update(skeleton.get('databases', []))
    
    schemas = load_multiple_schemas(all_databases, database_dir)
    print(f"成功加载 {len(schemas)} 个数据库的schema")
    
    # 7. 直接处理每个骨架（不使用subprocess，避免阻塞）
    print(f"\n{'='*70}")
    print(f"开始生成SQL（共 {len(selected_skeletons)} 个）...")
    print(f"并发数: {max_workers}, 最大重试: {max_retries}")
    print(f"{'='*70}\n")
    
    success_count = 0
    failed_count = 0
    results_with_data = 0  # 有结果的SQL数量
    
    import time
    start_time = time.time()
    last_print_time = start_time
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = []
        for i, skeleton in enumerate(selected_skeletons):
            future = executor.submit(
                process_cross_database_skeleton,
                skeleton, schemas, graph_dir, output_dir,
                database_dir, max_retries
            )
            futures.append((future, i+1))
        
        # 收集结果（显示进度条和实时统计）
        completed = 0
        pbar = tqdm(total=len(futures), desc="生成进度", ncols=120, unit="个", 
                   bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]')
        
        for future, idx_num in futures:
            try:
                # 设置超时，避免单个任务卡住太久
                idx, success, message = future.result(timeout=600)  # 10分钟超时（单个SQL生成）
                completed += 1
                pbar.update(1)
                
                if success:
                    success_count += 1
                    # 检查是否有结果数据
                    try:
                        output_file = os.path.join(output_dir, f"cross_db_generated_sql_{idx}.json")
                        if os.path.exists(output_file):
                            with open(output_file, 'r', encoding='utf-8') as f:
                                data = json.load(f)
                                results = data.get('results', [])
                                if results and len(results) > 0:
                                    results_with_data += 1
                    except:
                        pass
                else:
                    failed_count += 1
                
                # 每5秒或每10个打印一次详细统计（确保用户看到进度）
                current_time = time.time()
                if (current_time - last_print_time >= 5) or (completed % 10 == 0) or (completed == len(selected_skeletons)):
                    elapsed = current_time - start_time
                    rate = completed / elapsed if elapsed > 0 else 0
                    remaining = (len(selected_skeletons) - completed) / rate if rate > 0 else 0
                    pbar.set_postfix({
                        '成功': f'{success_count}',
                        '有结果': f'{results_with_data}',
                        '失败': f'{failed_count}',
                        '速度': f'{rate:.2f}/s'
                    })
                    # 额外打印一行详细日志（确保能看到）
                    if current_time - last_print_time >= 5:
                        print(f"\n[实时] 已完成: {completed}/{len(selected_skeletons)} | "
                              f"成功: {success_count} (有结果: {results_with_data}) | "
                              f"失败: {failed_count} | "
                              f"速度: {rate:.2f}个/秒 | "
                              f"预计剩余: {remaining/60:.1f}分钟", flush=True)
                        last_print_time = current_time
                    
            except Exception as e:
                failed_count += 1
                completed += 1
                pbar.update(1)
                print(f"\n[异常 #{completed}] {str(e)[:200]}", flush=True)
        
        pbar.close()
    
    elapsed_time = time.time() - start_time
    print(f"\n" + "=" * 70)
    print(f"生成完成!")
    print(f"总耗时: {elapsed_time/60:.2f} 分钟")
    print(f"成功: {success_count}/{len(selected_skeletons)} ({success_count/len(selected_skeletons)*100:.1f}%)")
    print(f"有结果: {results_with_data}/{len(selected_skeletons)} ({results_with_data/len(selected_skeletons)*100:.1f}%)")
    print(f"失败: {failed_count}/{len(selected_skeletons)} ({failed_count/len(selected_skeletons)*100:.1f}%)")
    print("=" * 70)

def main():
    parser = argparse.ArgumentParser(description='生成更多JOIN版本的SQL（不重复已有的）')
    parser.add_argument('--skeleton_file', type=str,
                       default='benchmark/generation/cross_database/cross_db_skeletons_join.json',
                       help='SQL骨架文件')
    parser.add_argument('--graph_dir', type=str,
                       default='benchmark/data/beijing/output/cross_db_graph_join',
                       help='图文件目录')
    parser.add_argument('--database_dir', type=str,
                       default='benchmark/data/beijing/database_chinese',
                       help='数据库目录')
    parser.add_argument('--output_dir', type=str,
                       default='benchmark/data/beijing/output/cross_db_single_join',
                       help='输出目录')
    parser.add_argument('--sql_dir', type=str,
                       default='benchmark/data/beijing/output/cross_db_single_join',
                       help='SQL文件目录（用于统计已有结果）')
    parser.add_argument('--max_retries', type=int, default=3,
                       help='最大重试次数')
    parser.add_argument('--max_workers', type=int, default=5,
                       help='并发线程数')
    parser.add_argument('--ignore_existing', action='store_true',
                       help='忽略已有结果，重新生成（用于清空后重新生成）')
    
    args = parser.parse_args()
    
    generate_more_sqls(
        args.skeleton_file,
        args.sql_dir,
        args.graph_dir,
        args.output_dir,
        args.database_dir,
        args.max_workers,
        args.max_retries,
        args.ignore_existing
    )

if __name__ == '__main__':
    main()

