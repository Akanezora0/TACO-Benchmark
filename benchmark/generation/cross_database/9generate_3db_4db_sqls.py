#!/usr/bin/env python3
"""
专门生成3个和4个数据库的JOIN SQL
"""

import os
import sys
import argparse

# 导入主生成脚本的函数
sys.path.insert(0, os.path.dirname(__file__))
import importlib.util
spec = importlib.util.spec_from_file_location(
    "generate_more_join_sqls_simple",
    os.path.join(os.path.dirname(__file__), "4generate_more_join_sqls_simple.py")
)
gen_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(gen_module)

load_multiple_schemas = gen_module.load_multiple_schemas

def main():
    # 获取脚本所在目录和项目根目录
    script_dir = os.path.dirname(os.path.abspath(__file__))
    # 项目根目录：从 cross_database 目录向上3级
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(script_dir)))
    
    parser = argparse.ArgumentParser(description='专门生成3个和4个数据库的JOIN SQL')
    parser.add_argument('--skeleton_file', type=str,
                       default=os.path.join(script_dir, 'cross_db_skeletons_join.json'),
                       help='SQL骨架文件')
    parser.add_argument('--graph_dir', type=str,
                       default=os.path.join(script_dir, 'cross_db_graphs_join'),
                       help='图文件目录')
    parser.add_argument('--database_dir', type=str,
                       default=os.path.join(project_root, 'benchmark', 'data', 'beijing', 'database_chinese'),
                       help='数据库目录')
    parser.add_argument('--output_dir', type=str,
                       default=os.path.join(project_root, 'benchmark', 'data', 'beijing', 'output', 'cross_db_single_join'),
                       help='输出目录')
    parser.add_argument('--sql_dir', type=str,
                       default=os.path.join(project_root, 'benchmark', 'data', 'beijing', 'output', 'cross_db_single_join'),
                       help='SQL文件目录（用于统计已有数量）')
    parser.add_argument('--max_workers', type=int, default=10,
                       help='最大并发数（默认10，3和4数据库的SQL更复杂，建议降低并发）')
    parser.add_argument('--max_retries', type=int, default=3,
                       help='最大重试次数')
    parser.add_argument('--only_3db', action='store_true',
                       help='只生成3个数据库的SQL')
    parser.add_argument('--only_4db', action='store_true',
                       help='只生成4个数据库的SQL')
    parser.add_argument('--num_3db', type=int, default=None,
                       help='3个数据库的SQL生成数量（默认根据目标自动计算）')
    parser.add_argument('--num_4db', type=int, default=None,
                       help='4个数据库的SQL生成数量（默认根据目标自动计算）')
    
    args = parser.parse_args()
    
    # 目标数量
    TARGET_COUNTS = {
        2: 359,  # 跨2个数据库（已完成，不生成）
        3: 105,  # 跨3个数据库
        4: 2     # 跨4个数据库
    }
    
    print("=" * 70)
    print("生成3个和4个数据库的JOIN SQL")
    print("=" * 70)
    print(f"\n骨架文件: {args.skeleton_file}")
    print(f"图文件目录: {args.graph_dir}")
    print(f"数据库目录: {args.database_dir}")
    print(f"输出目录: {args.output_dir}")
    print(f"最大并发数: {args.max_workers}")
    print()
    
    # 确定要生成的数据库数量
    db_counts_to_generate = []
    if args.only_3db:
        db_counts_to_generate = [3]
    elif args.only_4db:
        db_counts_to_generate = [4]
    else:
        db_counts_to_generate = [3, 4]
    
    print(f"将生成以下数据库数量的SQL: {db_counts_to_generate}")
    print()
    
    # 统计当前已有的数量
    import json
    current_counts = {3: 0, 4: 0}
    
    if os.path.exists(args.sql_dir):
        for f in os.listdir(args.sql_dir):
            if f.startswith('cross_db_generated_sql_') and f.endswith('.json'):
                try:
                    file_path = os.path.join(args.sql_dir, f)
                    with open(file_path, 'r', encoding='utf-8') as file:
                        data = json.load(file)
                        results = data.get('results', [])
                        num_databases = data.get('metadata', {}).get('num_databases', 
                                                                    len(data.get('databases', [])))
                        
                        if results is not None and len(results) > 0 and num_databases in [3, 4]:
                            current_counts[num_databases] += 1
                except:
                    pass
    
    print("当前已有数量:")
    for db_count in [3, 4]:
        target = TARGET_COUNTS[db_count]
        current = current_counts[db_count]
        needed = max(0, target - current)
        print(f"  {db_count}个数据库: {current} / {target} (还需要 {needed})")
    
    print()
    
    # 确定要生成的数量
    needed = {}
    for db_count in db_counts_to_generate:
        target = TARGET_COUNTS[db_count]
        current = current_counts[db_count]
        
        if db_count == 3 and args.num_3db is not None:
            needed[db_count] = args.num_3db
        elif db_count == 4 and args.num_4db is not None:
            needed[db_count] = args.num_4db
        else:
            needed[db_count] = max(0, target - current)
    
    # 如果所有目标都已达到
    total_needed = sum(needed.values())
    if total_needed == 0:
        print("✅ 所有目标数量都已达到，无需生成更多SQL")
        return
    
    print(f"将生成: {needed}")
    print()
    
    # 加载骨架文件
    print("加载SQL骨架...")
    with open(args.skeleton_file, 'r', encoding='utf-8') as file:
        all_skeletons = json.load(file)
    
    # 过滤出3个和4个数据库的骨架
    skeletons_3db = [s for s in all_skeletons if s.get('num_databases') == 3]
    skeletons_4db = [s for s in all_skeletons if s.get('num_databases') == 4]
    
    print(f"  总骨架数: {len(all_skeletons)}")
    print(f"  3个数据库的骨架: {len(skeletons_3db)} 个")
    print(f"  4个数据库的骨架: {len(skeletons_4db)} 个")
    
    # 检查骨架是否足够
    if 3 in needed and len(skeletons_3db) < needed[3]:
        print(f"  ⚠️  警告: 3个数据库的骨架数量({len(skeletons_3db)})少于需要数量({needed[3]})")
    if 4 in needed and len(skeletons_4db) < needed[4]:
        print(f"  ⚠️  警告: 4个数据库的骨架数量({len(skeletons_4db)})少于需要数量({needed[4]})")
    
    # 收集所有涉及的数据库
    all_databases = set()
    for skeleton in all_skeletons:
        if skeleton.get('num_databases') in db_counts_to_generate:
            all_databases.update(skeleton.get('databases', []))
    
    print(f"\n涉及的数据库: {sorted(all_databases)}")
    
    # 加载所有数据库的schema
    print("\n加载数据库schema...")
    schemas = load_multiple_schemas(list(all_databases), args.database_dir)
    print(f"成功加载 {len(schemas)} 个数据库的schema")
    
    if len(schemas) == 0:
        print("⚠️  警告: 没有加载到任何schema，请检查数据库目录路径")
        return
    
    # 调用生成函数（需要修改generate_more_sqls来支持自定义needed）
    # 由于generate_more_sqls内部会重新计算needed，我们需要临时修改TARGET_COUNTS
    # 或者直接调用内部逻辑
    
    # 由于generate_more_sqls会重新计算needed，我们需要直接调用内部逻辑
    # 这里我们直接使用gen_module中的函数
    
    get_existing_skeletons = gen_module.get_existing_skeletons
    filter_skeletons_by_db_count = gen_module.filter_skeletons_by_db_count
    process_cross_database_skeleton = gen_module.process_cross_database_skeleton
    from concurrent.futures import ThreadPoolExecutor, as_completed
    from tqdm import tqdm
    import time
    import random
    
    # 获取已使用的骨架标识
    existing_skeleton_signatures = get_existing_skeletons(args.sql_dir)
    print(f"  已使用的骨架数: {len(existing_skeleton_signatures)}")
    
    # 按数据库数量分类骨架
    skeletons_by_db_count = {}
    for db_count in db_counts_to_generate:
        skeletons_by_db_count[db_count] = filter_skeletons_by_db_count(all_skeletons, db_count)
    
    # 为每个数据库数量类别选择未使用的骨架
    selected_skeletons = []
    
    for db_count in sorted(needed.keys()):
        if needed[db_count] == 0:
            continue
        
        available_skeletons = skeletons_by_db_count[db_count]
        
        # 过滤出未使用的骨架
        unused_skeletons = []
        for skeleton in available_skeletons:
            databases = sorted(skeleton.get('databases', []))
            table_db_mapping = skeleton.get('table_database_mapping', {})
            tables = sorted(table_db_mapping.keys())
            
            if len(databases) >= 2 and len(tables) >= 2:
                signature = tuple(sorted(databases[:2]) + sorted([tables[0], tables[1] if len(tables) > 1 else tables[0]]))
                if signature not in existing_skeleton_signatures:
                    unused_skeletons.append(skeleton)
        
        print(f"\n  {db_count}个数据库:")
        print(f"    可用骨架: {len(available_skeletons)}")
        print(f"    未使用骨架: {len(unused_skeletons)}")
        print(f"    需要生成: {needed[db_count]}")
        
        if len(unused_skeletons) == 0:
            print(f"    ⚠️  没有可用的未使用骨架，无法生成更多SQL")
            continue
        
        # 选择骨架（优先选择未使用的，如果不够则随机选择）
        num_to_select = min(needed[db_count] * 3, len(unused_skeletons))  # 假设成功率约33%，生成3倍数量
        selected = random.sample(unused_skeletons, num_to_select)
        selected_skeletons.extend(selected)
        print(f"    选择了 {len(selected)} 个骨架")
    
    if len(selected_skeletons) == 0:
        print("\n⚠️  没有可用的骨架，无法生成SQL")
        return
    
    # 开始生成
    print(f"\n{'='*70}")
    print(f"开始生成SQL（共 {len(selected_skeletons)} 个）...")
    print(f"并发数: {args.max_workers}, 最大重试: {args.max_retries}")
    print(f"{'='*70}\n")
    
    success_count = 0
    failed_count = 0
    results_with_data = 0
    
    start_time = time.time()
    last_print_time = start_time
    
    with ThreadPoolExecutor(max_workers=args.max_workers) as executor:
        futures = []
        for i, skeleton in enumerate(selected_skeletons):
            future = executor.submit(
                process_cross_database_skeleton,
                skeleton, schemas, args.graph_dir, args.output_dir,
                args.database_dir, args.max_retries
            )
            futures.append((future, i+1))
        
        # 收集结果
        completed = 0
        pbar = tqdm(total=len(futures), desc="生成进度", ncols=120, unit="个", 
                   bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]')
        
        for future, idx_num in futures:
            try:
                idx, success, message = future.result(timeout=600)  # 10分钟超时
                completed += 1
                pbar.update(1)
                
                if success:
                    success_count += 1
                    # 检查是否有结果数据
                    try:
                        output_file = os.path.join(args.output_dir, f"cross_db_generated_sql_{idx}.json")
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
                
                # 每5秒打印一次详细统计
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

if __name__ == '__main__':
    main()

