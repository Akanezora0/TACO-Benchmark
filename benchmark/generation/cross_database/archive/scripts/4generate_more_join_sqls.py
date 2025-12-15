#!/usr/bin/env python3
"""
生成更多JOIN版本的SQL（不重复已有的）

基于已有结果，只生成新的SQL，避免重复
"""

import os
import json
import argparse
from collections import defaultdict

# 目标数量
TARGET_COUNTS = {
    2: 359,  # 跨2个数据库
    3: 105,  # 跨3个数据库
    4: 2     # 跨4个数据库
}

def get_existing_indices(sql_dir):
    """获取已有SQL文件的索引"""
    existing_indices = set()
    
    for f in os.listdir(sql_dir):
        if f.startswith('cross_db_generated_sql_') and f.endswith('.json'):
            try:
                idx = int(f.replace('cross_db_generated_sql_', '').replace('.json', ''))
                existing_indices.add(idx)
            except:
                pass
    
    return existing_indices

def get_existing_skeletons(sql_dir):
    """获取已使用的骨架标识（基于已有结果的SQL文件）
    
    使用表对信息作为唯一标识，因为original_file可能不存在
    格式: (db1, db2, table1, table2) -> 已使用
    """
    existing_skeleton_signatures = set()
    
    # 从SQL文件中提取表对信息作为唯一标识
    for f in os.listdir(sql_dir):
        if f.startswith('cross_db_generated_sql_') and f.endswith('.json'):
            file_path = os.path.join(sql_dir, f)
            try:
                with open(file_path, 'r', encoding='utf-8') as file:
                    data = json.load(file)
                    results = data.get('results', [])
                    
                    # 只考虑有结果的
                    if results is not None and len(results) > 0:
                        databases = sorted(data.get('databases', []))
                        table_db_mapping = data.get('table_database_mapping', {})
                        
                        # 从table_database_mapping中提取表名（不包含数据库前缀）
                        tables = sorted(table_db_mapping.keys())
                        
                        # 使用数据库和表的组合作为唯一标识
                        if len(databases) >= 2 and len(tables) >= 2:
                            # 确保表与数据库对应
                            table1 = tables[0] if tables[0] in table_db_mapping else None
                            table2 = tables[1] if len(tables) > 1 and tables[1] in table_db_mapping else None
                            
                            if table1 and table2:
                                signature = tuple(sorted(databases[:2]) + sorted([table1, table2]))
                                existing_skeleton_signatures.add(signature)
            except Exception as e:
                pass
    
    return existing_skeleton_signatures

def count_needed_by_db_count(sql_dir):
    """统计每个数据库数量类别还需要多少个"""
    needed = {}
    
    for db_count in [2, 3, 4]:
        target = TARGET_COUNTS[db_count]
        current = 0
        
        # 统计当前有结果的数量
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
                       max_workers=5, max_retries=3):
    """生成更多SQL"""
    
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
    
    # 3. 获取已使用的骨架标识
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
        
        # 过滤出未使用的骨架（使用表对信息作为唯一标识）
        unused_skeletons = []
        for skeleton in available_skeletons:
            databases = sorted(skeleton.get('databases', []))
            table_db_mapping = skeleton.get('table_database_mapping', {})
            tables = sorted(table_db_mapping.keys())
            
            # 使用数据库和表的组合作为唯一标识
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
        
        # 选择需要的数量（如果未使用的骨架不够，就全部使用）
        import random
        to_generate = min(needed[db_count] * 3, len(unused_skeletons))  # 生成3倍，因为成功率约22.5%
        selected = random.sample(unused_skeletons, to_generate) if len(unused_skeletons) >= to_generate else unused_skeletons
        
        print(f"  选择生成: {len(selected)} 个骨架")
        selected_skeletons.extend(selected)
    
    if not selected_skeletons:
        print("\n⚠️  没有可用的未使用骨架，无法生成更多SQL")
        return
    
    print(f"\n总计选择: {len(selected_skeletons)} 个骨架用于生成")
    
    # 6. 保存选中的骨架到临时文件
    temp_skeleton_file = skeleton_file.replace('.json', '_temp.json')
    with open(temp_skeleton_file, 'w', encoding='utf-8') as f:
        json.dump(selected_skeletons, f, ensure_ascii=False, indent=2)
    
    print(f"\n临时骨架文件: {temp_skeleton_file}")
    
    # 7. 调用SQL填充脚本（直接调用函数，不使用subprocess，避免阻塞）
    print("\n开始生成SQL...")
    print(f"将生成 {len(selected_skeletons)} 个SQL，这可能需要一些时间...")
    
    # 直接导入并调用函数，而不是使用subprocess
    import sys
    sys.path.insert(0, os.path.dirname(__file__))
    from cross_db_2fill_sql_placeholders_join import (
        load_multiple_schemas, 
        process_cross_database_skeleton
    )
    from concurrent.futures import ThreadPoolExecutor, as_completed
    from tqdm import tqdm
    
    # 加载所有数据库的schema
    all_databases = set()
    for skeleton in selected_skeletons:
        all_databases.update(skeleton.get('databases', []))
    
    schemas = load_multiple_schemas(all_databases, database_dir)
    
    # 处理每个骨架
    success_count = 0
    failed_count = 0
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = []
        for skeleton in selected_skeletons:
            future = executor.submit(
                process_cross_database_skeleton,
                skeleton, schemas, graph_dir, output_dir,
                database_dir, max_retries
            )
            futures.append(future)
        
        # 收集结果
        for future in tqdm(as_completed(futures), total=len(futures), desc="填充进度"):
            try:
                idx, success, message = future.result()
                if success:
                    success_count += 1
                else:
                    failed_count += 1
            except Exception as e:
                failed_count += 1
                print(f"处理失败: {e}")
    
    print(f"\n生成完成!")
    print(f"成功: {success_count}/{len(selected_skeletons)}")
    print(f"失败: {failed_count}/{len(selected_skeletons)}")
    
    # 8. 清理临时文件
    if os.path.exists(temp_skeleton_file):
        os.remove(temp_skeleton_file)
        print(f"\n已清理临时文件: {temp_skeleton_file}")

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
    
    args = parser.parse_args()
    
    generate_more_sqls(
        args.skeleton_file,
        args.sql_dir,
        args.graph_dir,
        args.output_dir,
        args.database_dir,
        args.max_workers,
        args.max_retries
    )

if __name__ == '__main__':
    main()

