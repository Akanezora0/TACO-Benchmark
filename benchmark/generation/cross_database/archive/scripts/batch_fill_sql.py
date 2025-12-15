#!/usr/bin/env python3
"""
批量填充所有跨数据库SQL骨架
"""

import os
import json
import sys
import importlib.util
from pathlib import Path
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed

# 动态导入
sys.path.insert(0, 'benchmark/generation/cross_database')

spec = importlib.util.spec_from_file_location(
    "fill_sql",
    "benchmark/generation/cross_database/cross_db_2fill_sql_placeholders.py"
)
fill_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(fill_module)

load_multiple_schemas = fill_module.load_multiple_schemas
load_cross_database_graph = fill_module.load_cross_database_graph
process_cross_database_skeleton = fill_module.process_cross_database_skeleton

def process_single_skeleton(skeleton, schemas, graph_file, graph_dir, output_dir, database_dir):
    """处理单个骨架"""
    try:
        # 处理骨架（process_cross_database_skeleton会自己加载图文件）
        # 注意：process_cross_database_skeleton返回 (idx, success, message) 或 None
        result = process_cross_database_skeleton(
            skeleton,
            schemas,
            graph_dir,  # graph_dir
            output_dir,
            database_dir
        )
        # result是 (idx, success, message) 或 None
        if result and len(result) >= 2:
            return result[1]  # success标志
        return False
    except Exception as e:
        import traceback
        traceback.print_exc()
        return False

def process_skeleton_file(skeleton_file_path, graph_dir, database_dir, output_dir, max_workers=5):
    """处理单个骨架文件中的所有骨架"""
    import re
    
    with open(skeleton_file_path, 'r', encoding='utf-8') as f:
        skeletons = json.load(f)
    
    if not skeletons:
        return 0, 0
    
    # 获取涉及的数据库
    first_skeleton = skeletons[0]
    databases = first_skeleton.get('databases', [])
    
    # 加载schema
    schemas = load_multiple_schemas(databases, database_dir)
    
    success_count = 0
    fail_count = 0
    
    # 使用线程池并行处理
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = []
        for skeleton in skeletons:
            # 查找对应的图文件
            original_file = skeleton.get('original_file', 'unknown')
            match = re.search(r'(\d+)', original_file)
            if match:
                idx = match.group(1)
                combo_name = '_'.join(sorted(databases))
                graph_file = os.path.join(graph_dir, f"cross_db_graph_{combo_name}_{idx}.json")
            else:
                import hashlib
                hash_id = hashlib.md5(skeleton['sql_skeleton'].encode()).hexdigest()[:8]
                combo_name = '_'.join(sorted(databases))
                graph_file = os.path.join(graph_dir, f"cross_db_graph_{combo_name}_{hash_id}.json")
            
            if not os.path.exists(graph_file):
                fail_count += 1
                continue
            
            future = executor.submit(
                process_single_skeleton,
                skeleton,
                schemas,
                graph_file,
                graph_dir,  # 传递graph_dir而不是graph_file
                output_dir,
                database_dir
            )
            futures.append(future)
        
        # 等待完成
        for future in as_completed(futures):
            try:
                if future.result():
                    success_count += 1
                else:
                    fail_count += 1
            except Exception as e:
                fail_count += 1
    
    return success_count, fail_count

def main():
    import argparse
    import re
    
    parser = argparse.ArgumentParser(description='批量填充跨数据库SQL骨架')
    parser.add_argument('--skeleton_dir', type=str,
                       default='benchmark/generation/cross_database/skeletons',
                       help='骨架文件目录')
    parser.add_argument('--graph_dir', type=str,
                       default='benchmark/data/beijing/output/cross_db_graph',
                       help='图文件目录')
    parser.add_argument('--database_dir', type=str,
                       default='benchmark/data/beijing/database_chinese',
                       help='数据库目录')
    parser.add_argument('--output_dir', type=str,
                       default='benchmark/data/beijing/output/cross_db_single',
                       help='SQL输出目录')
    parser.add_argument('--max_workers', type=int, default=5,
                       help='最大并发数')
    
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 获取所有骨架文件
    skeleton_files = []
    for f in os.listdir(args.skeleton_dir):
        if f.endswith('_skeletons.json'):
            skeleton_files.append(os.path.join(args.skeleton_dir, f))
    
    skeleton_files.sort()
    
    print("=" * 70)
    print("批量填充跨数据库SQL骨架")
    print("=" * 70)
    print(f"\n找到 {len(skeleton_files)} 个骨架文件")
    
    total_success = 0
    total_fail = 0
    
    for skeleton_file in tqdm(skeleton_files, desc="处理骨架文件"):
        success, fail = process_skeleton_file(
            skeleton_file,
            args.graph_dir,
            args.database_dir,
            args.output_dir,
            args.max_workers
        )
        total_success += success
        total_fail += fail
    
    print(f"\n" + "=" * 70)
    print(f"完成！")
    print(f"成功: {total_success}")
    print(f"失败: {total_fail}")
    print(f"输出目录: {args.output_dir}")
    print("=" * 70)

if __name__ == '__main__':
    main()

