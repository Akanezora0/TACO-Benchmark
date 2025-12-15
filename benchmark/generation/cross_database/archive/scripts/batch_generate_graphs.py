#!/usr/bin/env python3
"""
批量为所有跨数据库SQL骨架生成图文件
"""

import os
import json
import sys
import importlib.util
from pathlib import Path
from tqdm import tqdm

# 动态导入跨数据库图生成模块
sys.path.insert(0, 'benchmark/generation/cross_database')

spec = importlib.util.spec_from_file_location(
    "cross_db_graphs",
    "benchmark/generation/cross_database/cross_db_1build_schema_graphs.py"
)
graph_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(graph_module)

load_multiple_schemas = graph_module.load_multiple_schemas
process_cross_database_skeleton = graph_module.process_cross_database_skeleton

def process_all_skeleton_files(skeleton_dir, database_dir, output_dir):
    """处理所有骨架文件，生成图文件"""
    
    # 获取所有骨架文件
    skeleton_files = []
    for f in os.listdir(skeleton_dir):
        if f.endswith('_skeletons.json'):
            skeleton_files.append(f)
    
    skeleton_files.sort()
    
    print("=" * 70)
    print("批量生成跨数据库图文件")
    print("=" * 70)
    print(f"\n找到 {len(skeleton_files)} 个骨架文件")
    
    total_skeletons = 0
    total_graphs = 0
    
    # 按组合处理（相同组合的骨架文件一起处理，共享schema）
    combo_schemas_cache = {}  # 缓存已加载的schema
    
    for skeleton_file in tqdm(skeleton_files, desc="处理骨架文件"):
        skeleton_path = os.path.join(skeleton_dir, skeleton_file)
        
        try:
            with open(skeleton_path, 'r', encoding='utf-8') as f:
                skeletons = json.load(f)
            
            if not skeletons:
                continue
            
            # 获取涉及的数据库（从第一个骨架获取，因为同一文件中的骨架涉及相同的数据库）
            first_skeleton = skeletons[0]
            databases = first_skeleton.get('databases', [])
            combo_key = tuple(sorted(databases))
            
            # 加载schema（如果还没加载过）
            if combo_key not in combo_schemas_cache:
                schemas = load_multiple_schemas(databases, database_dir)
                combo_schemas_cache[combo_key] = schemas
                print(f"\n加载数据库组合 {combo_key} 的schema: {len(schemas)} 个数据库")
            else:
                schemas = combo_schemas_cache[combo_key]
            
            # 为每个骨架生成图
            for skeleton in skeletons:
                try:
                    # 生成图（process_cross_database_skeleton会自动处理文件名）
                    output_file = process_cross_database_skeleton(skeleton, schemas, output_dir)
                    if output_file:
                        total_graphs += 1
                        total_skeletons += 1
                    
                except Exception as e:
                    print(f"\n处理骨架失败 {skeleton_file}: {e}")
                    import traceback
                    traceback.print_exc()
                    continue
                    
        except Exception as e:
            print(f"\n读取骨架文件失败 {skeleton_file}: {e}")
            continue
    
    print(f"\n" + "=" * 70)
    print(f"完成！")
    print(f"处理了 {len(skeleton_files)} 个骨架文件")
    print(f"生成了 {total_graphs} 个图文件")
    print(f"输出目录: {output_dir}")
    print("=" * 70)

if __name__ == '__main__':
    import argparse
    import re
    
    parser = argparse.ArgumentParser(description='批量生成跨数据库图文件')
    parser.add_argument('--skeleton_dir', type=str,
                       default='benchmark/generation/cross_database/skeletons',
                       help='骨架文件目录')
    parser.add_argument('--database_dir', type=str,
                       default='benchmark/data/beijing/database_chinese',
                       help='数据库目录')
    parser.add_argument('--output_dir', type=str,
                       default='benchmark/data/beijing/output/cross_db_graph',
                       help='图文件输出目录')
    
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    process_all_skeleton_files(args.skeleton_dir, args.database_dir, args.output_dir)

