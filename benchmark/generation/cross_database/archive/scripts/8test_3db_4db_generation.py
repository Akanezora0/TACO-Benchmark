#!/usr/bin/env python3
"""
测试3个和4个数据库的SQL生成
先生成少量测试，验证流程是否正常
"""

import os
import json
import argparse
from tqdm import tqdm
import sys

# 导入SQL填充模块
sys.path.insert(0, os.path.dirname(__file__))
from cross_db_2fill_sql_placeholders_join import (
    process_cross_database_skeleton
)

# 导入load_schema函数
import importlib.util
sql_filling_dir = os.path.join(os.path.dirname(__file__), '..', 'sql_filling')
spec = importlib.util.spec_from_file_location(
    "fill_sql_placeholders_improved",
    os.path.join(sql_filling_dir, "2fill_sql_placeholders_improved.py")
)
fill_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(fill_module)
load_schema = fill_module.load_schema

def test_3db_4db_generation(skeleton_file, graph_dir, database_dir, output_dir, 
                            num_3db=5, num_4db=3):
    """测试生成3个和4个数据库的SQL"""
    
    # 加载骨架
    print("加载SQL骨架...")
    with open(skeleton_file, 'r', encoding='utf-8') as f:
        all_skeletons = json.load(f)
    
    # 筛选3个和4个数据库的骨架
    skeletons_3db = [s for s in all_skeletons if s.get('num_databases') == 3]
    skeletons_4db = [s for s in all_skeletons if s.get('num_databases') == 4]
    
    print(f"  找到 {len(skeletons_3db)} 个3数据库骨架")
    print(f"  找到 {len(skeletons_4db)} 个4数据库骨架")
    
    # 选择要测试的骨架
    test_skeletons = skeletons_3db[:num_3db] + skeletons_4db[:num_4db]
    print(f"\n将测试 {len(test_skeletons)} 个骨架（{num_3db}个3数据库 + {num_4db}个4数据库）")
    
    # 收集所有涉及的数据库
    all_databases = set()
    for skeleton in test_skeletons:
        all_databases.update(skeleton.get('databases', []))
    
    print(f"涉及的数据库: {sorted(all_databases)}")
    
    # 加载所有数据库的schema
    print("\n加载数据库schema...")
    # load_multiple_schemas期望的路径格式：database_dir/数据库名/数据库名.json
    schemas = {}
    for db_name in all_databases:
        schema_file = os.path.join(database_dir, db_name, f"{db_name}.json")
        if os.path.exists(schema_file):
            try:
                schema = load_schema(schema_file)
                if schema:
                    schemas[db_name] = schema
            except Exception as e:
                print(f"  警告: 加载schema失败 {schema_file}: {e}")
        else:
            print(f"  警告: 找不到schema文件 {schema_file}")
    print(f"成功加载 {len(schemas)} 个数据库的schema")
    
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    # 处理每个骨架
    print("\n开始生成SQL...")
    success_count = 0
    has_results_count = 0
    failed_count = 0
    
    for skeleton in tqdm(test_skeletons, desc="生成SQL"):
        # 找到对应的图文件
        original_file = skeleton.get('original_file', '')
        match = __import__('re').search(r'(\d+)', original_file)
        if match:
            graph_idx = match.group(1)
            graph_file = os.path.join(graph_dir, f"cross_db_graph_{graph_idx}.json")
        else:
            graph_file = None
        
        if not graph_file or not os.path.exists(graph_file):
            print(f"  警告: 找不到图文件 {graph_file}")
            failed_count += 1
            continue
        
        # 加载图文件
        with open(graph_file, 'r', encoding='utf-8') as f:
            graph_data = json.load(f)
        
        # 处理骨架
        try:
            # process_cross_database_skeleton需要graph_dir而不是graph_data
            # 我们需要找到graph_dir
            graph_dir = os.path.dirname(graph_file)
            
            # 调用处理函数
            # process_cross_database_skeleton返回：(idx, success, message)
            idx, success, message = process_cross_database_skeleton(
                skeleton,
                schemas,
                graph_dir,
                output_dir,
                database_dir
            )
            
            # 根据返回的idx构建输出文件路径
            result_file = os.path.join(output_dir, f"cross_db_generated_sql_{idx}.json")
            
            # 检查结果
            if os.path.exists(result_file):
                with open(result_file, 'r', encoding='utf-8') as f:
                    result_data = json.load(f)
                
                results = result_data.get('results', [])
                if results is not None and len(results) > 0:
                    has_results_count += 1
                    success_count += 1
                else:
                    failed_count += 1
            else:
                failed_count += 1
        except Exception as e:
            print(f"  错误: {e}")
            failed_count += 1
    
    # 统计结果
    print("\n" + "=" * 70)
    print("测试结果统计")
    print("=" * 70)
    print(f"总测试数: {len(test_skeletons)}")
    print(f"成功生成: {success_count} ({success_count/len(test_skeletons)*100:.1f}%)")
    print(f"有结果: {has_results_count} ({has_results_count/len(test_skeletons)*100:.1f}%)")
    print(f"失败: {failed_count} ({failed_count/len(test_skeletons)*100:.1f}%)")
    print("=" * 70)

def main():
    parser = argparse.ArgumentParser(description='测试3个和4个数据库的SQL生成')
    parser.add_argument('--skeleton_file', type=str,
                       default='benchmark/generation/cross_database/cross_db_skeletons_join.json',
                       help='SQL骨架文件')
    parser.add_argument('--graph_dir', type=str,
                       default='benchmark/generation/cross_database/cross_db_graphs_join',
                       help='图文件目录')
    parser.add_argument('--database_dir', type=str,
                       default='benchmark/data/beijing/database_chinese',
                       help='数据库目录')
    parser.add_argument('--output_dir', type=str,
                       default='benchmark/data/beijing/output/cross_db_single_join',
                       help='输出目录')
    parser.add_argument('--num_3db', type=int, default=5,
                       help='测试的3数据库骨架数量')
    parser.add_argument('--num_4db', type=int, default=3,
                       help='测试的4数据库骨架数量')
    
    args = parser.parse_args()
    
    test_3db_4db_generation(
        args.skeleton_file,
        args.graph_dir,
        args.database_dir,
        args.output_dir,
        args.num_3db,
        args.num_4db
    )

if __name__ == '__main__':
    main()

