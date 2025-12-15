"""
补齐SQL生成到200条
检查每个数据库已生成的SQL数量，如果不足200条，重新生成缺失的SQL
"""

import json
import os
import sys
import argparse
import importlib.util
from tqdm import tqdm

# 动态导入2fill_sql_placeholders_improved模块
script_dir = os.path.dirname(os.path.abspath(__file__))
module_path = os.path.join(script_dir, '2fill_sql_placeholders_improved.py')
spec = importlib.util.spec_from_file_location("fill_sql", module_path)
fill_sql_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(fill_sql_module)

# 导入需要的函数
process_single_sql_skeleton = fill_sql_module.process_single_sql_skeleton
load_schema = fill_sql_module.load_schema
extract_schema_info = fill_sql_module.extract_schema_info
load_config = fill_sql_module.load_config

def check_sql_count(database_name, skeleton_file, sql_dir):
    """检查已生成的SQL数量"""
    # 加载SQL骨架
    with open(skeleton_file, 'r', encoding='utf-8') as f:
        sql_skeletons = json.load(f)
    
    total_skeletons = len(sql_skeletons)
    
    # 检查已生成的SQL文件
    existing_indices = set()
    if os.path.exists(sql_dir):
        for f in os.listdir(sql_dir):
            if f.startswith('generated_sql_') and f.endswith('.json'):
                try:
                    idx = int(f.replace('generated_sql_', '').replace('.json', ''))
                    existing_indices.add(idx)
                except:
                    pass
    
    missing_indices = [i for i in range(total_skeletons) if i not in existing_indices]
    
    return total_skeletons, len(existing_indices), missing_indices

def complete_sql_for_database(database_name, skeleton_file, schema_file, graph_dir, output_dir, max_retries=3, max_workers=5):
    """为单个数据库补齐SQL到200条"""
    # 加载schema
    schema = load_schema(schema_file)
    schema_info = extract_schema_info(schema)
    
    # 加载SQL skeletons
    with open(skeleton_file, 'r', encoding='utf-8') as f:
        sql_skeletons = json.load(f)
    
    # 创建输出目录
    single_output_path = os.path.join(output_dir, 'single', database_name)
    os.makedirs(single_output_path, exist_ok=True)
    
    # 检查缺失的SQL
    total_skeletons, existing_count, missing_indices = check_sql_count(
        database_name, skeleton_file, single_output_path
    )
    
    print(f"\n数据库: {database_name}")
    print(f"  总骨架数: {total_skeletons}")
    print(f"  已生成: {existing_count}")
    print(f"  缺失: {len(missing_indices)}")
    
    if len(missing_indices) == 0:
        print(f"  ✅ 已完成，无需补齐")
        return existing_count, 0
    
    # 只处理缺失的索引
    print(f"  开始补齐 {len(missing_indices)} 条SQL...")
    
    from concurrent.futures import ThreadPoolExecutor, as_completed
    
    # 准备任务参数（只处理缺失的索引）
    tasks = []
    for idx in missing_indices:
        sql_skeleton = sql_skeletons[idx]
        tasks.append((
            idx, sql_skeleton, database_name, schema, schema_info, 
            graph_dir, single_output_path, schema_file, max_retries
        ))
    
    success_count = 0
    fail_count = 0
    
    # 使用线程池并发处理
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_idx = {executor.submit(process_single_sql_skeleton, task): task[0] for task in tasks}
        
        with tqdm(total=len(tasks), desc=f"  补齐进度") as pbar:
            for future in as_completed(future_to_idx):
                idx = future_to_idx[future]
                try:
                    result_idx, success, message = future.result()
                    if success:
                        if message != "已存在":
                            success_count += 1
                    else:
                        fail_count += 1
                except Exception as e:
                    fail_count += 1
                    print(f"  处理索引 {idx} 时发生异常: {e}")
                finally:
                    pbar.update(1)
    
    # 再次检查最终数量
    _, final_count, _ = check_sql_count(database_name, skeleton_file, single_output_path)
    print(f"  ✅ 补齐完成：新增 {success_count} 条，失败 {fail_count} 条，最终数量: {final_count}/{total_skeletons}")
    
    return final_count, fail_count

def main():
    parser = argparse.ArgumentParser(description='补齐SQL生成到200条')
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(script_dir)))
    
    parser.add_argument('--skeleton_dir', type=str, default=None,
                       help='SQL骨架目录')
    parser.add_argument('--database_dir', type=str, default=None,
                       help='数据库目录')
    parser.add_argument('--graph_dir', type=str, default=None,
                       help='图目录')
    parser.add_argument('--output_dir', type=str, default=None,
                       help='输出目录')
    parser.add_argument('--max_retries', type=int, default=3,
                       help='最大重试次数（默认3）')
    parser.add_argument('--max_workers', type=int, default=5,
                       help='并发数（默认5）')
    parser.add_argument('--config', type=str, default=None,
                       help='配置文件路径')
    parser.add_argument('--database', type=str, default=None,
                       help='只处理指定数据库（默认处理所有）')
    
    args = parser.parse_args()
    
    # 加载配置
    load_config(args.config)
    
    # 设置默认路径（US数据集）
    if args.skeleton_dir is None:
        args.skeleton_dir = os.path.join(project_root, 'benchmark', 'data', 'us', 'output', 'sql_skeleton')
    if args.database_dir is None:
        args.database_dir = os.path.join(project_root, 'benchmark', 'data', 'us', 'database')
    if args.graph_dir is None:
        args.graph_dir = os.path.join(project_root, 'benchmark', 'data', 'us', 'output', 'graph')
    if args.output_dir is None:
        args.output_dir = os.path.join(project_root, 'benchmark', 'data', 'us', 'output')
    
    # 转换为绝对路径
    args.skeleton_dir = os.path.abspath(args.skeleton_dir)
    args.database_dir = os.path.abspath(args.database_dir)
    args.graph_dir = os.path.abspath(args.graph_dir)
    args.output_dir = os.path.abspath(args.output_dir)
    
    # 获取所有SQL skeleton文件
    skeleton_files = [f for f in os.listdir(args.skeleton_dir) if f.endswith('_sql_skeleton.json')]
    
    if args.database:
        # 只处理指定数据库
        skeleton_files = [f for f in skeleton_files if args.database in f]
    
    print(f"找到 {len(skeleton_files)} 个数据库的SQL骨架文件")
    print(f"骨架目录: {args.skeleton_dir}")
    print(f"数据库目录: {args.database_dir}")
    print(f"图目录: {args.graph_dir}")
    print(f"输出目录: {args.output_dir}")
    print(f"并发数: {args.max_workers}, 最大重试次数: {args.max_retries}")
    print("="*60)
    
    total_success = 0
    total_fail = 0
    databases_status = []
    
    for skeleton_file in skeleton_files:
        database_name = skeleton_file.replace('_sql_skeleton.json', '')
        
        skeleton_path = os.path.join(args.skeleton_dir, skeleton_file)
        schema_path = os.path.join(args.database_dir, database_name, f"{database_name}.json")
        
        if not os.path.exists(schema_path):
            print(f"\n⚠️  数据库 '{database_name}' 的schema文件不存在: {schema_path}")
            continue
        
        final_count, fail_count = complete_sql_for_database(
            database_name, skeleton_path, schema_path,
            args.graph_dir, args.output_dir, args.max_retries, args.max_workers
        )
        
        databases_status.append({
            'database': database_name,
            'final_count': final_count,
            'fail_count': fail_count
        })
        
        total_success += final_count
        total_fail += fail_count
    
    print("\n" + "="*60)
    print("📊 汇总统计：")
    print("="*60)
    
    # 按最终数量排序
    databases_status.sort(key=lambda x: x['final_count'])
    
    for status in databases_status:
        db = status['database']
        count = status['final_count']
        fail = status['fail_count']
        if count >= 200:
            print(f"✅ {db}: {count}/200 (失败: {fail})")
        elif count >= 150:
            print(f"⚠️  {db}: {count}/200 (失败: {fail})")
        else:
            print(f"❌ {db}: {count}/200 (失败: {fail})")
    
    print("="*60)
    print(f"总计：成功 {total_success}，失败 {total_fail}")
    print("="*60)

if __name__ == '__main__':
    main()

