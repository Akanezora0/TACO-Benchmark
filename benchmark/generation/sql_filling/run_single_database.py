#!/usr/bin/env python3
"""
只处理单个数据库的SQL填充脚本
用于完成特定数据库的所有SQL骨架生成
"""

import sys
import os

# 添加当前目录到路径
script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, script_dir)

import importlib.util
import argparse

# 动态导入模块（因为模块名以数字开头）
spec = importlib.util.spec_from_file_location(
    "fill_module", 
    os.path.join(script_dir, "2fill_sql_placeholders_improved.py")
)
fill_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(fill_module)

process_single_database = fill_module.process_single_database
load_config = fill_module.load_config

def main():
    parser = argparse.ArgumentParser(description='处理单个数据库的SQL填充')
    parser.add_argument('--database_name', type=str, required=True,
                       help='数据库名称（例如：医疗健康）')
    parser.add_argument('--database_dir', type=str, 
                       default='../../data/beijing/database_chinese',
                       help='数据库目录')
    parser.add_argument('--skeleton_dir', type=str,
                       default='../../data/beijing/output/sql_skeleton',
                       help='SQL骨架目录')
    parser.add_argument('--graph_dir', type=str,
                       default='../../data/beijing/output/graph_chinese',
                       help='图文件目录')
    parser.add_argument('--output_dir', type=str,
                       default='../../data/beijing/output',
                       help='输出目录')
    parser.add_argument('--max_retries', type=int, default=3,
                       help='最大重试次数（默认3）')
    parser.add_argument('--config', type=str, default=None,
                       help='配置文件路径（默认：./config.yaml）')
    
    args = parser.parse_args()
    
    # 转换为绝对路径
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(script_dir)))
    
    database_dir = os.path.abspath(os.path.join(script_dir, args.database_dir))
    skeleton_dir = os.path.abspath(os.path.join(script_dir, args.skeleton_dir))
    graph_dir = os.path.abspath(os.path.join(script_dir, args.graph_dir))
    output_dir = os.path.abspath(os.path.join(script_dir, args.output_dir))
    
    # 构建文件路径
    skeleton_file = os.path.join(skeleton_dir, f"{args.database_name}_sql_skeleton.json")
    schema_file = os.path.join(database_dir, args.database_name, f"{args.database_name}.json")
    
    # 检查文件是否存在
    if not os.path.exists(skeleton_file):
        print(f"错误：SQL骨架文件不存在: {skeleton_file}")
        return
    
    if not os.path.exists(schema_file):
        print(f"错误：Schema文件不存在: {schema_file}")
        return
    
    # 加载配置
    config_file = args.config if args.config else os.path.join(script_dir, 'config.yaml')
    fill_module.API_CONFIG = fill_module.load_config(config_file)
    
    print(f"=== 开始处理数据库：{args.database_name} ===")
    print(f"SQL骨架文件: {skeleton_file}")
    print(f"Schema文件: {schema_file}")
    print(f"图文件目录: {graph_dir}")
    print(f"输出目录: {output_dir}")
    print(f"最大重试次数: {args.max_retries}")
    print()
    
    # 处理数据库
    success_count, fail_count = process_single_database(
        args.database_name,
        skeleton_file,
        schema_file,
        graph_dir,
        output_dir,
        max_retries=args.max_retries
    )
    
    print()
    print(f"=== 处理完成 ===")
    print(f"成功: {success_count}")
    print(f"失败: {fail_count}")
    print(f"总计: {success_count + fail_count}")

if __name__ == '__main__':
    main()

