"""
Origin实验设置实现

原始设置：原始查询 + 完整Schema
不使用任何TACO-SQL组件
"""

import sys
import os
from pathlib import Path

# 添加父目录到路径
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from experiments.taco_sql_exp.experiment_runner import run_experiment


def main():
    """运行Origin实验"""
    import argparse
    
    parser = argparse.ArgumentParser(description="运行Origin实验设置")
    parser.add_argument("--model", type=str, default="gpt-4o", help="模型名称")
    parser.add_argument("--dataset", type=str, default="taco_beijing", help="数据集名称")
    parser.add_argument("--test_data", type=str, help="测试数据路径")
    parser.add_argument("--output", type=str, help="输出路径")
    
    args = parser.parse_args()
    
    # 设置默认输出路径
    if not args.output:
        args.output = f"experiments/results/origin_{args.model}_{args.dataset}.json"
    
    print(f"运行Origin实验设置")
    print(f"  模型: {args.model}")
    print(f"  数据集: {args.dataset}")
    print(f"  输出: {args.output}")
    print()
    
    # 运行实验
    results = run_experiment(
        setting="origin",
        model_name=args.model,
        dataset_name=args.dataset,
        test_data_path=args.test_data,
        output_path=args.output
    )
    
    print(f"\n实验完成！")
    print(f"  处理查询数: {len(results)}")
    print(f"  结果已保存至: {args.output}")


if __name__ == "__main__":
    main()

