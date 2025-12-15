"""
QR实验设置实现

+ Question Rewriting：仅使用问题转写组件
"""

import sys
import os
from pathlib import Path

# 添加父目录到路径
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from experiments.taco_sql_exp.experiment_runner import run_experiment


def main():
    """运行QR实验"""
    import argparse
    
    parser = argparse.ArgumentParser(description="运行QR实验设置（+ Question Rewriting）")
    parser.add_argument("--model", type=str, default="gpt-4o", help="模型名称")
    parser.add_argument("--dataset", type=str, default="taco_beijing", help="数据集名称")
    parser.add_argument("--test_data", type=str, help="测试数据路径")
    parser.add_argument("--output", type=str, help="输出路径")
    parser.add_argument("--qr_temperature", type=float, default=0.3, help="Question Rewriting温度")
    
    args = parser.parse_args()
    
    # 设置默认输出路径
    if not args.output:
        args.output = f"experiments/results/qr_{args.model}_{args.dataset}.json"
    
    print(f"运行QR实验设置（+ Question Rewriting）")
    print(f"  模型: {args.model}")
    print(f"  数据集: {args.dataset}")
    print(f"  QR温度: {args.qr_temperature}")
    print(f"  输出: {args.output}")
    print()
    
    # 运行实验
    results = run_experiment(
        setting="qr",
        model_name=args.model,
        dataset_name=args.dataset,
        test_data_path=args.test_data,
        output_path=args.output,
        qr_temperature=args.qr_temperature
    )
    
    print(f"\n实验完成！")
    print(f"  处理查询数: {len(results)}")
    print(f"  结果已保存至: {args.output}")


if __name__ == "__main__":
    main()

