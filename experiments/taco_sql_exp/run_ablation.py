"""
TACO-SQL消融实验主脚本

统一入口，支持运行所有实验设置
"""

import sys
import os
from pathlib import Path

# 添加父目录到路径
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from experiments.taco_sql_exp.experiment_runner import run_experiment


def main():
    """运行消融实验"""
    import argparse
    
    parser = argparse.ArgumentParser(description="运行TACO-SQL消融实验")
    parser.add_argument(
        "--setting", 
        type=str, 
        choices=["origin", "qr", "qr_tl", "qr_tl_qp"],
        required=True,
        help="实验设置"
    )
    parser.add_argument("--model", type=str, default="gpt-4o", help="模型名称")
    parser.add_argument("--dataset", type=str, default="taco_beijing", help="数据集名称")
    parser.add_argument("--test_data", type=str, help="测试数据路径")
    parser.add_argument("--output", type=str, help="输出路径")
    
    # 组件参数
    parser.add_argument("--qr_temperature", type=float, default=0.3, help="Question Rewriting温度")
    parser.add_argument("--tl_top_k", type=int, default=5, help="Table Linking Top-K")
    parser.add_argument("--qp_temperature", type=float, default=0.3, help="Query Planning温度")
    
    args = parser.parse_args()
    
    # 设置默认输出路径
    if not args.output:
        args.output = f"experiments/results/{args.setting}_{args.model}_{args.dataset}.json"
    
    # 打印实验信息
    setting_names = {
        "origin": "Origin（原始查询 + 完整Schema）",
        "qr": "QR（+ Question Rewriting）",
        "qr_tl": "QR+TL（+ Question Rewriting + Table Linking）",
        "qr_tl_qp": "完整TACO-SQL（+ Question Rewriting + Table Linking + Query Planning）"
    }
    
    print(f"运行TACO-SQL消融实验")
    print(f"  实验设置: {setting_names[args.setting]}")
    print(f"  模型: {args.model}")
    print(f"  数据集: {args.dataset}")
    if args.setting in ["qr", "qr_tl", "qr_tl_qp"]:
        print(f"  QR温度: {args.qr_temperature}")
    if args.setting in ["qr_tl", "qr_tl_qp"]:
        print(f"  TL Top-K: {args.tl_top_k}")
    if args.setting == "qr_tl_qp":
        print(f"  QP温度: {args.qp_temperature}")
    print(f"  输出: {args.output}")
    print()
    
    # 构建kwargs
    kwargs = {}
    if args.setting in ["qr", "qr_tl", "qr_tl_qp"]:
        kwargs["qr_temperature"] = args.qr_temperature
    if args.setting in ["qr_tl", "qr_tl_qp"]:
        kwargs["tl_top_k"] = args.tl_top_k
    if args.setting == "qr_tl_qp":
        kwargs["qp_temperature"] = args.qp_temperature
    
    # 运行实验
    results = run_experiment(
        setting=args.setting,
        model_name=args.model,
        dataset_name=args.dataset,
        test_data_path=args.test_data,
        output_path=args.output,
        **kwargs
    )
    
    print(f"\n实验完成！")
    print(f"  处理查询数: {len(results)}")
    print(f"  结果已保存至: {args.output}")


if __name__ == "__main__":
    main()

