#!/usr/bin/env python3
"""
根据数据库组合生成跨数据库SQL骨架
"""

import os
import json
import argparse
import importlib.util
import sys
import tempfile

def load_combinations(combinations_file):
    """加载数据库组合"""
    with open(combinations_file, 'r', encoding='utf-8') as f:
        return json.load(f)

def main():
    parser = argparse.ArgumentParser(description='根据组合生成跨数据库SQL骨架')
    parser.add_argument('--combinations_file', type=str,
                       default='benchmark/generation/cross_database/database_combinations.json',
                       help='数据库组合文件')
    parser.add_argument('--plan_file', type=str,
                       default='benchmark/generation/cross_database/generation_plan.json',
                       help='生成计划文件')
    parser.add_argument('--sql_dir', type=str,
                       default='benchmark/data/beijing/output/single',
                       help='单数据库SQL目录')
    parser.add_argument('--output_dir', type=str,
                       default='benchmark/generation/cross_database/skeletons',
                       help='输出目录')
    
    args = parser.parse_args()
    
    # 动态导入模块（因为模块名以数字开头）
    sys.path.insert(0, 'benchmark/generation/cross_database')
    
    # 导入1select_candidates
    spec1 = importlib.util.spec_from_file_location(
        "select_candidates", 
        "benchmark/generation/cross_database/1select_candidates.py"
    )
    select_module = importlib.util.module_from_spec(spec1)
    spec1.loader.exec_module(select_module)
    load_single_database_sqls = select_module.load_single_database_sqls
    select_candidates = select_module.select_candidates
    
    # 导入2generate_cross_db_skeletons
    spec2 = importlib.util.spec_from_file_location(
        "generate_cross_db_skeletons", 
        "benchmark/generation/cross_database/2generate_cross_db_skeletons.py"
    )
    generate_module = importlib.util.module_from_spec(spec2)
    spec2.loader.exec_module(generate_module)
    generate_cross_database_skeletons = generate_module.generate_cross_database_skeletons
    
    # 加载组合和计划
    combinations_data = load_combinations(args.combinations_file)
    with open(args.plan_file, 'r', encoding='utf-8') as f:
        plan = json.load(f)
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    print("=" * 70)
    print("生成跨数据库SQL骨架")
    print("=" * 70)
    
    total_generated = 0
    
    # 预先加载所有SQL（避免重复加载）
    print("\n加载单数据库SQL...")
    all_sqls = load_single_database_sqls(args.sql_dir)
    print(f"找到 {len(all_sqls)} 条包含JOIN的SQL")
    
    # 生成2数据库组合的骨架
    print(f"\n1. 生成跨2数据库的SQL骨架...")
    print(f"   组合数: {len(combinations_data['2db_combinations'])}")
    print(f"   每种组合: {plan['2db']['per_combination']} 个骨架")
    
    for i, combo in enumerate(combinations_data['2db_combinations'], 1):
        combo_name = f"{combo[0]}_{combo[1]}"
        output_file = os.path.join(args.output_dir, f"2db_{combo_name}_skeletons.json")
        
        print(f"   [{i}/{len(combinations_data['2db_combinations'])}] {combo[0]} <-> {combo[1]}...", end=' ', flush=True)
        
        # 选择候选SQL
        candidates = select_candidates(all_sqls, num_candidates=plan['2db']['per_combination'] * 2, min_tables=2, max_tables=5)
        if len(candidates) < plan['2db']['per_combination']:
            while len(candidates) < plan['2db']['per_combination']:
                candidates.extend(candidates[:plan['2db']['per_combination'] - len(candidates)])
            candidates = candidates[:plan['2db']['per_combination']]
        else:
            candidates = candidates[:plan['2db']['per_combination']]
        
        # 创建临时候选文件
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False, encoding='utf-8') as f:
            json.dump(candidates, f, ensure_ascii=False, indent=2)
            temp_file = f.name
        
        # 生成骨架
        generate_cross_database_skeletons(
            temp_file,
            combo,
            output_file
        )
        
        os.unlink(temp_file)
        
        # 统计
        with open(output_file, 'r', encoding='utf-8') as f:
            skeletons = json.load(f)
        total_generated += len(skeletons)
        print(f"✅ {len(skeletons)} 个骨架")
    
    # 生成3数据库组合的骨架
    print(f"\n2. 生成跨3数据库的SQL骨架...")
    print(f"   组合数: {len(combinations_data['3db_combinations'])}")
    print(f"   每种组合: {plan['3db']['per_combination']} 个骨架")
    
    for i, combo in enumerate(combinations_data['3db_combinations'], 1):
        combo_name = f"{combo[0]}_{combo[1]}_{combo[2]}"
        output_file = os.path.join(args.output_dir, f"3db_{combo_name}_skeletons.json")
        
        print(f"   [{i}/{len(combinations_data['3db_combinations'])}] {combo[0]} + {combo[1]} + {combo[2]}...", end=' ', flush=True)
        
        # 选择候选SQL
        candidates = select_candidates(all_sqls, num_candidates=plan['3db']['per_combination'] * 2, min_tables=2, max_tables=5)
        if len(candidates) < plan['3db']['per_combination']:
            while len(candidates) < plan['3db']['per_combination']:
                candidates.extend(candidates[:plan['3db']['per_combination'] - len(candidates)])
            candidates = candidates[:plan['3db']['per_combination']]
        else:
            candidates = candidates[:plan['3db']['per_combination']]
        
        # 创建临时候选文件
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False, encoding='utf-8') as f:
            json.dump(candidates, f, ensure_ascii=False, indent=2)
            temp_file = f.name
        
        # 生成骨架
        generate_cross_database_skeletons(
            temp_file,
            combo,
            output_file
        )
        
        os.unlink(temp_file)
        
        # 统计
        with open(output_file, 'r', encoding='utf-8') as f:
            skeletons = json.load(f)
        total_generated += len(skeletons)
        print(f"✅ {len(skeletons)} 个骨架")
    
    # 生成4数据库组合的骨架
    print(f"\n3. 生成跨4数据库的SQL骨架...")
    print(f"   组合数: {len(combinations_data['4db_combinations'])}")
    print(f"   每种组合: {plan['4db']['per_combination']} 个骨架")
    
    for i, combo in enumerate(combinations_data['4db_combinations'], 1):
        combo_name = f"{combo[0]}_{combo[1]}_{combo[2]}_{combo[3]}"
        output_file = os.path.join(args.output_dir, f"4db_{combo_name}_skeletons.json")
        
        print(f"   [{i}/{len(combinations_data['4db_combinations'])}] {combo[0]} + {combo[1]} + {combo[2]} + {combo[3]}...", end=' ', flush=True)
        
        # 选择候选SQL
        candidates = select_candidates(all_sqls, num_candidates=plan['4db']['per_combination'] * 2, min_tables=2, max_tables=5)
        if len(candidates) < plan['4db']['per_combination']:
            while len(candidates) < plan['4db']['per_combination']:
                candidates.extend(candidates[:plan['4db']['per_combination'] - len(candidates)])
            candidates = candidates[:plan['4db']['per_combination']]
        else:
            candidates = candidates[:plan['4db']['per_combination']]
        
        # 创建临时候选文件
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False, encoding='utf-8') as f:
            json.dump(candidates, f, ensure_ascii=False, indent=2)
            temp_file = f.name
        
        # 生成骨架
        generate_cross_database_skeletons(
            temp_file,
            combo,
            output_file
        )
        
        os.unlink(temp_file)
        
        # 统计
        with open(output_file, 'r', encoding='utf-8') as f:
            skeletons = json.load(f)
        total_generated += len(skeletons)
        print(f"✅ {len(skeletons)} 个骨架")
    
    print(f"\n" + "=" * 70)
    print(f"完成！总共生成 {total_generated} 个SQL骨架")
    print(f"输出目录: {args.output_dir}")
    print("=" * 70)

if __name__ == '__main__':
    main()
