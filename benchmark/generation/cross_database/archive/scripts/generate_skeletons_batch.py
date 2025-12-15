#!/usr/bin/env python3
"""
批量生成跨数据库SQL骨架（简化版）
"""

import os
import json
import sys
import importlib.util
import tempfile

# 动态导入
sys.path.insert(0, 'benchmark/generation/cross_database')

spec1 = importlib.util.spec_from_file_location("select_candidates", "benchmark/generation/cross_database/1select_candidates.py")
select_module = importlib.util.module_from_spec(spec1)
spec1.loader.exec_module(select_module)

spec2 = importlib.util.spec_from_file_location("generate_skeletons", "benchmark/generation/cross_database/2generate_cross_db_skeletons.py")
generate_module = importlib.util.module_from_spec(spec2)
spec2.loader.exec_module(generate_module)

# 加载数据
with open('benchmark/generation/cross_database/database_combinations.json', 'r', encoding='utf-8') as f:
    combos = json.load(f)

with open('benchmark/generation/cross_database/generation_plan.json', 'r', encoding='utf-8') as f:
    plan = json.load(f)

output_dir = 'benchmark/generation/cross_database/skeletons'
os.makedirs(output_dir, exist_ok=True)

print("=" * 70)
print("批量生成跨数据库SQL骨架")
print("=" * 70)

# 加载SQL
print("\n加载单数据库SQL...")
all_sqls = select_module.load_single_database_sqls('benchmark/data/beijing/output/single')
print(f"找到 {len(all_sqls)} 条包含JOIN的SQL")

total_generated = 0

# 生成2数据库组合
print(f"\n生成跨2数据库的SQL骨架...")
print(f"组合数: {len(combos['2db_combinations'])}")
print(f"每种组合: {plan['2db']['per_combination']} 个骨架\n")

for i, combo in enumerate(combos['2db_combinations'], 1):
    combo_name = f"{combo[0]}_{combo[1]}"
    output_file = os.path.join(output_dir, f"2db_{combo_name}_skeletons.json")
    
    print(f"[{i}/{len(combos['2db_combinations'])}] {combo[0]} <-> {combo[1]}...", end=' ', flush=True)
    
    # 选择候选
    candidates = select_module.select_candidates(
        all_sqls, 
        num_candidates=plan['2db']['per_combination'] * 2, 
        min_tables=2, 
        max_tables=5
    )
    
    if len(candidates) < plan['2db']['per_combination']:
        while len(candidates) < plan['2db']['per_combination']:
            candidates.extend(candidates[:plan['2db']['per_combination'] - len(candidates)])
        candidates = candidates[:plan['2db']['per_combination']]
    else:
        candidates = candidates[:plan['2db']['per_combination']]
    
    # 生成骨架
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False, encoding='utf-8') as f:
        json.dump(candidates, f, ensure_ascii=False, indent=2)
        temp_file = f.name
    
    generate_module.generate_cross_database_skeletons(temp_file, combo, output_file)
    os.unlink(temp_file)
    
    # 统计
    with open(output_file, 'r', encoding='utf-8') as f:
        skeletons = json.load(f)
    total_generated += len(skeletons)
    print(f"✅ {len(skeletons)} 个骨架")

# 生成3数据库组合
print(f"\n生成跨3数据库的SQL骨架...")
print(f"组合数: {len(combos['3db_combinations'])}")
print(f"每种组合: {plan['3db']['per_combination']} 个骨架\n")

for i, combo in enumerate(combos['3db_combinations'], 1):
    combo_name = f"{combo[0]}_{combo[1]}_{combo[2]}"
    output_file = os.path.join(output_dir, f"3db_{combo_name}_skeletons.json")
    
    print(f"[{i}/{len(combos['3db_combinations'])}] {combo[0]} + {combo[1]} + {combo[2]}...", end=' ', flush=True)
    
    # 选择候选
    candidates = select_module.select_candidates(
        all_sqls, 
        num_candidates=plan['3db']['per_combination'] * 2, 
        min_tables=2, 
        max_tables=5
    )
    
    if len(candidates) < plan['3db']['per_combination']:
        while len(candidates) < plan['3db']['per_combination']:
            candidates.extend(candidates[:plan['3db']['per_combination'] - len(candidates)])
        candidates = candidates[:plan['3db']['per_combination']]
    else:
        candidates = candidates[:plan['3db']['per_combination']]
    
    # 生成骨架
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False, encoding='utf-8') as f:
        json.dump(candidates, f, ensure_ascii=False, indent=2)
        temp_file = f.name
    
    generate_module.generate_cross_database_skeletons(temp_file, combo, output_file)
    os.unlink(temp_file)
    
    # 统计
    with open(output_file, 'r', encoding='utf-8') as f:
        skeletons = json.load(f)
    total_generated += len(skeletons)
    print(f"✅ {len(skeletons)} 个骨架")

# 生成4数据库组合
print(f"\n生成跨4数据库的SQL骨架...")
print(f"组合数: {len(combos['4db_combinations'])}")
print(f"每种组合: {plan['4db']['per_combination']} 个骨架\n")

for i, combo in enumerate(combos['4db_combinations'], 1):
    combo_name = f"{combo[0]}_{combo[1]}_{combo[2]}_{combo[3]}"
    output_file = os.path.join(output_dir, f"4db_{combo_name}_skeletons.json")
    
    print(f"[{i}/{len(combos['4db_combinations'])}] {combo[0]} + {combo[1]} + {combo[2]} + {combo[3]}...", end=' ', flush=True)
    
    # 选择候选
    candidates = select_module.select_candidates(
        all_sqls, 
        num_candidates=plan['4db']['per_combination'] * 2, 
        min_tables=2, 
        max_tables=5
    )
    
    if len(candidates) < plan['4db']['per_combination']:
        while len(candidates) < plan['4db']['per_combination']:
            candidates.extend(candidates[:plan['4db']['per_combination'] - len(candidates)])
        candidates = candidates[:plan['4db']['per_combination']]
    else:
        candidates = candidates[:plan['4db']['per_combination']]
    
    # 生成骨架
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False, encoding='utf-8') as f:
        json.dump(candidates, f, ensure_ascii=False, indent=2)
        temp_file = f.name
    
    generate_module.generate_cross_database_skeletons(temp_file, combo, output_file)
    os.unlink(temp_file)
    
    # 统计
    with open(output_file, 'r', encoding='utf-8') as f:
        skeletons = json.load(f)
    total_generated += len(skeletons)
    print(f"✅ {len(skeletons)} 个骨架")

print(f"\n" + "=" * 70)
print(f"完成！总共生成 {total_generated} 个SQL骨架")
print(f"输出目录: {output_dir}")
print("=" * 70)


