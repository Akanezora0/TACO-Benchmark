#!/usr/bin/env python3
"""
Generate cross-database SQL skeletons from database combinations.
"""

import os
import json
import argparse
import importlib.util
import sys
import tempfile

def load_combinations(combinations_file):
    """Load database combinations."""
    with open(combinations_file, 'r', encoding='utf-8') as f:
        return json.load(f)

def main():
    parser = argparse.ArgumentParser(description='Generate cross-database SQL skeletons from combinations')
    parser.add_argument('--combinations_file', type=str,
                       default='benchmark/generation/cross_database/database_combinations.json',
                       help='Database combinations file')
    parser.add_argument('--plan_file', type=str,
                       default='benchmark/generation/cross_database/generation_plan.json',
                       help='Generation plan file')
    parser.add_argument('--sql_dir', type=str,
                       default='benchmark/data/beijing/output/single',
                       help='Single-database SQL directory')
    parser.add_argument('--output_dir', type=str,
                       default='benchmark/generation/cross_database/skeletons',
                       help='Output directory')
    
    args = parser.parse_args()
    
    # Dynamic import (module names start with a digit)
    sys.path.insert(0, 'benchmark/generation/cross_database')
    
    # Import 1select_candidates
    spec1 = importlib.util.spec_from_file_location(
        "select_candidates", 
        "benchmark/generation/cross_database/1select_candidates.py"
    )
    select_module = importlib.util.module_from_spec(spec1)
    spec1.loader.exec_module(select_module)
    load_single_database_sqls = select_module.load_single_database_sqls
    select_candidates = select_module.select_candidates
    
    # Import 2generate_cross_db_skeletons
    spec2 = importlib.util.spec_from_file_location(
        "generate_cross_db_skeletons", 
        "benchmark/generation/cross_database/2generate_cross_db_skeletons.py"
    )
    generate_module = importlib.util.module_from_spec(spec2)
    spec2.loader.exec_module(generate_module)
    generate_cross_database_skeletons = generate_module.generate_cross_database_skeletons
    
    # Load combinations and plan
    combinations_data = load_combinations(args.combinations_file)
    with open(args.plan_file, 'r', encoding='utf-8') as f:
        plan = json.load(f)
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    print("=" * 70)
    print("Generating cross-database SQL skeletons")
    print("=" * 70)
    
    total_generated = 0
    
    # Preload all SQL (avoid repeated loading)
    print("\nLoading single-database SQL...")
    all_sqls = load_single_database_sqls(args.sql_dir)
    print(f"Found {len(all_sqls)} SQL entries containing JOIN")
    
    # Generate skeletons for 2-database combinations
    print(f"\n1. Generating cross-2-database SQL skeletons...")
    print(f"   Combinations: {len(combinations_data['2db_combinations'])}")
    print(f"   Per combination: {plan['2db']['per_combination']} skeletons")
    
    for i, combo in enumerate(combinations_data['2db_combinations'], 1):
        combo_name = f"{combo[0]}_{combo[1]}"
        output_file = os.path.join(args.output_dir, f"2db_{combo_name}_skeletons.json")
        
        print(f"   [{i}/{len(combinations_data['2db_combinations'])}] {combo[0]} <-> {combo[1]}...", end=' ', flush=True)
        
        # Select candidate SQL
        candidates = select_candidates(all_sqls, num_candidates=plan['2db']['per_combination'] * 2, min_tables=2, max_tables=5)
        if len(candidates) < plan['2db']['per_combination']:
            while len(candidates) < plan['2db']['per_combination']:
                candidates.extend(candidates[:plan['2db']['per_combination'] - len(candidates)])
            candidates = candidates[:plan['2db']['per_combination']]
        else:
            candidates = candidates[:plan['2db']['per_combination']]
        
        # Create temporary candidates file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False, encoding='utf-8') as f:
            json.dump(candidates, f, ensure_ascii=False, indent=2)
            temp_file = f.name
        
        # Generate skeletons
        generate_cross_database_skeletons(
            temp_file,
            combo,
            output_file
        )
        
        os.unlink(temp_file)
        
        # Statistics
        with open(output_file, 'r', encoding='utf-8') as f:
            skeletons = json.load(f)
        total_generated += len(skeletons)
        print(f"✅ {len(skeletons)} skeletons")
    
    # Generate skeletons for 3-database combinations
    print(f"\n2. Generating cross-3-database SQL skeletons...")
    print(f"   Combinations: {len(combinations_data['3db_combinations'])}")
    print(f"   Per combination: {plan['3db']['per_combination']} skeletons")
    
    for i, combo in enumerate(combinations_data['3db_combinations'], 1):
        combo_name = f"{combo[0]}_{combo[1]}_{combo[2]}"
        output_file = os.path.join(args.output_dir, f"3db_{combo_name}_skeletons.json")
        
        print(f"   [{i}/{len(combinations_data['3db_combinations'])}] {combo[0]} + {combo[1]} + {combo[2]}...", end=' ', flush=True)
        
        # Select candidate SQL
        candidates = select_candidates(all_sqls, num_candidates=plan['3db']['per_combination'] * 2, min_tables=2, max_tables=5)
        if len(candidates) < plan['3db']['per_combination']:
            while len(candidates) < plan['3db']['per_combination']:
                candidates.extend(candidates[:plan['3db']['per_combination'] - len(candidates)])
            candidates = candidates[:plan['3db']['per_combination']]
        else:
            candidates = candidates[:plan['3db']['per_combination']]
        
        # Create temporary candidates file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False, encoding='utf-8') as f:
            json.dump(candidates, f, ensure_ascii=False, indent=2)
            temp_file = f.name
        
        # Generate skeletons
        generate_cross_database_skeletons(
            temp_file,
            combo,
            output_file
        )
        
        os.unlink(temp_file)
        
        # Statistics
        with open(output_file, 'r', encoding='utf-8') as f:
            skeletons = json.load(f)
        total_generated += len(skeletons)
        print(f"✅ {len(skeletons)} skeletons")
    
    # Generate skeletons for 4-database combinations
    print(f"\n3. Generating cross-4-database SQL skeletons...")
    print(f"   Combinations: {len(combinations_data['4db_combinations'])}")
    print(f"   Per combination: {plan['4db']['per_combination']} skeletons")
    
    for i, combo in enumerate(combinations_data['4db_combinations'], 1):
        combo_name = f"{combo[0]}_{combo[1]}_{combo[2]}_{combo[3]}"
        output_file = os.path.join(args.output_dir, f"4db_{combo_name}_skeletons.json")
        
        print(f"   [{i}/{len(combinations_data['4db_combinations'])}] {combo[0]} + {combo[1]} + {combo[2]} + {combo[3]}...", end=' ', flush=True)
        
        # Select candidate SQL
        candidates = select_candidates(all_sqls, num_candidates=plan['4db']['per_combination'] * 2, min_tables=2, max_tables=5)
        if len(candidates) < plan['4db']['per_combination']:
            while len(candidates) < plan['4db']['per_combination']:
                candidates.extend(candidates[:plan['4db']['per_combination'] - len(candidates)])
            candidates = candidates[:plan['4db']['per_combination']]
        else:
            candidates = candidates[:plan['4db']['per_combination']]
        
        # Create temporary candidates file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False, encoding='utf-8') as f:
            json.dump(candidates, f, ensure_ascii=False, indent=2)
            temp_file = f.name
        
        # Generate skeletons
        generate_cross_database_skeletons(
            temp_file,
            combo,
            output_file
        )
        
        os.unlink(temp_file)
        
        # Statistics
        with open(output_file, 'r', encoding='utf-8') as f:
            skeletons = json.load(f)
        total_generated += len(skeletons)
        print(f"✅ {len(skeletons)} skeletons")
    
    print(f"\n" + "=" * 70)
    print(f"Done! Generated {total_generated} SQL skeletons in total")
    print(f"Output directory: {args.output_dir}")
    print("=" * 70)

if __name__ == '__main__':
    main()
