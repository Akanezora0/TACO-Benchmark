#!/usr/bin/env python3
"""
Generate JOIN SQL for 3- and 4-database queries
"""

import os
import sys
import argparse

# Import functions from the main generation script
sys.path.insert(0, os.path.dirname(__file__))
import importlib.util
spec = importlib.util.spec_from_file_location(
    "generate_more_join_sqls_simple",
    os.path.join(os.path.dirname(__file__), "generate_more_join_sqls.py")
)
gen_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(gen_module)

load_multiple_schemas = gen_module.load_multiple_schemas

def main():
    # Get script directory and project root
    script_dir = os.path.dirname(os.path.abspath(__file__))
    # Project root: 3 levels up from cross_database
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(script_dir)))
    
    parser = argparse.ArgumentParser(description='Generate JOIN SQL for 3- and 4-database queries')
    parser.add_argument('--skeleton_file', type=str,
                       default=os.path.join(script_dir, 'cross_db_skeletons_join.json'),
                       help='SQL skeleton file')
    parser.add_argument('--graph_dir', type=str,
                       default=os.path.join(script_dir, 'cross_db_graphs_join'),
                       help='Graph file directory')
    parser.add_argument('--database_dir', type=str,
                       default=os.path.join(project_root, 'benchmark', 'data', 'beijing', 'database_chinese'),
                       help='Database directory')
    parser.add_argument('--output_dir', type=str,
                       default=os.path.join(project_root, 'benchmark', 'data', 'beijing', 'output', 'cross_db_single_join'),
                       help='Output directory')
    parser.add_argument('--sql_dir', type=str,
                       default=os.path.join(project_root, 'benchmark', 'data', 'beijing', 'output', 'cross_db_single_join'),
                       help='SQL file directory (for counting existing results)')
    parser.add_argument('--max_workers', type=int, default=10,
                       help='Maximum concurrency (default 10; lower for complex 3/4-database SQL)')
    parser.add_argument('--max_retries', type=int, default=3,
                       help='Maximum retry count')
    parser.add_argument('--only_3db', action='store_true',
                       help='Generate SQL for 3 databases only')
    parser.add_argument('--only_4db', action='store_true',
                       help='Generate SQL for 4 databases only')
    parser.add_argument('--num_3db', type=int, default=None,
                       help='Number of 3-database SQL to generate (auto-calculated from target by default)')
    parser.add_argument('--num_4db', type=int, default=None,
                       help='Number of 4-database SQL to generate (auto-calculated from target by default)')
    
    args = parser.parse_args()
    
    # Target counts
    TARGET_COUNTS = {
        2: 359,  # cross 2 databases (already complete; do not generate)
        3: 105,  # cross 3 databases
        4: 2     # cross 4 databases
    }
    
    print("=" * 70)
    print("Generate JOIN SQL for 3- and 4-database queries")
    print("=" * 70)
    print(f"\nSkeleton file: {args.skeleton_file}")
    print(f"Graph directory: {args.graph_dir}")
    print(f"Database directory: {args.database_dir}")
    print(f"Output directory: {args.output_dir}")
    print(f"Max concurrency: {args.max_workers}")
    print()
    
    # Determine which database counts to generate
    db_counts_to_generate = []
    if args.only_3db:
        db_counts_to_generate = [3]
    elif args.only_4db:
        db_counts_to_generate = [4]
    else:
        db_counts_to_generate = [3, 4]
    
    print(f"Will generate SQL for database counts: {db_counts_to_generate}")
    print()
    
    # Count existing results
    import json
    current_counts = {3: 0, 4: 0}
    
    if os.path.exists(args.sql_dir):
        for f in os.listdir(args.sql_dir):
            if f.startswith('cross_db_generated_sql_') and f.endswith('.json'):
                try:
                    file_path = os.path.join(args.sql_dir, f)
                    with open(file_path, 'r', encoding='utf-8') as file:
                        data = json.load(file)
                        results = data.get('results', [])
                        num_databases = data.get('metadata', {}).get('num_databases', 
                                                                    len(data.get('databases', [])))
                        
                        if results is not None and len(results) > 0 and num_databases in [3, 4]:
                            current_counts[num_databases] += 1
                except:
                    pass
    
    print("Current counts:")
    for db_count in [3, 4]:
        target = TARGET_COUNTS[db_count]
        current = current_counts[db_count]
        needed = max(0, target - current)
        print(f"  {db_count} databases: {current} / {target} (still need {needed})")
    
    print()
    
    # Determine generation counts
    needed = {}
    for db_count in db_counts_to_generate:
        target = TARGET_COUNTS[db_count]
        current = current_counts[db_count]
        
        if db_count == 3 and args.num_3db is not None:
            needed[db_count] = args.num_3db
        elif db_count == 4 and args.num_4db is not None:
            needed[db_count] = args.num_4db
        else:
            needed[db_count] = max(0, target - current)
    
    # If all targets are already met
    total_needed = sum(needed.values())
    if total_needed == 0:
        print("✅ All target counts reached; no need to generate more SQL")
        return
    
    print(f"Will generate: {needed}")
    print()
    
    # Load skeleton file
    print("Loading SQL skeletons...")
    with open(args.skeleton_file, 'r', encoding='utf-8') as file:
        all_skeletons = json.load(file)
    
    # Filter skeletons for 3- and 4-database queries
    skeletons_3db = [s for s in all_skeletons if s.get('num_databases') == 3]
    skeletons_4db = [s for s in all_skeletons if s.get('num_databases') == 4]
    
    print(f"  Total skeletons: {len(all_skeletons)}")
    print(f"  Skeletons for 3 databases: {len(skeletons_3db)}")
    print(f"  Skeletons for 4 databases: {len(skeletons_4db)}")
    
    # Check whether enough skeletons exist
    if 3 in needed and len(skeletons_3db) < needed[3]:
        print(f"  ⚠️  Warning: 3-database skeleton count ({len(skeletons_3db)}) is below required count ({needed[3]})")
    if 4 in needed and len(skeletons_4db) < needed[4]:
        print(f"  ⚠️  Warning: 4-database skeleton count ({len(skeletons_4db)}) is below required count ({needed[4]})")
    
    # Collect all involved databases
    all_databases = set()
    for skeleton in all_skeletons:
        if skeleton.get('num_databases') in db_counts_to_generate:
            all_databases.update(skeleton.get('databases', []))
    
    print(f"\nDatabases involved: {sorted(all_databases)}")
    
    # Load schemas for all databases
    print("\nLoading database schemas...")
    schemas = load_multiple_schemas(list(all_databases), args.database_dir)
    print(f"Successfully loaded schemas for {len(schemas)} databases")
    
    if len(schemas) == 0:
        print("⚠️  Warning: no schemas loaded; check database directory path")
        return
    
    # Call generation logic (generate_more_sqls needs custom needed support)
    # Since generate_more_sqls recalculates needed internally, we temporarily adjust TARGET_COUNTS
    # or call the internal logic directly
    
    # generate_more_sqls recalculates needed, so call internal logic directly here
    
    get_existing_skeletons = gen_module.get_existing_skeletons
    filter_skeletons_by_db_count = gen_module.filter_skeletons_by_db_count
    process_cross_database_skeleton = gen_module.process_cross_database_skeleton
    from concurrent.futures import ThreadPoolExecutor, as_completed
    from tqdm import tqdm
    import time
    import random
    
    # Get skeleton identifiers already used
    existing_skeleton_signatures = get_existing_skeletons(args.sql_dir)
    print(f"  Skeletons already used: {len(existing_skeleton_signatures)}")
    
    # Classify skeletons by database count
    skeletons_by_db_count = {}
    for db_count in db_counts_to_generate:
        skeletons_by_db_count[db_count] = filter_skeletons_by_db_count(all_skeletons, db_count)
    
    # Select unused skeletons for each database-count category
    selected_skeletons = []
    
    for db_count in sorted(needed.keys()):
        if needed[db_count] == 0:
            continue
        
        available_skeletons = skeletons_by_db_count[db_count]
        
        # Filter out already-used skeletons
        unused_skeletons = []
        for skeleton in available_skeletons:
            databases = sorted(skeleton.get('databases', []))
            table_db_mapping = skeleton.get('table_database_mapping', {})
            tables = sorted(table_db_mapping.keys())
            
            if len(databases) >= 2 and len(tables) >= 2:
                signature = tuple(sorted(databases[:2]) + sorted([tables[0], tables[1] if len(tables) > 1 else tables[0]]))
                if signature not in existing_skeleton_signatures:
                    unused_skeletons.append(skeleton)
        
        print(f"\n  {db_count} databases:")
        print(f"    Available skeletons: {len(available_skeletons)}")
        print(f"    Unused skeletons: {len(unused_skeletons)}")
        print(f"    Need to generate: {needed[db_count]}")
        
        if len(unused_skeletons) == 0:
            print(f"    ⚠️  No unused skeletons available; cannot generate more SQL")
            continue
        
        # Select skeletons (prefer unused; random if not enough)
        num_to_select = min(needed[db_count] * 3, len(unused_skeletons))  # Assume ~33% success rate; generate 3x
        selected = random.sample(unused_skeletons, num_to_select)
        selected_skeletons.extend(selected)
        print(f"    Selected {len(selected)} skeletons")
    
    if len(selected_skeletons) == 0:
        print("\n⚠️  No skeletons available; cannot generate SQL")
        return
    
    # Start generation
    print(f"\n{'='*70}")
    print(f"Starting SQL generation ({len(selected_skeletons)} total)...")
    print(f"Concurrency: {args.max_workers}, max retries: {args.max_retries}")
    print(f"{'='*70}\n")
    
    success_count = 0
    failed_count = 0
    results_with_data = 0
    
    start_time = time.time()
    last_print_time = start_time
    
    with ThreadPoolExecutor(max_workers=args.max_workers) as executor:
        futures = []
        for i, skeleton in enumerate(selected_skeletons):
            future = executor.submit(
                process_cross_database_skeleton,
                skeleton, schemas, args.graph_dir, args.output_dir,
                args.database_dir, args.max_retries
            )
            futures.append((future, i+1))
        
        # Collect results
        completed = 0
        pbar = tqdm(total=len(futures), desc="Generation progress", ncols=120, unit="item",
                   bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]')
        
        for future, idx_num in futures:
            try:
                idx, success, message = future.result(timeout=600)  # 10-minute timeout
                completed += 1
                pbar.update(1)
                
                if success:
                    success_count += 1
                    # Check whether result data exists
                    try:
                        output_file = os.path.join(args.output_dir, f"cross_db_generated_sql_{idx}.json")
                        if os.path.exists(output_file):
                            with open(output_file, 'r', encoding='utf-8') as f:
                                data = json.load(f)
                                results = data.get('results', [])
                                if results and len(results) > 0:
                                    results_with_data += 1
                    except:
                        pass
                else:
                    failed_count += 1
                
                # Print detailed stats every 5 seconds
                current_time = time.time()
                if (current_time - last_print_time >= 5) or (completed % 10 == 0) or (completed == len(selected_skeletons)):
                    elapsed = current_time - start_time
                    rate = completed / elapsed if elapsed > 0 else 0
                    remaining = (len(selected_skeletons) - completed) / rate if rate > 0 else 0
                    pbar.set_postfix({
                        'success': f'{success_count}',
                        'with_results': f'{results_with_data}',
                        'failed': f'{failed_count}',
                        'rate': f'{rate:.2f}/s'
                    })
                    if current_time - last_print_time >= 5:
                        print(f"\n[Live] Completed: {completed}/{len(selected_skeletons)} | "
                              f"Success: {success_count} (with results: {results_with_data}) | "
                              f"Failed: {failed_count} | "
                              f"Rate: {rate:.2f}/s | "
                              f"Estimated remaining: {remaining/60:.1f} min", flush=True)
                        last_print_time = current_time
                    
            except Exception as e:
                failed_count += 1
                completed += 1
                pbar.update(1)
                print(f"\n[Exception #{completed}] {str(e)[:200]}", flush=True)
        
        pbar.close()
    
    elapsed_time = time.time() - start_time
    print(f"\n" + "=" * 70)
    print(f"Generation complete!")
    print(f"Total time: {elapsed_time/60:.2f} min")
    print(f"Success: {success_count}/{len(selected_skeletons)} ({success_count/len(selected_skeletons)*100:.1f}%)")
    print(f"With results: {results_with_data}/{len(selected_skeletons)} ({results_with_data/len(selected_skeletons)*100:.1f}%)")
    print(f"Failed: {failed_count}/{len(selected_skeletons)} ({failed_count/len(selected_skeletons)*100:.1f}%)")
    print("=" * 70)

if __name__ == '__main__':
    main()
