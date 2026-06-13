#!/usr/bin/env python3
"""
Generate more JOIN-version SQL (simplified; calls functions directly without blocking)

Based on existing results, generate only new SQL and avoid duplicates
"""

import os
import json
import argparse
import random
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
import sys

# Import SQL fill module
sys.path.insert(0, os.path.dirname(__file__))
from cross_db_2fill_sql_placeholders_join import (
    load_multiple_schemas,
    process_cross_database_skeleton
)

# Target counts
TARGET_COUNTS = {
    2: 359,  # cross 2 databases
    3: 105,  # cross 3 databases
    4: 2     # cross 4 databases
}

def get_existing_skeletons(sql_dir):
    """Get skeleton identifiers already used (based on existing result SQL files)."""
    existing_skeleton_signatures = set()
    
    for f in os.listdir(sql_dir):
        if f.startswith('cross_db_generated_sql_') and f.endswith('.json'):
            file_path = os.path.join(sql_dir, f)
            try:
                with open(file_path, 'r', encoding='utf-8') as file:
                    data = json.load(file)
                    results = data.get('results', [])
                    
                    if results is not None and len(results) > 0:
                        databases = sorted(data.get('databases', []))
                        table_db_mapping = data.get('table_database_mapping', {})
                        tables = sorted(table_db_mapping.keys())
                        
                        if len(databases) >= 2 and len(tables) >= 2:
                            table1 = tables[0]
                            table2 = tables[1] if len(tables) > 1 else None
                            if table2:
                                signature = tuple(sorted(databases[:2]) + sorted([table1, table2]))
                                existing_skeleton_signatures.add(signature)
            except:
                pass
    
    return existing_skeleton_signatures

def count_needed_by_db_count(sql_dir):
    """Count how many more SQL are needed for each database-count category."""
    needed = {}
    
    for db_count in [2, 3, 4]:
        target = TARGET_COUNTS[db_count]
        current = 0
        
        for f in os.listdir(sql_dir):
            if f.startswith('cross_db_generated_sql_') and f.endswith('.json'):
                try:
                    file_path = os.path.join(sql_dir, f)
                    with open(file_path, 'r', encoding='utf-8') as file:
                        data = json.load(file)
                        results = data.get('results', [])
                        num_databases = data.get('metadata', {}).get('num_databases', 
                                                                    len(data.get('databases', [])))
                        
                        if results is not None and len(results) > 0 and num_databases == db_count:
                            current += 1
                except:
                    pass
        
        needed[db_count] = max(0, target - current)
    
    return needed

def filter_skeletons_by_db_count(skeletons, db_count):
    """Filter skeletons for the specified database count."""
    return [s for s in skeletons if s.get('num_databases', len(s.get('databases', []))) == db_count]

def generate_more_sqls(skeleton_file, sql_dir, graph_dir, output_dir, database_dir, 
                       max_workers=5, max_retries=3, ignore_existing=False):
    """Generate more SQL (calls functions directly without blocking).
    
    Args:
        ignore_existing: If True, ignore existing results and regenerate (for cleanup and rerun)
    """
    
    if ignore_existing:
        # Ignore existing results and generate a fresh batch
        print("=" * 70)
        print("Regenerate JOIN-version SQL (ignoring existing results)")
        print("=" * 70)
        print("\nWill generate a new batch of SQL using different skeletons...")
        needed = {2: 100}  # Generate about 100 each time; ~22.5% success rate, expect 20-25 with results
    else:
        # 1. Count how many more are needed
        needed = count_needed_by_db_count(sql_dir)
        
        print("=" * 70)
        print("Generate more JOIN-version SQL")
        print("=" * 70)
        print("\nTarget counts:")
        for db_count in sorted(needed.keys()):
            target = TARGET_COUNTS[db_count]
            current = target - needed[db_count]
            print(f"  Cross {db_count} databases: {current} / {target} (still need {needed[db_count]})")
        
        total_needed = sum(needed.values())
        if total_needed == 0:
            print("\n✅ All target counts reached; no need to generate more SQL")
            return
        
        print(f"\nTotal still needed: {total_needed} SQL")
    
    # 2. Load skeleton file
    print("\nLoading SQL skeletons...")
    with open(skeleton_file, 'r', encoding='utf-8') as f:
        all_skeletons = json.load(f)
    
    print(f"  Total skeletons: {len(all_skeletons)}")
    
    # 3. Get skeleton identifiers already used (skip if ignore_existing is True)
    if ignore_existing:
        existing_skeleton_signatures = set()
        print(f"  Ignoring existing results; will use all available skeletons")
    else:
        existing_skeleton_signatures = get_existing_skeletons(sql_dir)
        print(f"  Skeletons already used: {len(existing_skeleton_signatures)}")
    
    # 4. Classify skeletons by database count
    skeletons_by_db_count = {}
    for db_count in [2, 3, 4]:
        skeletons_by_db_count[db_count] = filter_skeletons_by_db_count(all_skeletons, db_count)
        print(f"  Skeletons for {db_count} databases: {len(skeletons_by_db_count[db_count])}")
    
    # 5. Select unused skeletons for each database-count category
    selected_skeletons = []
    
    for db_count in sorted(needed.keys()):
        if needed[db_count] == 0:
            continue
        
        available_skeletons = skeletons_by_db_count[db_count]
        
        if ignore_existing:
            # Ignore existing results and select randomly
            unused_skeletons = available_skeletons
        else:
            # Filter out already-used skeletons
            unused_skeletons = []
            for skeleton in available_skeletons:
                databases = sorted(skeleton.get('databases', []))
                table_db_mapping = skeleton.get('table_database_mapping', {})
                tables = sorted(table_db_mapping.keys())
                
                if len(databases) >= 2 and len(tables) >= 2:
                    table1 = tables[0]
                    table2 = tables[1] if len(tables) > 1 else None
                    if table2:
                        signature = tuple(sorted(databases[:2]) + sorted([table1, table2]))
                        if signature not in existing_skeleton_signatures:
                            unused_skeletons.append(skeleton)
        
        print(f"\nCross {db_count} databases:")
        print(f"  Available skeletons: {len(available_skeletons)}")
        print(f"  Unused skeletons: {len(unused_skeletons)}")
        print(f"  Need to generate: {needed[db_count]}")
        
        # Select required count (4x because success rate is ~22.5%)
        to_generate = min(needed[db_count] * 4, len(unused_skeletons))  # 4x for safety margin
        if to_generate > 0:
            selected = random.sample(unused_skeletons, to_generate) if len(unused_skeletons) >= to_generate else unused_skeletons
            print(f"  Selected for generation: {len(selected)} skeletons")
            selected_skeletons.extend(selected)
    
    if not selected_skeletons:
        print("\n⚠️  No unused skeletons available; cannot generate more SQL")
        return
    
    print(f"\nTotal selected: {len(selected_skeletons)} skeletons for generation")
    
    # 6. Load schemas for all databases
    print("\nLoading database schemas...")
    all_databases = set()
    for skeleton in selected_skeletons:
        all_databases.update(skeleton.get('databases', []))
    
    schemas = load_multiple_schemas(all_databases, database_dir)
    print(f"Successfully loaded schemas for {len(schemas)} databases")
    
    # 7. Process each skeleton directly (no subprocess, avoids blocking)
    print(f"\n{'='*70}")
    print(f"Starting SQL generation ({len(selected_skeletons)} total)...")
    print(f"Concurrency: {max_workers}, max retries: {max_retries}")
    print(f"{'='*70}\n")
    
    success_count = 0
    failed_count = 0
    results_with_data = 0  # SQL count with non-empty results
    
    import time
    start_time = time.time()
    last_print_time = start_time
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = []
        for i, skeleton in enumerate(selected_skeletons):
            future = executor.submit(
                process_cross_database_skeleton,
                skeleton, schemas, graph_dir, output_dir,
                database_dir, max_retries
            )
            futures.append((future, i+1))
        
        # Collect results (show progress bar and live stats)
        completed = 0
        pbar = tqdm(total=len(futures), desc="Generation progress", ncols=120, unit="item",
                   bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]')
        
        for future, idx_num in futures:
            try:
                # Set timeout to avoid a single task hanging too long
                idx, success, message = future.result(timeout=600)  # 10-minute timeout per SQL generation
                completed += 1
                pbar.update(1)
                
                if success:
                    success_count += 1
                    # Check whether result data exists
                    try:
                        output_file = os.path.join(output_dir, f"cross_db_generated_sql_{idx}.json")
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
                
                # Print detailed stats every 5 seconds or every 10 items (ensure user sees progress)
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
                    # Extra log line (ensure visibility)
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

def main():
    parser = argparse.ArgumentParser(description='Generate more JOIN-version SQL (skip existing results)')
    parser.add_argument('--skeleton_file', type=str,
                       default='benchmark/generation/cross_database/cross_db_skeletons_join.json',
                       help='SQL skeleton file')
    parser.add_argument('--graph_dir', type=str,
                       default='benchmark/data/beijing/output/cross_db_graph_join',
                       help='Graph file directory')
    parser.add_argument('--database_dir', type=str,
                       default='benchmark/data/beijing/database_chinese',
                       help='Database directory')
    parser.add_argument('--output_dir', type=str,
                       default='benchmark/data/beijing/output/cross_db_single_join',
                       help='Output directory')
    parser.add_argument('--sql_dir', type=str,
                       default='benchmark/data/beijing/output/cross_db_single_join',
                       help='SQL file directory (for counting existing results)')
    parser.add_argument('--max_retries', type=int, default=3,
                       help='Maximum retry count')
    parser.add_argument('--max_workers', type=int, default=5,
                       help='Number of concurrent threads')
    parser.add_argument('--ignore_existing', action='store_true',
                       help='Ignore existing results and regenerate (for cleanup and rerun)')
    
    args = parser.parse_args()
    
    generate_more_sqls(
        args.skeleton_file,
        args.sql_dir,
        args.graph_dir,
        args.output_dir,
        args.database_dir,
        args.max_workers,
        args.max_retries,
        args.ignore_existing
    )

if __name__ == '__main__':
    main()
