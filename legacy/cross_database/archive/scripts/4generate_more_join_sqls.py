#!/usr/bin/env python3
"""
Generate additional JOIN-variant SQL (without duplicating existing results).

Based on existing results, generate only new SQL to avoid duplicates.
"""

import os
import json
import argparse
from collections import defaultdict

# Target counts
TARGET_COUNTS = {
    2: 359,  # Cross 2 databases
    3: 105,  # Cross 3 databases
    4: 2     # Cross 4 databases
}

def get_existing_indices(sql_dir):
    """Get indices of existing SQL files."""
    existing_indices = set()
    
    for f in os.listdir(sql_dir):
        if f.startswith('cross_db_generated_sql_') and f.endswith('.json'):
            try:
                idx = int(f.replace('cross_db_generated_sql_', '').replace('.json', ''))
                existing_indices.add(idx)
            except:
                pass
    
    return existing_indices

def get_existing_skeletons(sql_dir):
    """Get identifiers of skeletons already used (from existing result SQL files).
    
    Use table-pair information as the unique identifier, since original_file may be missing.
    Format: (db1, db2, table1, table2) -> used
    """
    existing_skeleton_signatures = set()
    
    # Extract table-pair info from SQL files as unique identifiers
    for f in os.listdir(sql_dir):
        if f.startswith('cross_db_generated_sql_') and f.endswith('.json'):
            file_path = os.path.join(sql_dir, f)
            try:
                with open(file_path, 'r', encoding='utf-8') as file:
                    data = json.load(file)
                    results = data.get('results', [])
                    
                    # Only consider files with results
                    if results is not None and len(results) > 0:
                        databases = sorted(data.get('databases', []))
                        table_db_mapping = data.get('table_database_mapping', {})
                        
                        # Extract table names from table_database_mapping (without database prefix)
                        tables = sorted(table_db_mapping.keys())
                        
                        # Use database and table combination as unique identifier
                        if len(databases) >= 2 and len(tables) >= 2:
                            # Ensure tables correspond to databases
                            table1 = tables[0] if tables[0] in table_db_mapping else None
                            table2 = tables[1] if len(tables) > 1 and tables[1] in table_db_mapping else None
                            
                            if table1 and table2:
                                signature = tuple(sorted(databases[:2]) + sorted([table1, table2]))
                                existing_skeleton_signatures.add(signature)
            except Exception as e:
                pass
    
    return existing_skeleton_signatures

def count_needed_by_db_count(sql_dir):
    """Count how many more are needed per database-count category."""
    needed = {}
    
    for db_count in [2, 3, 4]:
        target = TARGET_COUNTS[db_count]
        current = 0
        
        # Count current results
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
                       max_workers=5, max_retries=3):
    """Generate more SQL."""
    
    # 1. Count how many more are needed
    needed = count_needed_by_db_count(sql_dir)
    
    print("=" * 70)
    print("Generating more JOIN-variant SQL")
    print("=" * 70)
    print("\nTarget counts:")
    for db_count in sorted(needed.keys()):
        target = TARGET_COUNTS[db_count]
        current = target - needed[db_count]
        print(f"  {db_count} databases: {current} / {target} ({needed[db_count]} more needed)")
    
    total_needed = sum(needed.values())
    if total_needed == 0:
        print("\n✅ All target counts reached; no additional SQL needed")
        return
    
    print(f"\nTotal still needed: {total_needed} SQL entries")
    
    # 2. Load skeleton file
    print("\nLoading SQL skeletons...")
    with open(skeleton_file, 'r', encoding='utf-8') as f:
        all_skeletons = json.load(f)
    
    print(f"  Total skeletons: {len(all_skeletons)}")
    
    # 3. Get used skeleton identifiers
    existing_skeleton_signatures = get_existing_skeletons(sql_dir)
    print(f"  Skeletons already used: {len(existing_skeleton_signatures)}")
    
    # 4. Group skeletons by database count
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
        
        # Filter out used skeletons (table-pair info as unique identifier)
        unused_skeletons = []
        for skeleton in available_skeletons:
            databases = sorted(skeleton.get('databases', []))
            table_db_mapping = skeleton.get('table_database_mapping', {})
            tables = sorted(table_db_mapping.keys())
            
            # Use database and table combination as unique identifier
            if len(databases) >= 2 and len(tables) >= 2:
                table1 = tables[0]
                table2 = tables[1] if len(tables) > 1 else None
                
                if table2:
                    signature = tuple(sorted(databases[:2]) + sorted([table1, table2]))
                    if signature not in existing_skeleton_signatures:
                        unused_skeletons.append(skeleton)
        
        print(f"\n{db_count} databases:")
        print(f"  Available skeletons: {len(available_skeletons)}")
        print(f"  Unused skeletons: {len(unused_skeletons)}")
        print(f"  Need to generate: {needed[db_count]}")
        
        # Select required count (use all if unused skeletons are insufficient)
        import random
        to_generate = min(needed[db_count] * 3, len(unused_skeletons))  # Generate 3x because success rate is ~22.5%
        selected = random.sample(unused_skeletons, to_generate) if len(unused_skeletons) >= to_generate else unused_skeletons
        
        print(f"  Selected for generation: {len(selected)} skeletons")
        selected_skeletons.extend(selected)
    
    if not selected_skeletons:
        print("\n⚠️  No unused skeletons available; cannot generate more SQL")
        return
    
    print(f"\nTotal selected: {len(selected_skeletons)} skeletons for generation")
    
    # 6. Save selected skeletons to a temporary file
    temp_skeleton_file = skeleton_file.replace('.json', '_temp.json')
    with open(temp_skeleton_file, 'w', encoding='utf-8') as f:
        json.dump(selected_skeletons, f, ensure_ascii=False, indent=2)
    
    print(f"\nTemporary skeleton file: {temp_skeleton_file}")
    
    # 7. Call SQL filling script (direct function call, not subprocess, to avoid blocking)
    print("\nStarting SQL generation...")
    print(f"Will generate {len(selected_skeletons)} SQL entries; this may take a while...")
    
    # Import and call directly instead of using subprocess
    import sys
    sys.path.insert(0, os.path.dirname(__file__))
    from cross_db_2fill_sql_placeholders_join import (
        load_multiple_schemas, 
        process_cross_database_skeleton
    )
    from concurrent.futures import ThreadPoolExecutor, as_completed
    from tqdm import tqdm
    
    # Load schemas for all databases
    all_databases = set()
    for skeleton in selected_skeletons:
        all_databases.update(skeleton.get('databases', []))
    
    schemas = load_multiple_schemas(all_databases, database_dir)
    
    # Process each skeleton
    success_count = 0
    failed_count = 0
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = []
        for skeleton in selected_skeletons:
            future = executor.submit(
                process_cross_database_skeleton,
                skeleton, schemas, graph_dir, output_dir,
                database_dir, max_retries
            )
            futures.append(future)
        
        # Collect results
        for future in tqdm(as_completed(futures), total=len(futures), desc="Fill progress"):
            try:
                idx, success, message = future.result()
                if success:
                    success_count += 1
                else:
                    failed_count += 1
            except Exception as e:
                failed_count += 1
                print(f"Processing failed: {e}")
    
    print(f"\nGeneration complete!")
    print(f"Success: {success_count}/{len(selected_skeletons)}")
    print(f"Failed: {failed_count}/{len(selected_skeletons)}")
    
    # 8. Clean up temporary file
    if os.path.exists(temp_skeleton_file):
        os.remove(temp_skeleton_file)
        print(f"\nCleaned up temporary file: {temp_skeleton_file}")

def main():
    parser = argparse.ArgumentParser(description='Generate more JOIN-variant SQL without duplicating existing results')
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
                       help='Maximum number of retries')
    parser.add_argument('--max_workers', type=int, default=5,
                       help='Number of concurrent threads')
    
    args = parser.parse_args()
    
    generate_more_sqls(
        args.skeleton_file,
        args.sql_dir,
        args.graph_dir,
        args.output_dir,
        args.database_dir,
        args.max_workers,
        args.max_retries
    )

if __name__ == '__main__':
    main()
