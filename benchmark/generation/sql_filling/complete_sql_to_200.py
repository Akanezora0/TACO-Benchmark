"""
Complete SQL generation to 200 entries.
Check generated SQL count per database; regenerate missing SQLs if below 200.
"""

import json
import os
import sys
import argparse
import importlib.util
from tqdm import tqdm

# Dynamic import of fill_sql_placeholders module
script_dir = os.path.dirname(os.path.abspath(__file__))
module_path = os.path.join(script_dir, 'fill_sql_placeholders.py')
spec = importlib.util.spec_from_file_location("fill_sql", module_path)
fill_sql_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(fill_sql_module)

# Import required functions
process_single_sql_skeleton = fill_sql_module.process_single_sql_skeleton
load_schema = fill_sql_module.load_schema
extract_schema_info = fill_sql_module.extract_schema_info
load_config = fill_sql_module.load_config

def check_sql_count(database_name, skeleton_file, sql_dir):
    """Check count of generated SQLs"""
    # Load SQL skeletons
    with open(skeleton_file, 'r', encoding='utf-8') as f:
        sql_skeletons = json.load(f)
    
    total_skeletons = len(sql_skeletons)
    
    # Check generated SQL files
    existing_indices = set()
    if os.path.exists(sql_dir):
        for f in os.listdir(sql_dir):
            if f.startswith('generated_sql_') and f.endswith('.json'):
                try:
                    idx = int(f.replace('generated_sql_', '').replace('.json', ''))
                    existing_indices.add(idx)
                except:
                    pass
    
    missing_indices = [i for i in range(total_skeletons) if i not in existing_indices]
    
    return total_skeletons, len(existing_indices), missing_indices

def complete_sql_for_database(database_name, skeleton_file, schema_file, graph_dir, output_dir, max_retries=3, max_workers=5):
    """Complete SQL generation to 200 entries for a single database"""
    # Load schema
    schema = load_schema(schema_file)
    schema_info = extract_schema_info(schema)
    
    # Load SQL skeletons
    with open(skeleton_file, 'r', encoding='utf-8') as f:
        sql_skeletons = json.load(f)
    
    # Create output directory
    single_output_path = os.path.join(output_dir, 'single', database_name)
    os.makedirs(single_output_path, exist_ok=True)
    
    # Check missing SQLs
    total_skeletons, existing_count, missing_indices = check_sql_count(
        database_name, skeleton_file, single_output_path
    )
    
    print(f"\nDatabase: {database_name}")
    print(f"  Total skeletons: {total_skeletons}")
    print(f"  Generated: {existing_count}")
    print(f"  Missing: {len(missing_indices)}")
    
    if len(missing_indices) == 0:
        print(f"  ✅ Complete, no need to fill in")
        return existing_count, 0
    
    # Only process missing indices
    print(f"  Starting to fill in {len(missing_indices)} SQLs...")
    
    from concurrent.futures import ThreadPoolExecutor, as_completed
    
    # Prepare task arguments (only missing indices)
    tasks = []
    for idx in missing_indices:
        sql_skeleton = sql_skeletons[idx]
        tasks.append((
            idx, sql_skeleton, database_name, schema, schema_info, 
            graph_dir, single_output_path, schema_file, max_retries
        ))
    
    success_count = 0
    fail_count = 0
    
    # Process concurrently with thread pool
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_idx = {executor.submit(process_single_sql_skeleton, task): task[0] for task in tasks}
        
        with tqdm(total=len(tasks), desc=f"  Fill progress") as pbar:
            for future in as_completed(future_to_idx):
                idx = future_to_idx[future]
                try:
                    result_idx, success, message = future.result()
                    if success:
                        if message != "Already exists":
                            success_count += 1
                    else:
                        fail_count += 1
                except Exception as e:
                    fail_count += 1
                    print(f"  Exception processing index {idx}: {e}")
                finally:
                    pbar.update(1)
    
    # Check final count again
    _, final_count, _ = check_sql_count(database_name, skeleton_file, single_output_path)
    print(f"  ✅ Fill complete: added {success_count}, failed {fail_count}, final count: {final_count}/{total_skeletons}")
    
    return final_count, fail_count

def main():
    parser = argparse.ArgumentParser(description='Complete SQL generation to 200 entries')
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(script_dir)))
    
    parser.add_argument('--skeleton_dir', type=str, default=None,
                       help='SQL skeleton directory')
    parser.add_argument('--database_dir', type=str, default=None,
                       help='Database directory')
    parser.add_argument('--graph_dir', type=str, default=None,
                       help='Graph directory')
    parser.add_argument('--output_dir', type=str, default=None,
                       help='Output directory')
    parser.add_argument('--max_retries', type=int, default=3,
                       help='Maximum retry times (default: 3)')
    parser.add_argument('--max_workers', type=int, default=5,
                       help='Concurrency (default: 5)')
    parser.add_argument('--config', type=str, default=None,
                       help='Config file path')
    parser.add_argument('--database', type=str, default=None,
                       help='Process only specified database (default: all)')
    
    args = parser.parse_args()
    
    # Load config
    load_config(args.config)
    
    # Set default paths (US dataset)
    if args.skeleton_dir is None:
        args.skeleton_dir = os.path.join(project_root, 'benchmark', 'data', 'us', 'output', 'sql_skeleton')
    if args.database_dir is None:
        args.database_dir = os.path.join(project_root, 'benchmark', 'data', 'us', 'database')
    if args.graph_dir is None:
        args.graph_dir = os.path.join(project_root, 'benchmark', 'data', 'us', 'output', 'graph')
    if args.output_dir is None:
        args.output_dir = os.path.join(project_root, 'benchmark', 'data', 'us', 'output')
    
    # Convert to absolute paths
    args.skeleton_dir = os.path.abspath(args.skeleton_dir)
    args.database_dir = os.path.abspath(args.database_dir)
    args.graph_dir = os.path.abspath(args.graph_dir)
    args.output_dir = os.path.abspath(args.output_dir)
    
    # Get all SQL skeleton files
    skeleton_files = [f for f in os.listdir(args.skeleton_dir) if f.endswith('_sql_skeleton.json')]
    
    if args.database:
        # Process only specified database
        skeleton_files = [f for f in skeleton_files if args.database in f]
    
    print(f"Found {len(skeleton_files)} database SQL skeleton files")
    print(f"Skeleton directory: {args.skeleton_dir}")
    print(f"Database directory: {args.database_dir}")
    print(f"Graph directory: {args.graph_dir}")
    print(f"Output directory: {args.output_dir}")
    print(f"Concurrency: {args.max_workers}, max retries: {args.max_retries}")
    print("="*60)
    
    total_success = 0
    total_fail = 0
    databases_status = []
    
    for skeleton_file in skeleton_files:
        database_name = skeleton_file.replace('_sql_skeleton.json', '')
        
        skeleton_path = os.path.join(args.skeleton_dir, skeleton_file)
        schema_path = os.path.join(args.database_dir, database_name, f"{database_name}.json")
        
        if not os.path.exists(schema_path):
            print(f"\n⚠️  Schema file does not exist for database '{database_name}': {schema_path}")
            continue
        
        final_count, fail_count = complete_sql_for_database(
            database_name, skeleton_path, schema_path,
            args.graph_dir, args.output_dir, args.max_retries, args.max_workers
        )
        
        databases_status.append({
            'database': database_name,
            'final_count': final_count,
            'fail_count': fail_count
        })
        
        total_success += final_count
        total_fail += fail_count
    
    print("\n" + "="*60)
    print("📊 Summary:")
    print("="*60)
    
    # Sort by final count
    databases_status.sort(key=lambda x: x['final_count'])
    
    for status in databases_status:
        db = status['database']
        count = status['final_count']
        fail = status['fail_count']
        if count >= 200:
            print(f"✅ {db}: {count}/200 (failed: {fail})")
        elif count >= 150:
            print(f"⚠️  {db}: {count}/200 (failed: {fail})")
        else:
            print(f"❌ {db}: {count}/200 (failed: {fail})")
    
    print("="*60)
    print(f"Total: success {total_success}, failed {total_fail}")
    print("="*60)

if __name__ == '__main__':
    main()
