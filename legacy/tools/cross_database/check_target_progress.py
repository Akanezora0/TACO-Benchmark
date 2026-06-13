#!/usr/bin/env python3
"""
Check target completion for cross-database SQL generation.
Report total completion counts for JOIN and UNION methods compared to targets.
"""

import os
import json
from collections import defaultdict

# Target counts (from README.md)
TARGET_COUNTS = {
    2: 359,  # cross 2 databases
    3: 105,  # cross 3 databases
    4: 2     # cross 4 databases
}

def count_sqls_with_results_in_directory(sql_dir, is_backup=False):
    """Count SQL files with results in directory, by database count"""
    stats = defaultdict(int)
    
    if not os.path.exists(sql_dir):
        return stats
    
    for filename in os.listdir(sql_dir):
        if not filename.startswith('cross_db_generated_sql_') or not filename.endswith('.json'):
            continue
        
        file_path = os.path.join(sql_dir, filename)
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            results = data.get('results', [])
            databases = data.get('databases', [])
            num_databases = len(databases)
            
            # If num_databases is in metadata, prefer that value
            metadata = data.get('metadata', {})
            if 'num_databases' in metadata:
                num_databases = metadata['num_databases']
            
            if num_databases < 2 or num_databases > 4:
                continue
            
            # Determine whether the file has results
            if is_backup:
                # Files in backup directory are assumed to have results
                has_results = True
            else:
                has_results = results is not None and len(results) > 0
            
            if has_results:
                stats[num_databases] += 1
                
        except Exception as e:
            print(f"Warning: Cannot read file {filename}: {e}")
            continue
    
    return stats

def main():
    # Get script directory and project root
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(script_dir)))
    
    # Define directory paths
    base_output_dir = os.path.join(project_root, 'benchmark', 'data', 'beijing', 'output')
    
    # JOIN-related directories
    join_dir = os.path.join(base_output_dir, 'cross_db_single_join')
    join_backup_dir = os.path.join(base_output_dir, 'cross_db_single_join_backup_51')
    
    # UNION-related directories
    union_dir = os.path.join(base_output_dir, 'cross_db_single_union_version')
    
    print("=" * 80)
    print("Cross-database SQL generation target completion statistics")
    print("=" * 80)
    print()
    
    # Statistics for JOIN method (current directory)
    print("📊 Counting JOIN method (current directory)...")
    join_stats = count_sqls_with_results_in_directory(join_dir, is_backup=False)
    
    # Statistics for JOIN method (backup directory)
    print("📊 Counting JOIN method (backup directory)...")
    join_backup_stats = count_sqls_with_results_in_directory(join_backup_dir, is_backup=True)
    
    # Merge JOIN statistics
    join_total_stats = defaultdict(int)
    for db_count in [2, 3, 4]:
        join_total_stats[db_count] = join_stats[db_count] + join_backup_stats[db_count]
    
    # Statistics for UNION method
    print("📊 Counting UNION method...")
    union_stats = count_sqls_with_results_in_directory(union_dir, is_backup=False)
    
    # Merge all methods (JOIN + UNION)
    total_stats = defaultdict(int)
    for db_count in [2, 3, 4]:
        total_stats[db_count] = join_total_stats[db_count] + union_stats[db_count]
    
    # Output results
    print("\n" + "=" * 80)
    print("Target completion status")
    print("=" * 80)
    print()
    
    print(f"{'Database count':<15} {'Target count':<15} {'JOIN completed':<15} {'UNION completed':<15} {'Total completed':<15} {'Completion rate':<15} {'Still needed':<15}")
    print("-" * 80)
    
    total_target = 0
    total_completed = 0
    
    for db_count in [2, 3, 4]:
        target = TARGET_COUNTS[db_count]
        join_completed = join_total_stats[db_count]
        union_completed = union_stats[db_count]
        total_completed_count = total_stats[db_count]
        completion_rate = (total_completed_count / target * 100) if target > 0 else 0
        remaining = max(0, target - total_completed_count)
        
        total_target += target
        total_completed += total_completed_count
        
        print(f"{db_count} databases{'':<6} {target:<15} {join_completed:<15} {union_completed:<15} {total_completed_count:<15} {completion_rate:.1f}%{'':<10} {remaining:<15}")
    
    print("-" * 80)
    total_completion_rate = (total_completed / total_target * 100) if total_target > 0 else 0
    total_remaining = max(0, total_target - total_completed)
    print(f"{'Total':<15} {total_target:<15} {sum(join_total_stats.values()):<15} {sum(union_stats.values()):<15} {total_completed:<15} {total_completion_rate:.1f}%{'':<10} {total_remaining:<15}")
    print()
    
    # Detailed statistics
    print("=" * 80)
    print("Detailed statistics")
    print("=" * 80)
    print()
    
    for db_count in [2, 3, 4]:
        target = TARGET_COUNTS[db_count]
        join_completed = join_total_stats[db_count]
        union_completed = union_stats[db_count]
        total_completed_count = total_stats[db_count]
        completion_rate = (total_completed_count / target * 100) if target > 0 else 0
        remaining = max(0, target - total_completed_count)
        
        print(f"[{db_count} databases]")
        print(f"  Target count: {target}")
        print(f"  JOIN completed: {join_completed} (current directory: {join_stats[db_count]}, backup directory: {join_backup_stats[db_count]})")
        print(f"  UNION completed: {union_completed}")
        print(f"  Total completed: {total_completed_count}")
        print(f"  Completion rate: {completion_rate:.1f}%")
        print(f"  Still needed: {remaining}")
        print()
    
    # Overall statistics
    print("=" * 80)
    print("Overall statistics")
    print("=" * 80)
    print()
    print(f"Total target: {total_target}")
    print(f"JOIN completed: {sum(join_total_stats.values())}")
    print(f"UNION completed: {sum(union_stats.values())}")
    print(f"Total completed: {total_completed}")
    print(f"Overall completion rate: {total_completion_rate:.1f}%")
    print(f"Still needed: {total_remaining}")
    print()
    
    # Output directory paths
    print("=" * 80)
    print("Directory paths")
    print("=" * 80)
    print(f"JOIN current directory: {join_dir}")
    print(f"JOIN backup directory: {join_backup_dir}")
    print(f"UNION directory: {union_dir}")
    print()

if __name__ == '__main__':
    main()
