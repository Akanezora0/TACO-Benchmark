#!/usr/bin/env python3
"""
Detailed statistics for union and join methods in cross-database SQL
Including result SQL in backup directory
Statistics by database count (2, 3, 4) and method (union, join)
"""

import os
import json
from collections import defaultdict

def count_sqls_in_directory(sql_dir, is_backup=False):
    """Count SQL files in directory by database count and whether they have results
    
    Args:
        sql_dir: Directory path
        is_backup: Whether this is backup directory (files assumed to have results)
    """
    stats = {
        'total': 0,
        'with_results': 0,
        'without_results': 0,
        'by_db_count': defaultdict(lambda: {'total': 0, 'with_results': 0, 'without_results': 0})
    }
    
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
            
            stats['total'] += 1
            stats['by_db_count'][num_databases]['total'] += 1
            
            if has_results:
                stats['with_results'] += 1
                stats['by_db_count'][num_databases]['with_results'] += 1
            else:
                stats['without_results'] += 1
                stats['by_db_count'][num_databases]['without_results'] += 1
                
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
    print("Cross-database SQL detailed statistics: UNION vs JOIN (including backup directory)")
    print("=" * 80)
    print()
    
    # Statistics for JOIN method (current directory)
    print("📊 Counting JOIN method (current directory)...")
    join_stats = count_sqls_in_directory(join_dir, is_backup=False)
    
    # Statistics for JOIN method (backup directory)
    print("📊 Counting JOIN method (backup directory)...")
    join_backup_stats = count_sqls_in_directory(join_backup_dir, is_backup=True)
    
    # Merge JOIN statistics
    join_total_stats = {
        'total': join_stats['total'] + join_backup_stats['total'],
        'with_results': join_stats['with_results'] + join_backup_stats['with_results'],
        'without_results': join_stats['without_results'] + join_backup_stats['without_results'],
        'by_db_count': defaultdict(lambda: {'total': 0, 'with_results': 0, 'without_results': 0})
    }
    
    for db_count in [2, 3, 4]:
        join_total_stats['by_db_count'][db_count]['total'] = (
            join_stats['by_db_count'][db_count]['total'] + 
            join_backup_stats['by_db_count'][db_count]['total']
        )
        join_total_stats['by_db_count'][db_count]['with_results'] = (
            join_stats['by_db_count'][db_count]['with_results'] + 
            join_backup_stats['by_db_count'][db_count]['with_results']
        )
        join_total_stats['by_db_count'][db_count]['without_results'] = (
            join_stats['by_db_count'][db_count]['without_results'] + 
            join_backup_stats['by_db_count'][db_count]['without_results']
        )
    
    # Statistics for UNION method
    print("📊 Counting UNION method...")
    union_stats = count_sqls_in_directory(union_dir, is_backup=False)
    
    # Output statistics results
    print("\n" + "=" * 80)
    print("Detailed statistics results")
    print("=" * 80)
    print()
    
    # Output by database count
    for db_count in [2, 3, 4]:
        print(f"[{db_count} databases]")
        print("-" * 80)
        
        # JOIN method statistics
        join_current_total = join_stats['by_db_count'][db_count]['total']
        join_current_with_results = join_stats['by_db_count'][db_count]['with_results']
        join_current_without_results = join_stats['by_db_count'][db_count]['without_results']
        
        join_backup_total = join_backup_stats['by_db_count'][db_count]['total']
        join_backup_with_results = join_backup_stats['by_db_count'][db_count]['with_results']
        
        join_total = join_total_stats['by_db_count'][db_count]['total']
        join_with_results = join_total_stats['by_db_count'][db_count]['with_results']
        join_without_results = join_total_stats['by_db_count'][db_count]['without_results']
        
        print(f"  JOIN method:")
        print(f"    Current directory: total={join_current_total}, with results={join_current_with_results}, without results={join_current_without_results}")
        print(f"    Backup directory: total={join_backup_total}, with results={join_backup_with_results}")
        print(f"    Combined: total={join_total}, with results={join_with_results} ({join_with_results/join_total*100:.1f}%)" if join_total > 0 else "    Combined: total=0")
        print(f"              without results={join_without_results}")
        
        # UNION method statistics
        union_total = union_stats['by_db_count'][db_count]['total']
        union_with_results = union_stats['by_db_count'][db_count]['with_results']
        union_without_results = union_stats['by_db_count'][db_count]['without_results']
        
        print(f"  UNION method:")
        print(f"    Total: {union_total}")
        print(f"    with results: {union_with_results} ({union_with_results/union_total*100:.1f}%)" if union_total > 0 else "    with results: 0")
        print(f"    without results: {union_without_results}")
        
        # Combined statistics (JOIN + UNION)
        total_all = join_total + union_total
        with_results_all = join_with_results + union_with_results
        without_results_all = join_without_results + union_without_results
        
        print(f"  Combined (JOIN + UNION):")
        print(f"    Total: {total_all}")
        print(f"    with results: {with_results_all} ({with_results_all/total_all*100:.1f}%)" if total_all > 0 else "    with results: 0")
        print(f"    without results: {without_results_all}")
        print()
    
    # Overall statistics
    print("=" * 80)
    print("Overall statistics (all database counts merged)")
    print("=" * 80)
    print()
    
    # JOIN overall statistics
    join_current_total_all = join_stats['total']
    join_current_with_results_all = join_stats['with_results']
    join_current_without_results_all = join_stats['without_results']
    
    join_backup_total_all = join_backup_stats['total']
    join_backup_with_results_all = join_backup_stats['with_results']
    
    join_total_all = join_total_stats['total']
    join_with_results_all = join_total_stats['with_results']
    join_without_results_all = join_total_stats['without_results']
    
    print(f"JOIN method:")
    print(f"  Current directory: total={join_current_total_all}, with results={join_current_with_results_all}, without results={join_current_without_results_all}")
    print(f"  Backup directory: total={join_backup_total_all}, with results={join_backup_with_results_all}")
    print(f"  Combined: total={join_total_all}, with results={join_with_results_all} ({join_with_results_all/join_total_all*100:.1f}%)" if join_total_all > 0 else "  Combined: total=0")
    print(f"            without results={join_without_results_all}")
    print()
    
    # UNION overall statistics
    union_total_all = union_stats['total']
    union_with_results_all = union_stats['with_results']
    union_without_results_all = union_stats['without_results']
    
    print(f"UNION method:")
    print(f"  Total: {union_total_all}")
    print(f"  with results: {union_with_results_all} ({union_with_results_all/union_total_all*100:.1f}%)" if union_total_all > 0 else "  with results: 0")
    print(f"  without results: {union_without_results_all}")
    print()
    
    # Final combined statistics (JOIN + UNION)
    total_all = join_total_all + union_total_all
    with_results_all = join_with_results_all + union_with_results_all
    without_results_all = join_without_results_all + union_without_results_all
    
    print(f"Final combined statistics (JOIN + UNION):")
    print(f"  Total: {total_all}")
    print(f"  with results: {with_results_all} ({with_results_all/total_all*100:.1f}%)" if total_all > 0 else "  with results: 0")
    print(f"  without results: {without_results_all}")
    print()
    
    # Output table format (show only with results)
    print("=" * 80)
    print("Statistics table (SQL count with results)")
    print("=" * 80)
    print()
    print(f"{'Method':<15} {'2 databases':<20} {'3 databases':<20} {'4 databases':<20} {'Total':<20}")
    print("-" * 80)
    print(f"{'JOIN (current)':<15} {join_stats['by_db_count'][2]['with_results']:<20} {join_stats['by_db_count'][3]['with_results']:<20} {join_stats['by_db_count'][4]['with_results']:<20} {join_current_with_results_all:<20}")
    print(f"{'JOIN (backup)':<15} {join_backup_stats['by_db_count'][2]['with_results']:<20} {join_backup_stats['by_db_count'][3]['with_results']:<20} {join_backup_stats['by_db_count'][4]['with_results']:<20} {join_backup_with_results_all:<20}")
    print(f"{'JOIN (total)':<15} {join_total_stats['by_db_count'][2]['with_results']:<20} {join_total_stats['by_db_count'][3]['with_results']:<20} {join_total_stats['by_db_count'][4]['with_results']:<20} {join_with_results_all:<20}")
    print(f"{'UNION':<15} {union_stats['by_db_count'][2]['with_results']:<20} {union_stats['by_db_count'][3]['with_results']:<20} {union_stats['by_db_count'][4]['with_results']:<20} {union_with_results_all:<20}")
    print(f"{'Total':<15} {join_total_stats['by_db_count'][2]['with_results'] + union_stats['by_db_count'][2]['with_results']:<20} {join_total_stats['by_db_count'][3]['with_results'] + union_stats['by_db_count'][3]['with_results']:<20} {join_total_stats['by_db_count'][4]['with_results'] + union_stats['by_db_count'][4]['with_results']:<20} {with_results_all:<20}")
    print()
    
    # Output directory path info
    print("=" * 80)
    print("Directory paths")
    print("=" * 80)
    print(f"JOIN current directory: {join_dir}")
    print(f"JOIN backup directory: {join_backup_dir}")
    print(f"UNION directory: {union_dir}")
    print()

if __name__ == '__main__':
    main()
