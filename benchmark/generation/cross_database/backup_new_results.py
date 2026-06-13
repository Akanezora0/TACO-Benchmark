#!/usr/bin/env python3
"""
Back up newly generated SQL with results to backup directory, continuing numbering
"""

import os
import json
import shutil
import argparse

def get_next_index(backup_dir):
    """Get next available index in backup directory (max continuous index + 1)"""
    indices = []
    if os.path.exists(backup_dir):
        for f in os.listdir(backup_dir):
            if f.startswith('cross_db_generated_sql_') and f.endswith('.json'):
                try:
                    idx = int(f.replace('cross_db_generated_sql_', '').replace('.json', ''))
                    indices.append(idx)
                except:
                    pass
    
    if not indices:
        # If backup directory empty or missing, start from 0
        return 0
    
    # Find max continuous index (starting from 0)
    indices_sorted = sorted(indices)
    max_continuous = -1
    
    # Check consecutive numbering starting from 0
    expected = 0
    for idx in indices_sorted:
        if idx == expected:
            max_continuous = idx
            expected += 1
        elif idx > expected:
            # Gap found, stop
            break
    
    # Return max continuous index + 1
    return max_continuous + 1

def backup_new_results(sql_dir, backup_dir):
    """Back up newly generated SQL files with results"""
    
    # Create backup directory
    os.makedirs(backup_dir, exist_ok=True)
    
    # Get next index
    next_index = get_next_index(backup_dir)
    print(f"Next index: {next_index}")
    
    # Collect all SQL files with results
    valid_files = []
    for f in os.listdir(sql_dir):
        if f.startswith('cross_db_generated_sql_') and f.endswith('.json'):
            file_path = os.path.join(sql_dir, f)
            try:
                with open(file_path, 'r', encoding='utf-8') as file:
                    data = json.load(file)
                    results = data.get('results', [])
                    # Files with results
                    if results is not None and len(results) > 0:
                        valid_files.append((file_path, f))
            except:
                pass
    
    print(f"Found {len(valid_files)} SQL files with results")
    
    # Sort by filename
    valid_files.sort(key=lambda x: x[1])
    
    # Back up and rename
    backed_up = 0
    for idx, (file_path, original_name) in enumerate(valid_files):
        new_name = f"cross_db_generated_sql_{next_index + idx}.json"
        backup_path = os.path.join(backup_dir, new_name)
        
        # Skip if already exists (avoid duplicate backup)
        if not os.path.exists(backup_path):
            shutil.copy2(file_path, backup_path)
            backed_up += 1
            print(f"  Backup: {original_name} -> {new_name}")
    
    return backed_up

def main():
    # Get script directory and project root
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(script_dir)))
    
    parser = argparse.ArgumentParser(description='Back up newly generated SQL with results')
    parser.add_argument('--sql_dir', type=str,
                       default=os.path.join(project_root, 'benchmark', 'data', 'beijing', 'output', 'cross_db_single_join'),
                       help='SQL file directory')
    parser.add_argument('--backup_dir', type=str,
                       default=os.path.join(project_root, 'benchmark', 'data', 'beijing', 'output', 'cross_db_single_join_backup_51'),
                       help='Backup directory')
    
    args = parser.parse_args()
    
    # Convert to absolute paths (if user provided relative paths)
    if not os.path.isabs(args.sql_dir):
        args.sql_dir = os.path.join(project_root, args.sql_dir) if not os.path.isabs(args.sql_dir) else args.sql_dir
    if not os.path.isabs(args.backup_dir):
        args.backup_dir = os.path.join(project_root, args.backup_dir) if not os.path.isabs(args.backup_dir) else args.backup_dir
    
    print("=" * 70)
    print("Back up newly generated SQL with results")
    print("=" * 70)
    
    count = backup_new_results(args.sql_dir, args.backup_dir)
    print(f"\n✅ Backed up {count} new files to {args.backup_dir}")
    
    print("\n" + "=" * 70)
    print("Done!")
    print("=" * 70)

if __name__ == '__main__':
    main()

