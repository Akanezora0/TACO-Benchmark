#!/usr/bin/env python3
"""
Remove incorrectly numbered files from the backup directory (indices greater than 50 and less than 342).
These were from a previous incorrect backup and should be deleted.
"""

import os
import argparse

def cleanup_wrong_backup(backup_dir, max_correct_index=50):
    """Remove incorrectly numbered files (delete all indices greater than max_correct_index; numbering should continue from max_correct_index+1)."""
    
    if not os.path.exists(backup_dir):
        print(f"Backup directory does not exist: {backup_dir}")
        return 0
    
    wrong_files = []
    for f in os.listdir(backup_dir):
        if f.startswith('cross_db_generated_sql_') and f.endswith('.json'):
            try:
                idx = int(f.replace('cross_db_generated_sql_', '').replace('.json', ''))
                # Files with index greater than max_correct_index are incorrect backups (should be consecutive from max_correct_index+1)
                if idx > max_correct_index:
                    wrong_files.append((f, idx))
            except:
                pass
    
    if not wrong_files:
        print("No files found that need cleanup")
        return 0
    
    wrong_files.sort(key=lambda x: x[1])
    print(f"Found {len(wrong_files)} incorrectly numbered files (indices between {max_correct_index+1} and 341)")
    print(f"These files will be deleted...")
    
    deleted = 0
    for f, idx in wrong_files:
        file_path = os.path.join(backup_dir, f)
        try:
            os.remove(file_path)
            deleted += 1
            print(f"  Deleted: {f} (index: {idx})")
        except Exception as e:
            print(f"  Failed to delete: {f} - {e}")
    
    return deleted

def main():
    parser = argparse.ArgumentParser(description='Remove incorrectly numbered files from the backup directory')
    parser.add_argument('--backup_dir', type=str,
                       default='benchmark/data/beijing/output/cross_db_single_join_backup_51',
                       help='Backup directory')
    parser.add_argument('--max_correct_index', type=int, default=50,
                       help='Maximum correct index (default: 50, i.e. 0-50 are correct)')
    
    args = parser.parse_args()
    
    print("=" * 70)
    print("Cleaning up incorrectly numbered files in backup directory")
    print("=" * 70)
    print(f"Backup directory: {args.backup_dir}")
    print(f"Maximum correct index: {args.max_correct_index}")
    print()
    
    deleted = cleanup_wrong_backup(args.backup_dir, args.max_correct_index)
    
    print(f"\n✅ Deleted {deleted} incorrectly numbered files")
    print("\n" + "=" * 70)
    print("Done!")
    print("=" * 70)

if __name__ == '__main__':
    main()
