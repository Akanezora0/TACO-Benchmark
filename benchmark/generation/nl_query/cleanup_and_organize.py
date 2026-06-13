"""
Clean up and organize NL query files so each database has exactly 200 entries
- Keep files with indices 0-199
- Delete files with indices >= 200
- Ensure 50 regenerated simple queries fall within indices 0-199
"""

import json
import os
import shutil
import argparse

def cleanup_database(nl_dir: str, database: str, dry_run: bool = False):
    """Clean up NL query files for a database"""
    if not os.path.exists(nl_dir):
        print(f"Directory does not exist: {nl_dir}")
        return
    
    all_files = sorted([f for f in os.listdir(nl_dir) if f.startswith('generated_nl_query_') and f.endswith('.json')],
                      key=lambda x: int(x.split('_')[-1].split('.')[0]) if x.split('_')[-1].split('.')[0].isdigit() else 0)
    
    files_to_keep = []
    files_to_remove = []
    regenerated_count = 0
    
    for filename in all_files:
        idx_str = filename.split('_')[-1].split('.')[0]
        if not idx_str.isdigit():
            continue
        
        idx = int(idx_str)
        filepath = os.path.join(nl_dir, filename)
        
        # Check whether this file was regenerated
        is_regenerated = False
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                data = json.load(f)
            is_regenerated = data.get('regenerated', False)
        except:
            pass
        
        if idx < 200:
            files_to_keep.append((idx, filename, is_regenerated))
            if is_regenerated:
                regenerated_count += 1
        else:
            files_to_remove.append((idx, filename))
    
    print(f"{database}:")
    print(f"  Total files: {len(all_files)}")
    print(f"  Files to keep (indices 0-199): {len(files_to_keep)}")
    print(f"  Files to delete (indices >= 200): {len(files_to_remove)}")
    print(f"  Regenerated simple queries: {regenerated_count}")
    
    if files_to_remove:
        print(f"\n  Files to be deleted (first 10):")
        for idx, filename in files_to_remove[:10]:
            print(f"    Index {idx}: {filename}")
        if len(files_to_remove) > 10:
            print(f"    ... and {len(files_to_remove) - 10} more files")
    
    if not dry_run:
        # Delete files with indices >= 200
        removed_count = 0
        for idx, filename in files_to_remove:
            filepath = os.path.join(nl_dir, filename)
            try:
                os.remove(filepath)
                removed_count += 1
            except Exception as e:
                print(f"  Failed to delete {filename}: {e}")
        
        print(f"\n  Deleted {removed_count} files")
    
    print()
    return len(files_to_keep), regenerated_count

def main():
    parser = argparse.ArgumentParser(description='Clean up and organize NL query files')
    parser.add_argument('--nl_query_base_dir', type=str, 
                       default='benchmark/data/beijing/output/nl_query',
                       help='Base directory for NL query files')
    parser.add_argument('--databases', type=str, nargs='+',
                       default=['企业服务', '社会保障', '医疗健康'],
                       help='List of databases to process')
    parser.add_argument('--dry-run', action='store_true',
                       help='Show planned actions only without deleting files')
    
    args = parser.parse_args()
    
    print("="*80)
    if args.dry_run:
        print("Clean up and organize NL query files (preview mode)")
    else:
        print("Clean up and organize NL query files")
    print("="*80)
    print()
    
    total_kept = 0
    total_regenerated = 0
    
    for db in args.databases:
        nl_dir = os.path.join(args.nl_query_base_dir, db)
        kept, regenerated = cleanup_database(nl_dir, db, args.dry_run)
        total_kept += kept
        total_regenerated += regenerated
    
    print("="*80)
    print("Summary:")
    print(f"  Total files kept: {total_kept}")
    print(f"  Total regenerated simple queries: {total_regenerated}")
    print("="*80)
    
    if args.dry_run:
        print("\nThis is preview mode; no files were deleted.")
        print("Run without --dry-run to perform the cleanup.")

if __name__ == '__main__':
    main()
