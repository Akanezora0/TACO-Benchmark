#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Batch generate single-database SQLs for all databases in the US dataset.

One-click execution for all databases with command-line parameter control.
"""

import os
import sys
import json
import argparse
import subprocess
from pathlib import Path
from datetime import datetime
from collections import defaultdict

# Project root directory
PROJECT_ROOT = Path(__file__).parent.parent.parent.parent

# Default paths
DEFAULT_DATABASE_DIR = PROJECT_ROOT / "benchmark" / "data" / "us" / "database"
DEFAULT_SKELETON_DIR = PROJECT_ROOT / "benchmark" / "data" / "us" / "output" / "sql_skeleton"
DEFAULT_GRAPH_DIR = PROJECT_ROOT / "benchmark" / "data" / "us" / "output" / "graph"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "benchmark" / "data" / "us" / "output"
DEFAULT_TARGET_COUNT = 220  # Target 220 per database

def get_all_databases(skeleton_dir):
    """Get list of all databases"""
    databases = []
    if not skeleton_dir.exists():
        return databases
    
    for skeleton_file in skeleton_dir.glob("*_sql_skeleton.json"):
        db_name = skeleton_file.name.replace("_sql_skeleton.json", "")
        databases.append(db_name)
    
    return sorted(databases)

def count_existing_sqls(output_dir, database_name):
    """Count currently existing SQLs"""
    sql_dir = output_dir / "single" / database_name
    if not sql_dir.exists():
        return 0
    
    count = 0
    for f in sql_dir.glob("generated_sql_*.json"):
        try:
            # Check if file is valid (has results field)
            with open(f, 'r', encoding='utf-8') as file:
                data = json.load(file)
                if 'results' in data and data['results'] is not None:
                    count += 1
        except:
            pass
    
    return count

def get_database_status(database_name, skeleton_dir, output_dir, target_count):
    """Get generation status for a database"""
    skeleton_file = skeleton_dir / f"{database_name}_sql_skeleton.json"
    current_count = count_existing_sqls(output_dir, database_name)
    need_count = max(0, target_count - current_count)
    
    return {
        'database': database_name,
        'skeleton_exists': skeleton_file.exists(),
        'current_count': current_count,
        'target_count': target_count,
        'need_count': need_count,
        'completed': need_count == 0
    }

def print_status_table(databases_status):
    """Print status table"""
    print("=" * 100)
    print(f"{'Database Name':<50} {'Current':<10} {'Target':<10} {'Need':<10} {'Status':<10}")
    print("=" * 100)
    
    total_current = 0
    total_target = 0
    total_need = 0
    completed_count = 0
    
    for status in databases_status:
        db_name = status['database']
        current = status['current_count']
        target = status['target_count']
        need = status['need_count']
        completed = status['completed']
        
        # Truncate long database names
        display_name = db_name[:47] + "..." if len(db_name) > 50 else db_name
        
        status_str = "✅ Done" if completed else "⏳ In progress"
        
        print(f"{display_name:<50} {current:<10} {target:<10} {need:<10} {status_str:<10}")
        
        total_current += current
        total_target += target
        total_need += need
        if completed:
            completed_count += 1
    
    print("=" * 100)
    print(f"{'Total':<50} {total_current:<10} {total_target:<10} {total_need:<10} {completed_count}/{len(databases_status)} done")
    print("=" * 100)

def generate_sql_for_database(database_name, script_path, database_dir, skeleton_dir, 
                             graph_dir, output_dir, target_count, max_retries, 
                             background=False, log_dir=None, max_workers=None):
    """Generate SQL for a single database"""
    cmd = [
        sys.executable,
        str(script_path),
        "--database_name", database_name,
        "--database_dir", str(database_dir),
        "--skeleton_dir", str(skeleton_dir),
        "--graph_dir", str(graph_dir),
        "--output_dir", str(output_dir),
        "--target_count", str(target_count),
        "--max_retries", str(max_retries)
    ]
    
    # Add max_workers to command if specified
    if max_workers:
        cmd.extend(["--max_workers", str(max_workers)])
    
    if background:
        if log_dir is None:
            log_dir = PROJECT_ROOT / "benchmark" / "generation" / "sql_filling" / "logs"
        log_dir.mkdir(parents=True, exist_ok=True)
        
        log_file = log_dir / f"generate_us_sql_{database_name.replace(' ', '_').replace('-', '_').replace('/', '_')}.log"
        
        with open(log_file, 'w', encoding='utf-8') as f:
            process = subprocess.Popen(
                cmd,
                stdout=f,
                stderr=subprocess.STDOUT,
                cwd=str(script_path.parent)
            )
        
        return process, log_file
    else:
        try:
            process = subprocess.run(
                cmd,
                cwd=str(script_path.parent),
                timeout=None  # No timeout; let process complete naturally
            )
            return process, None
        except subprocess.TimeoutExpired:
            # Timeout handling
            print(f"Warning: {database_name} generation timed out")
            process = subprocess.CompletedProcess(cmd, -1, None, None)
            return process, None
        except KeyboardInterrupt:
            # User interrupt
            print(f"\nWarning: {database_name} generation interrupted by user")
            process = subprocess.CompletedProcess(cmd, -15, None, None)
            return process, None
        except Exception as e:
            # Other exceptions
            print(f"Error: Exception during {database_name} generation: {e}")
            process = subprocess.CompletedProcess(cmd, -1, None, None)
            return process, None

def main():
    parser = argparse.ArgumentParser(
        description='Batch generate single-database SQLs for all databases in the US dataset',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Show status for all databases
  python3 generate_all_us_databases.py --status
  
  # Generate all databases that need generation (foreground)
  python3 generate_all_us_databases.py
  
  # Generate all databases that need generation (background)
  python3 generate_all_us_databases.py --background
  
  # Generate only specified databases
  python3 generate_all_us_databases.py --databases "City of Austin - 1586" "City of Chicago - 854"
  
  # Skip completed databases
  python3 generate_all_us_databases.py --skip-completed
  
  # Set custom target count
  python3 generate_all_us_databases.py --target-count 250
        """
    )
    
    parser.add_argument('--database-dir', type=str, default=None,
                       help=f'Database directory (default: {DEFAULT_DATABASE_DIR})')
    parser.add_argument('--skeleton-dir', type=str, default=None,
                       help=f'SQL skeleton directory (default: {DEFAULT_SKELETON_DIR})')
    parser.add_argument('--graph-dir', type=str, default=None,
                       help=f'Graph file directory (default: {DEFAULT_GRAPH_DIR})')
    parser.add_argument('--output-dir', type=str, default=None,
                       help=f'Output directory (default: {DEFAULT_OUTPUT_DIR})')
    parser.add_argument('--target-count', type=int, default=DEFAULT_TARGET_COUNT,
                       help=f'Target count per database (default: {DEFAULT_TARGET_COUNT})')
    parser.add_argument('--max-retries', type=int, default=3,
                       help='Maximum retry times (default: 3)')
    parser.add_argument('--databases', type=str, nargs='+', default=None,
                       help='List of databases to generate (default: all databases)')
    parser.add_argument('--skip-completed', action='store_true',
                       help='Skip completed databases')
    parser.add_argument('--background', action='store_true',
                       help='Run in background (each database generates in background)')
    parser.add_argument('--status', action='store_true',
                       help='Show status only, do not generate')
    parser.add_argument('--log-dir', type=str, default=None,
                       help='Log directory (for background runs, default: benchmark/generation/sql_filling/logs)')
    parser.add_argument('--max-workers', type=int, default=None,
                       help='Max concurrent workers per database (default: from config.yaml or 20)')
    parser.add_argument('--retry-failed', action='store_true',
                       help='Retry previously failed databases')
    parser.add_argument('--max-retry-attempts', type=int, default=2,
                       help='Maximum retry attempts per database (default: 2)')
    
    args = parser.parse_args()
    
    # Set paths
    database_dir = Path(args.database_dir) if args.database_dir else DEFAULT_DATABASE_DIR
    skeleton_dir = Path(args.skeleton_dir) if args.skeleton_dir else DEFAULT_SKELETON_DIR
    graph_dir = Path(args.graph_dir) if args.graph_dir else DEFAULT_GRAPH_DIR
    output_dir = Path(args.output_dir) if args.output_dir else DEFAULT_OUTPUT_DIR
    script_path = Path(__file__).parent / "generate_us_single_db_sqls.py"
    
    # Check if script exists
    if not script_path.exists():
        print(f"Error: Generation script does not exist: {script_path}")
        return 1
    
    # Get all databases
    all_databases = get_all_databases(skeleton_dir)
    if not all_databases:
        print(f"Error: No database skeleton files found in {skeleton_dir}")
        return 1
    
    # If database list specified, process only those
    if args.databases:
        databases_to_process = [db for db in args.databases if db in all_databases]
        if not databases_to_process:
            print(f"Error: None of the specified databases exist")
            print(f"Available databases: {', '.join(all_databases[:5])}...")
            return 1
    else:
        databases_to_process = all_databases
    
    # Get status
    databases_status = []
    for db_name in databases_to_process:
        status = get_database_status(db_name, skeleton_dir, output_dir, args.target_count)
        databases_status.append(status)
    
    # Print status
    print_status_table(databases_status)
    
    # If status only, return
    if args.status:
        return 0
    
    # Filter databases that need generation
    databases_to_generate = []
    for status in databases_status:
        if args.skip_completed and status['completed']:
            continue
        if not status['skeleton_exists']:
            print(f"Warning: Skeleton file does not exist for {status['database']}, skipping")
            continue
        if status['need_count'] > 0:
            databases_to_generate.append(status)
    
    if not databases_to_generate:
        print("\nAll databases are complete, no generation needed")
        return 0
    
    print(f"\nDatabases to generate: {len(databases_to_generate)}")
    print(f"Target count: {args.target_count} per database")
    print(f"Maximum retries: {args.max_retries}")
    print(f"Run mode: {'Background' if args.background else 'Foreground'}")
    
    if not args.background:
        # Foreground: process one by one
        confirm = input("\nStart generation? (y/n): ")
        if confirm.lower() != 'y':
            print("Cancelled")
            return 0
        
        print("\nStarting generation...\n")
        
        success_count = 0
        fail_count = 0
        
        failed_databases = []  # Track failed databases for retry
        
        for i, status in enumerate(databases_to_generate, 1):
            db_name = status['database']
            print(f"\n[{i}/{len(databases_to_generate)}] Processing database: {db_name}")
            print(f"Current: {status['current_count']}, Target: {status['target_count']}, Need: {status['need_count']}")
            
            # Retry logic
            retry_count = 0
            max_retry_attempts = args.max_retry_attempts
            success = False
            
            while retry_count <= max_retry_attempts and not success:
                if retry_count > 0:
                    print(f"  Retry {retry_count}/{max_retry_attempts}...")
                    # Wait before retry to avoid resource contention
                    import time
                    time.sleep(5)
                
                try:
                    process, _ = generate_sql_for_database(
                        db_name, script_path, database_dir, skeleton_dir,
                        graph_dir, output_dir, args.target_count, args.max_retries,
                        background=False, max_workers=args.max_workers
                    )
                    
                    if process.returncode == 0:
                        success = True
                        success_count += 1
                        print(f"✅ {db_name} generation complete")
                    else:
                        # Check return code
                        if process.returncode == -15:
                            print(f"⚠️  {db_name} process terminated (possibly resource exhaustion or timeout)")
                        elif process.returncode < 0:
                            print(f"⚠️  {db_name} process exited abnormally (signal: {abs(process.returncode)})")
                        else:
                            print(f"⚠️  {db_name} generation failed (return code: {process.returncode})")
                        
                        # Check if target count reached (may have generated some even if process failed)
                        current_count = count_existing_sqls(output_dir, db_name)
                        if current_count >= args.target_count:
                            print(f"   But target count reached ({current_count}/{args.target_count}), treating as success")
                            success = True
                            success_count += 1
                        elif retry_count < max_retry_attempts:
                            retry_count += 1
                            continue
                        else:
                            failed_databases.append(db_name)
                            fail_count += 1
                            print(f"❌ {db_name} generation failed after {max_retry_attempts} retries")
                
                except KeyboardInterrupt:
                    print(f"\n⚠️  User interrupted, {db_name} generation cancelled")
                    failed_databases.append(db_name)
                    break
                except Exception as e:
                    print(f"❌ Exception during {db_name} generation: {e}")
                    if retry_count < max_retry_attempts:
                        retry_count += 1
                        continue
                    else:
                        failed_databases.append(db_name)
                        fail_count += 1
        
        print("\n" + "=" * 100)
        print("Generation complete")
        print(f"Success: {success_count}")
        print(f"Failed: {fail_count}")
        if failed_databases:
            print(f"\nFailed databases: {', '.join(failed_databases)}")
            print("Tip: Use --databases to retry failed databases individually")
        print("=" * 100)
        
    else:
        # Background: start all tasks concurrently
        confirm = input("\nStart background generation? (y/n): ")
        if confirm.lower() != 'y':
            print("Cancelled")
            return 0
        
        print("\nStarting background tasks...\n")
        
        processes = []
        log_files = []
        
        for status in databases_to_generate:
            db_name = status['database']
            print(f"Starting: {db_name} (need: {status['need_count']})")
            
            process, log_file = generate_sql_for_database(
                db_name, script_path, database_dir, skeleton_dir,
                graph_dir, output_dir, args.target_count, args.max_retries,
                background=True, log_dir=args.log_dir, max_workers=args.max_workers
            )
            
            processes.append((db_name, process))
            if log_file:
                log_files.append((db_name, log_file))
                print(f"  Process ID: {process.pid}")
                print(f"  Log file: {log_file}")
            print()
        
        print("=" * 100)
        print(f"Started {len(processes)} background tasks")
        print("=" * 100)
        print("\nCheck task status:")
        print("  ps aux | grep 'generate_us_single_db_sqls.py' | grep -v grep")
        print("\nView logs:")
        for db_name, log_file in log_files[:5]:  # Show first 5 only
            print(f"  tail -f {log_file}")
        if len(log_files) > 5:
            print(f"  ... and {len(log_files) - 5} more log files")
        print()
    
    return 0

if __name__ == '__main__':
    sys.exit(main())
