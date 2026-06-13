#!/usr/bin/env python3
"""
Single-database end-to-end pipeline script
Pipeline: SQL skeleton generation -> graph generation -> SQL filling -> NL query generation

Strategy:
- Generate 3x the target number of SQL skeletons
- Fill SQL until the target count is reached, then stop
- Generate all NL queries in one pass
"""

import os
import sys
import json
import subprocess
import shutil
from pathlib import Path
from typing import List, Dict
import argparse

# Project root
PROJECT_ROOT = Path(__file__).parent.parent.parent

# Add module paths
sys.path.insert(0, str(PROJECT_ROOT / "benchmark/generation/sql_skeleton_generation"))
sys.path.insert(0, str(PROJECT_ROOT / "benchmark/generation/sql_filling"))
sys.path.insert(0, str(PROJECT_ROOT / "benchmark/generation/nl_query"))

def run_command(cmd: List[str], cwd: str = None, check: bool = True) -> bool:
    """Run a command"""
    try:
        result = subprocess.run(
            cmd,
            cwd=cwd or PROJECT_ROOT,
            check=check,
            capture_output=True,
            text=True
        )
        if result.returncode != 0:
            print(f"Command failed: {' '.join(cmd)}")
            print(f"Error output: {result.stderr}")
            return False
        return True
    except Exception as e:
        print(f"Error running command: {e}")
        return False

def count_successful_sqls(sql_dir: str, database: str) -> int:
    """Count successfully generated SQL statements"""
    db_dir = os.path.join(sql_dir, database)
    if not os.path.exists(db_dir):
        return 0
    
    count = 0
    for file in os.listdir(db_dir):
        if file.startswith('generated_sql_') and file.endswith('.json') and '_error' not in file:
            file_path = os.path.join(db_dir, file)
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    # Check for valid SQL and results
                    if data.get('sql') and data.get('results') is not None:
                        count += 1
            except:
                pass
    return count

def step1_generate_skeletons(database: str, target_count: int, output_dir: str, 
                             database_dir: str, expert_file: str, old_cfg_file: str = None,
                             old_data_file: str = None, new_logs_file: str = None) -> bool:
    """Step 1: generate SQL skeletons (4x count for diversity)"""
    print(f"\n{'='*80}")
    print(f"Step 1: Generate SQL skeletons - {database}")
    print(f"{'='*80}")
    
    # Check current generated SQL count
    current_sql_count = count_successful_sqls(os.path.join(output_dir, "single"), database)
    need_sql = max(0, target_count - current_sql_count)
    
    if need_sql == 0:
        print(f"✅ Already have {current_sql_count} SQL statements, target {target_count} reached; skipping skeleton generation")
        return True
    
    # Compute skeleton count (4x strategy to ensure enough candidates)
    skeleton_count = need_sql * 4
    # For diversity, structure count should be high enough but within a reasonable range
    # Structure count should be 1.2-1.5x skeleton count, capped at 5000
    structure_count = min(int(skeleton_count * 1.2), 5000)
    # Ensure at least skeleton count
    structure_count = max(structure_count, skeleton_count)
    
    print(f"Target SQL count: {target_count}")
    print(f"Current SQL count: {current_sql_count}")
    print(f"SQL still needed: {need_sql}")
    print(f"SQL skeleton count to generate: {skeleton_count} (4x strategy)")
    print(f"SQL structure count to generate: {structure_count} (diversity, max 5000)")
    
    try:
        # Import SQL skeleton generation module
        from generate_for_databases_improved import (
            generate_cfg_for_database,
            generate_structures_for_database,
            generate_skeletons_for_database
        )
        
        # Create output directories
        cfg_dir = os.path.join(output_dir, 'ast_cfg')
        structure_dir = os.path.join(output_dir, 'sql_structure')
        skeleton_dir = os.path.join(output_dir, 'sql_skeleton')
        
        os.makedirs(cfg_dir, exist_ok=True)
        os.makedirs(structure_dir, exist_ok=True)
        os.makedirs(skeleton_dir, exist_ok=True)
        
        # Step 1.1: generate CFG
        cfg_file = os.path.join(cfg_dir, f"{database}_ast_cfg.json")
        print(f"  Generating CFG file...")
        try:
            count = generate_cfg_for_database(expert_file, old_cfg_file, cfg_file, database)
            print(f"  ✓ CFG generated: {count} entries")
        except Exception as e:
            print(f"  ✗ CFG generation failed: {e}")
            return False
        
        # Step 1.2: generate SQL structures (for diversity)
        structure_file = os.path.join(structure_dir, f"{database}_structure.json")
        print(f"  Generating SQL structures (count: {structure_count}, for diversity)...")
        try:
            structures = generate_structures_for_database(cfg_file, old_cfg_file, structure_file, database, structure_count)
            print(f"  ✓ SQL structures generated: {len(structures)}")
        except Exception as e:
            print(f"  ✗ SQL structure generation failed: {e}")
            return False
        
        # Step 1.3: generate SQL skeletons (for diversity, using more structures)
        skeleton_file = os.path.join(skeleton_dir, f"{database}_sql_skeleton.json")
        print(f"  Generating SQL skeletons (target: {skeleton_count}, for diversity)...")
        try:
            # Use a more balanced difficulty mix for diversity
            count = generate_skeletons_for_database(
                structure_file, skeleton_file, old_data_file, new_logs_file, 
                skeleton_count, database, 0.4, 0.4, 0.2  # adjust ratios to add medium/complex queries
            )
            print(f"  ✓ SQL skeletons generated: {count}")
        except Exception as e:
            print(f"  ✗ SQL skeleton generation failed: {e}")
            return False
        
        # Verify output
        if os.path.exists(skeleton_file):
            with open(skeleton_file, 'r', encoding='utf-8') as f:
                skeletons = json.load(f)
            print(f"✅ SQL skeleton generation succeeded: {len(skeletons)} entries")
            return True
        else:
            print(f"❌ SQL skeleton file does not exist: {skeleton_file}")
            return False
            
    except ImportError as e:
        print(f"❌ Failed to import module: {e}")
        print("Trying command-line fallback...")
        # Fallback to command-line execution
        script_path = PROJECT_ROOT / "benchmark/generation/sql_skeleton_generation/generate_for_databases_improved.py"
        cmd = [
            "python3", str(script_path),
            "--total_skeletons", str(skeleton_count),
            "--num_samples", str(structure_count),
            "--simple_ratio", "0.4",
            "--medium_ratio", "0.4",
            "--complex_ratio", "0.2",
        ]
        # Note: command-line mode generates for all databases; this is fallback only
        if not run_command(cmd):
            return False
        
        skeleton_dir = os.path.join(output_dir, "sql_skeleton")
        skeleton_file = os.path.join(skeleton_dir, f"{database}_sql_skeleton.json")
        
        if os.path.exists(skeleton_file):
            with open(skeleton_file, 'r', encoding='utf-8') as f:
                skeletons = json.load(f)
            print(f"✅ SQL skeleton generation succeeded: {len(skeletons)} entries")
            return True
        else:
            print(f"❌ SQL skeleton file does not exist: {skeleton_file}")
            return False

def step2_build_graphs(database: str, output_dir: str, schema_dir: str) -> bool:
    """Step 2: generate schema linking graphs"""
    print(f"\n{'='*80}")
    print(f"Step 2: Generate schema linking graphs - {database}")
    print(f"{'='*80}")
    
    skeleton_dir = os.path.join(output_dir, "sql_skeleton")
    skeleton_file = os.path.join(skeleton_dir, f"{database}_sql_skeleton.json")
    graph_dir = os.path.join(output_dir, "graph")
    schema_file = os.path.join(schema_dir, database, f"{database}.json")
    
    if not os.path.exists(skeleton_file):
        print(f"❌ SQL skeleton file does not exist: {skeleton_file}")
        return False
    
    if not os.path.exists(schema_file):
        print(f"❌ Schema file does not exist: {schema_file}")
        return False
    
    try:
        # Import graph generation module
        from importlib import import_module
        graph_module = import_module('build_schema_graphs')
        process_database = graph_module.process_database
        
        # Call processing function
        process_database(database, skeleton_file, schema_file, graph_dir)
        
        # Verify output (graph files are .graphml and saved in subdirectories)
        db_graph_dir = os.path.join(graph_dir, database)
        if os.path.exists(db_graph_dir):
            graph_files = [f for f in os.listdir(db_graph_dir) if f.endswith('.graphml')]
            print(f"✅ Graph generation succeeded: {len(graph_files)} files")
            return True
        else:
            print(f"❌ Graph directory does not exist: {db_graph_dir}")
            return False
            
    except Exception as e:
        print(f"❌ Graph generation failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def step3_fill_sqls(database: str, target_count: int, output_dir: str, schema_dir: str) -> bool:
    """Step 3: fill SQL until target count is reached"""
    print(f"\n{'='*80}")
    print(f"Step 3: Fill SQL - {database}")
    print(f"{'='*80}")
    
    skeleton_dir = os.path.join(output_dir, "sql_skeleton")
    skeleton_file = os.path.join(skeleton_dir, f"{database}_sql_skeleton.json")
    graph_dir = os.path.join(output_dir, "graph")
    schema_file = os.path.join(schema_dir, database, f"{database}.json")
    
    if not os.path.exists(skeleton_file):
        print(f"❌ SQL skeleton file does not exist: {skeleton_file}")
        return False
    
    if not os.path.exists(schema_file):
        print(f"❌ Schema file does not exist: {schema_file}")
        return False
    
    # Check current generated SQL count
    current_count = count_successful_sqls(os.path.join(output_dir, "single"), database)
    print(f"Current generated SQL count: {current_count}")
    
    if current_count >= target_count:
        print(f"✅ Target count reached; skipping fill step")
        return True
    
    needed = target_count - current_count
    print(f"Need to generate: {needed}")
    
    try:
        # Import SQL filling module
        from importlib import import_module
        fill_module = import_module('fill_sql_placeholders')
        process_single_sql_skeleton = fill_module.process_single_sql_skeleton
        load_schema = fill_module.load_schema
        extract_schema_info = fill_module.extract_schema_info
        
        # Load schema
        schema = load_schema(schema_file)
        schema_info = extract_schema_info(schema)
        
        # Load SQL skeletons; process first needed*4 entries (account for success rate)
        with open(skeleton_file, 'r', encoding='utf-8') as f:
            all_skeletons = json.load(f)
        
        skeletons_to_process = all_skeletons[:needed * 4]  # process 4x count to improve success rate
        print(f"Processing {len(skeletons_to_process)} SQL skeletons (target: {needed} successful SQL)")
        
        # Create output directory
        single_output_path = os.path.join(output_dir, 'single', database)
        os.makedirs(single_output_path, exist_ok=True)
        
        # Process one by one and stop when target count is reached
        from concurrent.futures import ThreadPoolExecutor, as_completed
        from tqdm import tqdm
        
        success_count = 0
        fail_count = 0
        max_workers = 5
        max_retries = 3
        
        # Prepare tasks
        tasks = []
        for idx, sql_skeleton in enumerate(skeletons_to_process):
            tasks.append((
                idx, sql_skeleton, database, schema, schema_info,
                graph_dir, single_output_path, schema_file, max_retries
            ))
        
        # Process concurrently while monitoring success count
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_idx = {executor.submit(process_single_sql_skeleton, task): task[0] for task in tasks}
            
            with tqdm(total=len(tasks), desc=f"{database} fill progress") as pbar:
                for future in as_completed(future_to_idx):
                    idx = future_to_idx[future]
                    try:
                        result_idx, success, message = future.result()
                        if success:
                            if message != "已存在":
                                success_count += 1
                                # Check whether target is reached
                                current = count_successful_sqls(os.path.join(output_dir, "single"), database)
                                if current >= target_count:
                                    print(f"\n✅ Target count {target_count} reached; stopping")
                                    # Cancel remaining tasks
                                    for f in future_to_idx:
                                        f.cancel()
                                    break
                        else:
                            fail_count += 1
                    except Exception as e:
                        fail_count += 1
                        print(f"Exception while processing index {idx}: {e}")
                    finally:
                        pbar.update(1)
        
        # Check final result
        final_count = count_successful_sqls(os.path.join(output_dir, "single"), database)
        print(f"SQL count after fill: {final_count} (success this run: {success_count}, failed: {fail_count})")
        
        if final_count >= target_count:
            print(f"✅ Target count reached")
            return True
        else:
            print(f"⚠️  Target not fully reached; current: {final_count}, target: {target_count}")
            return final_count > 0  # at least some successes
            
    except Exception as e:
        print(f"❌ SQL fill failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def step4_generate_nl_queries(database: str, target_count: int, output_dir: str, schema_dir: str) -> bool:
    """Step 4: generate NL queries"""
    print(f"\n{'='*80}")
    print(f"Step 4: Generate NL queries - {database}")
    print(f"{'='*80}")
    
    script_path = PROJECT_ROOT / "benchmark/generation/nl_query/generate_nl_queries.py"
    sql_dir = os.path.join(output_dir, "single")
    nl_output_dir = os.path.join(output_dir, "nl_query")
    
    # Check current generated SQL count
    current_sql_count = count_successful_sqls(sql_dir, database)
    print(f"Available SQL count: {current_sql_count}")
    
    if current_sql_count == 0:
        print(f"❌ No SQL available; skipping NL query generation")
        return False
    
    # Check existing NL query count
    nl_db_dir = os.path.join(nl_output_dir, database)
    existing_nl_count = 0
    if os.path.exists(nl_db_dir):
        existing_nl_count = len([f for f in os.listdir(nl_db_dir) 
                                 if f.startswith('generated_nl_query_') and f.endswith('.json')])
    
    # Compute how many more to generate (target - current)
    needed = max(0, target_count - existing_nl_count)
    
    if needed == 0:
        print(f"✅ Target count reached ({existing_nl_count}/{target_count}); skipping NL query generation")
        return True
    
    print(f"Target NL query count: {target_count}")
    print(f"Current NL query count: {existing_nl_count}")
    print(f"Still need to generate: {needed}")
    
    cmd = [
        "python3", str(script_path),
        "--sql_dir", sql_dir,
        "--schema_dir", schema_dir,
        "--output_dir", nl_output_dir,
        "--database", database,
        "--limit", str(needed),
        "--max_workers", "3"
    ]
    
    print(f"Running command: {' '.join(cmd)}")
    
    if not run_command(cmd):
        return False
    
    # Check results
    final_nl_count = 0
    if os.path.exists(nl_db_dir):
        final_nl_count = len([f for f in os.listdir(nl_db_dir) 
                             if f.startswith('generated_nl_query_') and f.endswith('.json')])
    
    print(f"NL query count after generation: {final_nl_count}")
    
    if final_nl_count >= target_count:
        print(f"✅ Target count reached")
        return True
    else:
        print(f"⚠️  Target not fully reached; current: {final_nl_count}, target: {target_count}")
        return final_nl_count > 0

def main():
    parser = argparse.ArgumentParser(description='Single-database end-to-end pipeline script')
    parser.add_argument('--database', type=str, required=True, help='Database name')
    parser.add_argument('--target_count', type=int, required=True, help='Target NL query count')
    parser.add_argument('--output_dir', type=str, 
                       default=str(PROJECT_ROOT / "benchmark/data/beijing/output"),
                       help='Output directory')
    parser.add_argument('--schema_dir', type=str,
                       default=str(PROJECT_ROOT / "benchmark/data/beijing/database_chinese"),
                       help='Schema directory')
    parser.add_argument('--database_dir', type=str,
                       default=str(PROJECT_ROOT / "benchmark/data/beijing/database"),
                       help='Database directory (for SQL skeleton generation)')
    parser.add_argument('--expert_file', type=str,
                       default=str(PROJECT_ROOT / "benchmark/data/target/expert_skeletons_beijing.json"),
                       help='Expert example file')
    parser.add_argument('--old_cfg_file', type=str, default=None,
                       help='Legacy database CFG file (optional)')
    parser.add_argument('--old_data_file', type=str, default=None,
                       help='Legacy data file (optional)')
    parser.add_argument('--new_logs_file', type=str, default=None,
                       help='New logs file (optional)')
    parser.add_argument('--skip_skeleton', action='store_true', help='Skip SQL skeleton generation')
    parser.add_argument('--skip_graph', action='store_true', help='Skip graph generation')
    parser.add_argument('--skip_fill', action='store_true', help='Skip SQL fill')
    parser.add_argument('--skip_nl', action='store_true', help='Skip NL query generation')
    
    args = parser.parse_args()
    
    print("=" * 80)
    print("Single-database end-to-end pipeline")
    print("=" * 80)
    print(f"Database: {args.database}")
    print(f"Target NL query count: {args.target_count}")
    print(f"Output directory: {args.output_dir}")
    print(f"Schema directory: {args.schema_dir}")
    print()
    
    success = True
    
    # Step 1: generate SQL skeletons
    if not args.skip_skeleton:
        if not step1_generate_skeletons(
            args.database, args.target_count, args.output_dir,
            args.database_dir, args.expert_file, args.old_cfg_file,
            args.old_data_file, args.new_logs_file
        ):
            print("❌ SQL skeleton generation failed")
            success = False
    else:
        print("⏭️  Skipping SQL skeleton generation")
    
    # Step 2: generate graphs
    if success and not args.skip_graph:
        if not step2_build_graphs(args.database, args.output_dir, args.schema_dir):
            print("❌ Graph generation failed")
            success = False
    else:
        print("⏭️  Skipping graph generation")
    
    # Step 3: fill SQL
    if success and not args.skip_fill:
        if not step3_fill_sqls(args.database, args.target_count, args.output_dir, args.schema_dir):
            print("❌ SQL fill failed")
            success = False
    else:
        print("⏭️  Skipping SQL fill")
    
    # Step 4: generate NL queries
    if success and not args.skip_nl:
        if not step4_generate_nl_queries(args.database, args.target_count, args.output_dir, args.schema_dir):
            print("❌ NL query generation failed")
            success = False
    else:
        print("⏭️  Skipping NL query generation")
    
    print("\n" + "=" * 80)
    if success:
        print("✅ Pipeline complete")
    else:
        print("❌ Pipeline failed")
    print("=" * 80)
    
    return 0 if success else 1

if __name__ == '__main__':
    sys.exit(main())
