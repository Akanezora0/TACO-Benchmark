#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Beijing dataset cross-database SQL generation - one-click runner

Supports step-by-step execution or the full pipeline.
"""

import sys
import argparse
import subprocess
from pathlib import Path

SCRIPT_DIR = Path(__file__).parent


def resolve_script(script_name: str) -> Path:
    """Resolve a script in this directory."""
    return SCRIPT_DIR / script_name


def run_step(step_num, step_name, script_name, description=""):
    """Run a single step."""
    print("\n" + "=" * 80)
    print(f"Step {step_num}: {step_name}")
    if description:
        print(f"Description: {description}")
    print("=" * 80)

    script_path = resolve_script(script_name)
    if not script_path.exists():
        print(f"Error: script does not exist: {script_path}")
        return False

    try:
        result = subprocess.run(
            [sys.executable, str(script_path)],
            cwd=str(script_path.parent),
            check=False,
        )
        return result.returncode == 0
    except Exception as e:
        print(f"Error: exception while running script: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Beijing cross-database SQL generation - one-click runner",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Step overview:
  1. analyze_joinable_tables   - Find joinable table pairs
  2. generate_skeletons        - Build cross-DB SQL skeletons
  3. build_graphs              - Schema-linking graphs (local, not in git)
  4. generate_2db              - 2-database JOIN SQL
  5. generate_3db_4db          - 3- and 4-database JOIN SQL

Examples:
  python3 run_all.py --status
  python3 run_all.py
  python3 run_all.py --step 3
  python3 run_all.py --from-step 3
        """,
    )

    parser.add_argument("--status", action="store_true", help="Show status only")
    parser.add_argument("--step", type=int, default=None, help="Run only step 1-5")
    parser.add_argument("--from-step", type=int, default=1, help="Start step (default: 1)")
    parser.add_argument("--to-step", type=int, default=5, help="End step (default: 5)")
    parser.add_argument("--skip-step", type=int, nargs="+", default=[], help="Skip steps")
    parser.add_argument("-y", "--yes", action="store_true", help="Skip confirmation prompt")

    args = parser.parse_args()

    if args.status:
        status_script = SCRIPT_DIR / "check_generation_status.py"
        if status_script.exists():
            subprocess.run([sys.executable, str(status_script)], cwd=str(SCRIPT_DIR))
        else:
            print("Error: status check script does not exist")
        return

    steps = [
        (1, "Analyze joinable table pairs", "analyze_joinable_tables.py", "Find joinable table pairs across Beijing databases"),
        (2, "Generate SQL skeletons", "generate_cross_db_skeletons_join.py", "Build JOIN SQL skeletons (2/3/4 DB)"),
        (3, "Build schema graphs", "cross_db_1build_schema_graphs.py", "Generate schema-linking graphs (writes cross_db_graphs_join/)"),
        (4, "Generate 2-database SQL", "generate_more_join_sqls.py", "Batch-generate 2-database JOIN SQL"),
        (5, "Generate 3- and 4-database SQL", "generate_3db_4db_sqls.py", "Generate 3- and 4-database JOIN SQL"),
    ]

    if args.step:
        steps_to_run = [s for s in steps if s[0] == args.step]
    else:
        steps_to_run = [
            s for s in steps
            if args.from_step <= s[0] <= args.to_step and s[0] not in args.skip_step
        ]

    if not steps_to_run:
        print("No steps to execute")
        return

    print("=" * 80)
    print("Beijing cross-database SQL generation - one-click runner")
    print("=" * 80)
    print("\nSteps to execute:")
    for step_num, step_name, _, _ in steps_to_run:
        print(f"  {step_num}. {step_name}")

    if not args.yes:
        confirm = input("\nStart execution? (y/n): ")
        if confirm.lower() != "y":
            print("Cancelled")
            return

    success_count = 0
    fail_count = 0

    for step_num, step_name, script_name, description in steps_to_run:
        success = run_step(step_num, step_name, script_name, description)
        if success:
            success_count += 1
            print(f"Step {step_num} complete")
        else:
            fail_count += 1
            print(f"Step {step_num} failed")
            if step_num < steps_to_run[-1][0] and not args.yes:
                continue_choice = input(f"\nStep {step_num} failed. Continue? (y/n): ")
                if continue_choice.lower() != "y":
                    break

    print("\n" + "=" * 80)
    print("Execution complete")
    print(f"Success: {success_count} steps")
    print(f"Failed: {fail_count} steps")
    print("=" * 80)

    print("\nFinal status:")
    status_script = SCRIPT_DIR / "check_generation_status.py"
    if status_script.exists():
        subprocess.run([sys.executable, str(status_script)], cwd=str(SCRIPT_DIR))


if __name__ == "__main__":
    main()
