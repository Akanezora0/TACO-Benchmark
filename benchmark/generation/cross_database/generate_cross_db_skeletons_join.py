#!/usr/bin/env python3
"""
Generate cross-database SQL skeletons from joinable table pairs (JOIN version)

Strategy:
1. Select high-quality table pairs from joinable_table_pairs.json
2. Generate SQL skeletons that include JOINs
3. Record recommended JOIN column pairs
"""

import os
import json
import re
import argparse
import random
from collections import defaultdict

def convert_to_skeleton(sql):
    """Convert SQL to an SQL skeleton."""
    sql_skeleton = re.sub(r"'[^']*'", '_', sql)
    sql_skeleton = re.sub(r'"[^"]*"', '_', sql_skeleton)
    sql_skeleton = re.sub(r'\b\d+\b', '_', sql_skeleton)
    sql_keywords = set(['SELECT', 'FROM', 'WHERE', 'GROUP', 'BY', 'HAVING', 'ORDER', 'LIMIT', 'JOIN', 'ON', 'AS', 'AND', 'OR', 'IN', 'NOT', 'NULL', 'IS', 'DISTINCT', 'COUNT', 'SUM', 'AVG', 'MIN', 'MAX', 'INNER', 'LEFT', 'RIGHT', 'FULL', 'OUTER', 'UNION', 'ALL'])
    def replace_identifier(match):
        word = match.group(0)
        if word.upper() in sql_keywords or word == '*':
            return word
        else:
            return '_'
    sql_skeleton = re.sub(r'\b\w+\b', replace_identifier, sql_skeleton)
    sql_skeleton = re.sub(r'(_\s*)+', '_ ', sql_skeleton)
    sql_skeleton = ' '.join(sql_skeleton.strip().split())
    return sql_skeleton

def generate_join_skeleton_template(table1, table2, col1, col2, use_aggregate=False, use_order_by=False):
    """Generate a JOIN SQL skeleton template."""
    if use_aggregate:
        # JOIN with aggregate functions
        skeleton = f"SELECT _ ._ , COUNT(_ ._ ) AS _ FROM {table1} JOIN {table2} ON {table1}._ = {table2}._ GROUP BY _ ._ "
        if use_order_by:
            skeleton += "ORDER BY _ ._ "
    else:
        # Simple JOIN
        skeleton = f"SELECT _ ._ , _ ._ FROM {table1} JOIN {table2} ON {table1}._ = {table2}._ WHERE _ ._ IS NOT NULL "
        if use_order_by:
            skeleton += "ORDER BY _ ._ "
    
    return convert_to_skeleton(skeleton)

def generate_cross_database_skeletons_join(joinable_pairs_file, output_file, num_skeletons_by_db={2: 200}, min_similarity=10.0):
    """Generate cross-database SQL skeletons from joinable table pairs.
    
    Args:
        num_skeletons_by_db: {database_count: skeleton_count}, e.g. {2: 200, 3: 100, 4: 2}
    """
    
    # 1. Load joinable table pairs
    print("Loading joinable table pairs...")
    with open(joinable_pairs_file, 'r', encoding='utf-8') as f:
        joinable_data = json.load(f)
    
    # 2. Filter table pairs by database count (2-database pairs)
    pairs_2db = [
        pair for pair in joinable_data['joinable_pairs']
        if pair['best_similarity'] >= min_similarity
    ]
    
    print(f"  Total table pairs: {len(joinable_data['joinable_pairs'])}")
    print(f"  High-quality 2-database pairs (similarity >= {min_similarity}): {len(pairs_2db)}")
    
    # 3. Generate skeletons for each database-count category
    all_selected_skeletons = []
    
    for db_count in sorted(num_skeletons_by_db.keys()):
        num_skeletons = num_skeletons_by_db[db_count]
        
        if db_count == 2:
            # 2 databases: use table pairs directly
            if len(pairs_2db) < num_skeletons:
                print(f"  Warning: 2-database pair count ({len(pairs_2db)}) is below target ({num_skeletons})")
                additional_pairs = [
                    pair for pair in joinable_data['joinable_pairs']
                    if 8.0 <= pair['best_similarity'] < min_similarity
                ]
                pairs_2db.extend(additional_pairs[:num_skeletons - len(pairs_2db)])
            
            # Ensure diversity: group by database combination, then sample evenly from each group
            pairs_by_combo = defaultdict(list)
            for pair in pairs_2db:
                combo = tuple(sorted([pair['db1'], pair['db2']]))
                pairs_by_combo[combo].append(pair)
            
            print(f"  Found {len(pairs_by_combo)} distinct database combinations")
            
            # Sample evenly from each combination to ensure diversity
            selected_pairs = []
            pairs_per_combo = max(1, num_skeletons // len(pairs_by_combo))
            remaining = num_skeletons
            
            # Ensure at least one pair from each combination first
            for combo, combo_pairs in pairs_by_combo.items():
                if remaining > 0 and len(combo_pairs) > 0:
                    sample_size = min(pairs_per_combo, len(combo_pairs), remaining)
                    selected = random.sample(combo_pairs, sample_size)
                    selected_pairs.extend(selected)
                    remaining -= len(selected)
            
            # Fill remaining slots randomly if needed
            if remaining > 0:
                all_remaining = [p for p in pairs_2db if p not in selected_pairs]
                if len(all_remaining) > 0:
                    additional = random.sample(all_remaining, min(remaining, len(all_remaining)))
                    selected_pairs.extend(additional)
            
            # Shuffle to add randomness
            random.shuffle(selected_pairs)
            selected_pairs = selected_pairs[:num_skeletons]
            
            print(f"  Selected {len(selected_pairs)} 2-database table pairs (from {len(set(tuple(sorted([p['db1'], p['db2']])) for p in selected_pairs))} distinct combinations)")
            
            # Generate a skeleton for each table pair
            for idx, pair in enumerate(selected_pairs):
                db1 = pair['db1']
                db2 = pair['db2']
                table1 = pair['table1']
                table2 = pair['table2']
                
                # Get the best column pair
                best_col_pair = pair['column_pairs'][0]
                col1 = best_col_pair['col1']
                col2 = best_col_pair['col2']
                
                # Decide whether to use aggregate functions and ORDER BY
                use_aggregate = random.random() < 0.4
                use_order_by = random.random() < 0.3
                
                # Generate SQL skeleton
                skeleton = generate_join_skeleton_template(
                    table1, table2, col1, col2, 
                    use_aggregate=use_aggregate,
                    use_order_by=use_order_by
                )
                
                # Build table-to-database mapping
                table_database_mapping = {
                    table1: db1,
                    table2: db2
                }
                
                # Record recommended JOIN column pairs
                recommended_join_columns = {
                    f"{db1}.{table1}.{col1}": f"{db2}.{table2}.{col2}"
                }
                
                skeleton_data = {
                    'original_sql': None,
                    'original_database': None,
                    'original_file': f"join_skeleton_{len(all_selected_skeletons)}",
                    'sql_skeleton': skeleton,
                    'databases': [db1, db2],
                    'table_database_mapping': table_database_mapping,
                    'tables': [table1, table2],
                    'table_count': 2,
                    'is_cross_database': True,
                    'num_databases': 2,
                    'join_type': 'JOIN',
                    'recommended_join_columns': recommended_join_columns,
                    'similarity': pair['best_similarity'],
                    'use_aggregate': use_aggregate,
                    'use_order_by': use_order_by
                }
                all_selected_skeletons.append(skeleton_data)
        
        elif db_count == 3:
            # 3 databases: combine 2-database table pairs
            # Strategy: find 3 table pairs covering 3 distinct databases
            # Option 1: chain A-B, B-C (shared database B)
            # Option 2: star A-B, A-C (shared database A, but B and C differ)
            print(f"\nGenerating {num_skeletons} 3-database skeletons...")
            
            # Build mapping from database pairs to table pairs
            db_pair_to_pairs = defaultdict(list)
            for pair in pairs_2db:
                db_pair = (pair['db1'], pair['db2'])
                db_pair_to_pairs[db_pair].append(pair)
            
            # Attempt to generate 3-database skeletons
            generated_3db = 0
            max_attempts = num_skeletons * 10  # Maximum number of attempts
            attempts = 0
            
            while generated_3db < num_skeletons and attempts < max_attempts:
                attempts += 1
                
                # Strategy 1: chain A-B, B-C
                # Randomly select a table pair A-B
                if len(pairs_2db) == 0:
                    break
                pair1 = random.choice(pairs_2db)
                db1, db2 = pair1['db1'], pair1['db2']
                
                # Find a table pair B-C connecting db2 to another database
                candidate_pairs = [
                    p for p in pairs_2db 
                    if (p['db1'] == db2 and p['db2'] != db1) or (p['db2'] == db2 and p['db1'] != db1)
                ]
                
                if not candidate_pairs:
                    continue
                
                pair2 = random.choice(candidate_pairs)
                if pair2['db1'] == db2:
                    db3 = pair2['db2']
                else:
                    db3 = pair2['db1']
                
                # Ensure 3 distinct databases
                databases = sorted([db1, db2, db3])
                if len(set(databases)) != 3:
                    continue
                
                # Build 3-database JOIN skeleton
                table1 = pair1['table1']
                table2 = pair1['table2']
                table3 = pair2['table2'] if pair2['db1'] == db2 else pair2['table1']
                
                # Get JOIN column pairs
                best_col_pair1 = pair1['column_pairs'][0]
                best_col_pair2 = pair2['column_pairs'][0]
                
                # Decide whether to use aggregate functions and ORDER BY
                use_aggregate = random.random() < 0.4
                use_order_by = random.random() < 0.3
                
                # Generate SQL skeleton: table1 JOIN table2 ON ... JOIN table3 ON ...
                if use_aggregate:
                    skeleton = f"SELECT _ ._ , COUNT(_ ._ ) AS _ FROM {table1} JOIN {table2} ON {table1}._ = {table2}._ JOIN {table3} ON {table2}._ = {table3}._ GROUP BY _ ._ "
                    if use_order_by:
                        skeleton += "ORDER BY _ ._ "
                else:
                    skeleton = f"SELECT _ ._ , _ ._ , _ ._ FROM {table1} JOIN {table2} ON {table1}._ = {table2}._ JOIN {table3} ON {table2}._ = {table3}._ WHERE _ ._ IS NOT NULL "
                    if use_order_by:
                        skeleton += "ORDER BY _ ._ "
                
                skeleton = convert_to_skeleton(skeleton)
                
                # Build table-to-database mapping
                table_database_mapping = {
                    table1: db1,
                    table2: db2,
                    table3: db3
                }
                
                # Record recommended JOIN column pairs
                recommended_join_columns = {
                    f"{db1}.{table1}.{best_col_pair1['col1']}": f"{db2}.{table2}.{best_col_pair1['col2']}",
                    f"{db2}.{table2}.{best_col_pair2['col1'] if pair2['db1'] == db2 else best_col_pair2['col2']}": 
                    f"{db3}.{table3}.{best_col_pair2['col2'] if pair2['db1'] == db2 else best_col_pair2['col1']}"
                }
                
                skeleton_data = {
                    'original_sql': None,
                    'original_database': None,
                    'original_file': f"join_skeleton_{len(all_selected_skeletons)}",
                    'sql_skeleton': skeleton,
                    'databases': databases,
                    'table_database_mapping': table_database_mapping,
                    'tables': [table1, table2, table3],
                    'table_count': 3,
                    'is_cross_database': True,
                    'num_databases': 3,
                    'join_type': 'JOIN',
                    'recommended_join_columns': recommended_join_columns,
                    'similarity': min(pair1['best_similarity'], pair2['best_similarity']),
                    'use_aggregate': use_aggregate,
                    'use_order_by': use_order_by
                }
                all_selected_skeletons.append(skeleton_data)
                generated_3db += 1
            
            print(f"  Successfully generated {generated_3db} 3-database skeletons")
            if generated_3db < num_skeletons:
                print(f"  Warning: only generated {generated_3db}/{num_skeletons} 3-database skeletons")
        
        elif db_count == 4:
            # 4 databases: combine 2-database table pairs
            # Strategy: find 4 table pairs forming a chain A-B, B-C, C-D
            print(f"\nGenerating {num_skeletons} 4-database skeletons...")
            
            generated_4db = 0
            max_attempts = num_skeletons * 20  # Maximum number of attempts
            attempts = 0
            
            while generated_4db < num_skeletons and attempts < max_attempts:
                attempts += 1
                
                # Strategy: chain A-B, B-C, C-D
                # Randomly select a table pair A-B
                if len(pairs_2db) == 0:
                    break
                pair1 = random.choice(pairs_2db)
                db1, db2 = pair1['db1'], pair1['db2']
                
                # Find a table pair B-C connecting db2 to another database
                candidate_pairs2 = [
                    p for p in pairs_2db 
                    if (p['db1'] == db2 and p['db2'] not in [db1]) or (p['db2'] == db2 and p['db1'] not in [db1])
                ]
                
                if not candidate_pairs2:
                    continue
                
                pair2 = random.choice(candidate_pairs2)
                if pair2['db1'] == db2:
                    db3 = pair2['db2']
                else:
                    db3 = pair2['db1']
                
                # Find a table pair C-D connecting db3 to another database
                candidate_pairs3 = [
                    p for p in pairs_2db 
                    if ((p['db1'] == db3 and p['db2'] not in [db1, db2]) or 
                        (p['db2'] == db3 and p['db1'] not in [db1, db2]))
                ]
                
                if not candidate_pairs3:
                    continue
                
                pair3 = random.choice(candidate_pairs3)
                if pair3['db1'] == db3:
                    db4 = pair3['db2']
                else:
                    db4 = pair3['db1']
                
                # Ensure 4 distinct databases
                databases = sorted([db1, db2, db3, db4])
                if len(set(databases)) != 4:
                    continue
                
                # Build 4-database JOIN skeleton
                table1 = pair1['table1']
                table2 = pair1['table2']
                table3 = pair2['table2'] if pair2['db1'] == db2 else pair2['table1']
                table4 = pair3['table2'] if pair3['db1'] == db3 else pair3['table1']
                
                # Get JOIN column pairs
                best_col_pair1 = pair1['column_pairs'][0]
                best_col_pair2 = pair2['column_pairs'][0]
                best_col_pair3 = pair3['column_pairs'][0]
                
                # Decide whether to use aggregate functions and ORDER BY
                use_aggregate = random.random() < 0.4
                use_order_by = random.random() < 0.3
                
                # Generate SQL skeleton: table1 JOIN table2 ON ... JOIN table3 ON ... JOIN table4 ON ...
                if use_aggregate:
                    skeleton = f"SELECT _ ._ , COUNT(_ ._ ) AS _ FROM {table1} JOIN {table2} ON {table1}._ = {table2}._ JOIN {table3} ON {table2}._ = {table3}._ JOIN {table4} ON {table3}._ = {table4}._ GROUP BY _ ._ "
                    if use_order_by:
                        skeleton += "ORDER BY _ ._ "
                else:
                    skeleton = f"SELECT _ ._ , _ ._ , _ ._ , _ ._ FROM {table1} JOIN {table2} ON {table1}._ = {table2}._ JOIN {table3} ON {table2}._ = {table3}._ JOIN {table4} ON {table3}._ = {table4}._ WHERE _ ._ IS NOT NULL "
                    if use_order_by:
                        skeleton += "ORDER BY _ ._ "
                
                skeleton = convert_to_skeleton(skeleton)
                
                # Build table-to-database mapping
                table_database_mapping = {
                    table1: db1,
                    table2: db2,
                    table3: db3,
                    table4: db4
                }
                
                # Record recommended JOIN column pairs
                recommended_join_columns = {
                    f"{db1}.{table1}.{best_col_pair1['col1']}": f"{db2}.{table2}.{best_col_pair1['col2']}",
                    f"{db2}.{table2}.{best_col_pair2['col1'] if pair2['db1'] == db2 else best_col_pair2['col2']}": 
                    f"{db3}.{table3}.{best_col_pair2['col2'] if pair2['db1'] == db2 else best_col_pair2['col1']}",
                    f"{db3}.{table3}.{best_col_pair3['col1'] if pair3['db1'] == db3 else best_col_pair3['col2']}": 
                    f"{db4}.{table4}.{best_col_pair3['col2'] if pair3['db1'] == db3 else best_col_pair3['col1']}"
                }
                
                skeleton_data = {
                    'original_sql': None,
                    'original_database': None,
                    'original_file': f"join_skeleton_{len(all_selected_skeletons)}",
                    'sql_skeleton': skeleton,
                    'databases': databases,
                    'table_database_mapping': table_database_mapping,
                    'tables': [table1, table2, table3, table4],
                    'table_count': 4,
                    'is_cross_database': True,
                    'num_databases': 4,
                    'join_type': 'JOIN',
                    'recommended_join_columns': recommended_join_columns,
                    'similarity': min(pair1['best_similarity'], pair2['best_similarity'], pair3['best_similarity']),
                    'use_aggregate': use_aggregate,
                    'use_order_by': use_order_by
                }
                all_selected_skeletons.append(skeleton_data)
                generated_4db += 1
            
            print(f"  Successfully generated {generated_4db} 4-database skeletons")
            if generated_4db < num_skeletons:
                print(f"  Warning: only generated {generated_4db}/{num_skeletons} 4-database skeletons")
    
    # 4. Save skeletons
    print(f"\nSaving SQL skeletons...")
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(all_selected_skeletons, f, ensure_ascii=False, indent=2)
    
    print(f"Generated {len(all_selected_skeletons)} cross-database SQL skeletons (JOIN version)")
    
    # 5. Statistics
    print("\nStatistics:")
    similarity_dist = defaultdict(int)
    aggregate_count = sum(1 for s in all_selected_skeletons if s.get('use_aggregate', False))
    order_by_count = sum(1 for s in all_selected_skeletons if s.get('use_order_by', False))
    
    for skeleton in all_selected_skeletons:
        sim = skeleton.get('similarity', 0)
        sim_range = int(sim // 2) * 2
        similarity_dist[sim_range] += 1
    
    if all_selected_skeletons:
        print(f"  Using aggregate functions: {aggregate_count} ({aggregate_count/len(all_selected_skeletons)*100:.1f}%)")
        print(f"  Using ORDER BY: {order_by_count} ({order_by_count/len(all_selected_skeletons)*100:.1f}%)")
        print(f"  Similarity distribution:")
        for sim_range in sorted(similarity_dist.keys(), reverse=True):
            print(f"    {sim_range}-{sim_range+2}: {similarity_dist[sim_range]}")
    
    # 6. Statistics by database combination
    db_combo_dist = defaultdict(int)
    for skeleton in all_selected_skeletons:
        dbs = sorted(skeleton.get('databases', []))
        if len(dbs) >= 2:
            key = f"{dbs[0]} + {dbs[1]}"
            db_combo_dist[key] += 1
    
    if db_combo_dist:
        print(f"\n  Distribution by database combination (top 10):")
        for combo, count in sorted(db_combo_dist.items(), key=lambda x: x[1], reverse=True)[:10]:
            print(f"    {combo}: {count}")

def main():
    parser = argparse.ArgumentParser(description='Generate cross-database SQL skeletons from joinable table pairs (JOIN version)')
    parser.add_argument('--joinable_pairs_file', type=str,
                       default='benchmark/generation/cross_database/joinable_table_pairs.json',
                       help='Joinable table pairs file')
    parser.add_argument('--output_file', type=str,
                       default='benchmark/generation/cross_database/cross_db_skeletons_join.json',
                       help='Output file')
    parser.add_argument('--num_skeletons_2db', type=int, default=200,
                       help='Number of SQL skeletons for 2 databases')
    parser.add_argument('--num_skeletons_3db', type=int, default=0,
                       help='Number of SQL skeletons for 3 databases')
    parser.add_argument('--num_skeletons_4db', type=int, default=0,
                       help='Number of SQL skeletons for 4 databases')
    parser.add_argument('--min_similarity', type=float, default=10.0,
                       help='Minimum similarity threshold')
    
    args = parser.parse_args()
    
    num_skeletons_by_db = {}
    if args.num_skeletons_2db > 0:
        num_skeletons_by_db[2] = args.num_skeletons_2db
    if args.num_skeletons_3db > 0:
        num_skeletons_by_db[3] = args.num_skeletons_3db
    if args.num_skeletons_4db > 0:
        num_skeletons_by_db[4] = args.num_skeletons_4db
    
    print("=" * 70)
    print("Generate cross-database SQL skeletons (JOIN version)")
    print("=" * 70)
    print(f"\nJoinable table pairs file: {args.joinable_pairs_file}")
    print(f"Output file: {args.output_file}")
    print(f"Target counts: {num_skeletons_by_db}")
    print(f"Minimum similarity: {args.min_similarity}")
    print()
    
    generate_cross_database_skeletons_join(
        args.joinable_pairs_file,
        args.output_file,
        num_skeletons_by_db,
        args.min_similarity
    )
    
    print("\n" + "=" * 70)
    print("Done!")
    print("=" * 70)

if __name__ == '__main__':
    main()
