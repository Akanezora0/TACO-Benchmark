#!/usr/bin/env python3
"""
基于可JOIN表对生成跨数据库SQL骨架（JOIN版本）

策略：
1. 从joinable_table_pairs.json中选择高质量的表对
2. 生成包含JOIN的SQL骨架
3. 记录推荐的JOIN列对
"""

import os
import json
import re
import argparse
import random
from collections import defaultdict

def convert_to_skeleton(sql):
    """将SQL转换为SQL骨架"""
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
    """生成JOIN SQL骨架模板"""
    if use_aggregate:
        # 带聚合函数的JOIN
        skeleton = f"SELECT _ ._ , COUNT(_ ._ ) AS _ FROM {table1} JOIN {table2} ON {table1}._ = {table2}._ GROUP BY _ ._ "
        if use_order_by:
            skeleton += "ORDER BY _ ._ "
    else:
        # 简单JOIN
        skeleton = f"SELECT _ ._ , _ ._ FROM {table1} JOIN {table2} ON {table1}._ = {table2}._ WHERE _ ._ IS NOT NULL "
        if use_order_by:
            skeleton += "ORDER BY _ ._ "
    
    return convert_to_skeleton(skeleton)

def generate_cross_database_skeletons_join(joinable_pairs_file, output_file, num_skeletons_by_db={2: 200}, min_similarity=10.0):
    """基于可JOIN表对生成跨数据库SQL骨架
    
    Args:
        num_skeletons_by_db: {数据库数量: 骨架数量}，例如 {2: 200, 3: 100, 4: 2}
    """
    
    # 1. 加载可JOIN表对
    print("加载可JOIN表对...")
    with open(joinable_pairs_file, 'r', encoding='utf-8') as f:
        joinable_data = json.load(f)
    
    # 2. 按数据库数量分类表对（2个数据库的表对）
    pairs_2db = [
        pair for pair in joinable_data['joinable_pairs']
        if pair['best_similarity'] >= min_similarity
    ]
    
    print(f"  总表对数: {len(joinable_data['joinable_pairs'])}")
    print(f"  高质量2数据库表对（相似度 >= {min_similarity}）: {len(pairs_2db)}")
    
    # 3. 为每个数据库数量类别生成骨架
    all_selected_skeletons = []
    
    for db_count in sorted(num_skeletons_by_db.keys()):
        num_skeletons = num_skeletons_by_db[db_count]
        
        if db_count == 2:
            # 2个数据库：直接使用表对
            if len(pairs_2db) < num_skeletons:
                print(f"  警告: 2数据库表对数量({len(pairs_2db)})少于目标数量({num_skeletons})")
                additional_pairs = [
                    pair for pair in joinable_data['joinable_pairs']
                    if 8.0 <= pair['best_similarity'] < min_similarity
                ]
                pairs_2db.extend(additional_pairs[:num_skeletons - len(pairs_2db)])
            
            # 确保多样性：按数据库组合分组，然后从每组中均匀采样
            pairs_by_combo = defaultdict(list)
            for pair in pairs_2db:
                combo = tuple(sorted([pair['db1'], pair['db2']]))
                pairs_by_combo[combo].append(pair)
            
            print(f"  找到 {len(pairs_by_combo)} 个不同的数据库组合")
            
            # 从每个组合中均匀采样，确保多样性
            selected_pairs = []
            pairs_per_combo = max(1, num_skeletons // len(pairs_by_combo))
            remaining = num_skeletons
            
            # 先确保每个组合至少有一个
            for combo, combo_pairs in pairs_by_combo.items():
                if remaining > 0 and len(combo_pairs) > 0:
                    sample_size = min(pairs_per_combo, len(combo_pairs), remaining)
                    selected = random.sample(combo_pairs, sample_size)
                    selected_pairs.extend(selected)
                    remaining -= len(selected)
            
            # 如果还需要更多，随机补充
            if remaining > 0:
                all_remaining = [p for p in pairs_2db if p not in selected_pairs]
                if len(all_remaining) > 0:
                    additional = random.sample(all_remaining, min(remaining, len(all_remaining)))
                    selected_pairs.extend(additional)
            
            # 打乱顺序，增加随机性
            random.shuffle(selected_pairs)
            selected_pairs = selected_pairs[:num_skeletons]
            
            print(f"  选择了 {len(selected_pairs)} 个2数据库表对（来自 {len(set(tuple(sorted([p['db1'], p['db2']])) for p in selected_pairs))} 个不同组合）")
            
            # 为每个表对生成骨架
            for idx, pair in enumerate(selected_pairs):
                db1 = pair['db1']
                db2 = pair['db2']
                table1 = pair['table1']
                table2 = pair['table2']
                
                # 获取最佳列对
                best_col_pair = pair['column_pairs'][0]
                col1 = best_col_pair['col1']
                col2 = best_col_pair['col2']
                
                # 决定是否使用聚合函数和ORDER BY
                use_aggregate = random.random() < 0.4
                use_order_by = random.random() < 0.3
                
                # 生成SQL骨架
                skeleton = generate_join_skeleton_template(
                    table1, table2, col1, col2, 
                    use_aggregate=use_aggregate,
                    use_order_by=use_order_by
                )
                
                # 构建表到数据库的映射
                table_database_mapping = {
                    table1: db1,
                    table2: db2
                }
                
                # 记录推荐的JOIN列对
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
            # 3个数据库：从2数据库表对组合生成
            # 策略：找到3个表对，使得它们覆盖3个不同的数据库
            # 方式1：链式连接 A-B, B-C（共享数据库B）
            # 方式2：星型连接 A-B, A-C（共享数据库A，但B和C不同）
            print(f"\n生成 {num_skeletons} 个3数据库骨架...")
            
            # 构建数据库对到表对的映射
            db_pair_to_pairs = defaultdict(list)
            for pair in pairs_2db:
                db_pair = (pair['db1'], pair['db2'])
                db_pair_to_pairs[db_pair].append(pair)
            
            # 尝试生成3数据库骨架
            generated_3db = 0
            max_attempts = num_skeletons * 10  # 最多尝试次数
            attempts = 0
            
            while generated_3db < num_skeletons and attempts < max_attempts:
                attempts += 1
                
                # 策略1：链式连接 A-B, B-C
                # 随机选择一个表对 A-B
                if len(pairs_2db) == 0:
                    break
                pair1 = random.choice(pairs_2db)
                db1, db2 = pair1['db1'], pair1['db2']
                
                # 找到与db2连接的另一个数据库的表对 B-C
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
                
                # 确保是3个不同的数据库
                databases = sorted([db1, db2, db3])
                if len(set(databases)) != 3:
                    continue
                
                # 构建3数据库JOIN骨架
                table1 = pair1['table1']
                table2 = pair1['table2']
                table3 = pair2['table2'] if pair2['db1'] == db2 else pair2['table1']
                
                # 获取JOIN列对
                best_col_pair1 = pair1['column_pairs'][0]
                best_col_pair2 = pair2['column_pairs'][0]
                
                # 决定是否使用聚合函数和ORDER BY
                use_aggregate = random.random() < 0.4
                use_order_by = random.random() < 0.3
                
                # 生成SQL骨架：table1 JOIN table2 ON ... JOIN table3 ON ...
                if use_aggregate:
                    skeleton = f"SELECT _ ._ , COUNT(_ ._ ) AS _ FROM {table1} JOIN {table2} ON {table1}._ = {table2}._ JOIN {table3} ON {table2}._ = {table3}._ GROUP BY _ ._ "
                    if use_order_by:
                        skeleton += "ORDER BY _ ._ "
                else:
                    skeleton = f"SELECT _ ._ , _ ._ , _ ._ FROM {table1} JOIN {table2} ON {table1}._ = {table2}._ JOIN {table3} ON {table2}._ = {table3}._ WHERE _ ._ IS NOT NULL "
                    if use_order_by:
                        skeleton += "ORDER BY _ ._ "
                
                skeleton = convert_to_skeleton(skeleton)
                
                # 构建表到数据库的映射
                table_database_mapping = {
                    table1: db1,
                    table2: db2,
                    table3: db3
                }
                
                # 记录推荐的JOIN列对
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
            
            print(f"  成功生成 {generated_3db} 个3数据库骨架")
            if generated_3db < num_skeletons:
                print(f"  警告: 只生成了 {generated_3db}/{num_skeletons} 个3数据库骨架")
        
        elif db_count == 4:
            # 4个数据库：从2数据库表对组合生成
            # 策略：找到4个表对，形成链式连接 A-B, B-C, C-D
            print(f"\n生成 {num_skeletons} 个4数据库骨架...")
            
            generated_4db = 0
            max_attempts = num_skeletons * 20  # 最多尝试次数
            attempts = 0
            
            while generated_4db < num_skeletons and attempts < max_attempts:
                attempts += 1
                
                # 策略：链式连接 A-B, B-C, C-D
                # 随机选择一个表对 A-B
                if len(pairs_2db) == 0:
                    break
                pair1 = random.choice(pairs_2db)
                db1, db2 = pair1['db1'], pair1['db2']
                
                # 找到与db2连接的另一个数据库的表对 B-C
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
                
                # 找到与db3连接的另一个数据库的表对 C-D
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
                
                # 确保是4个不同的数据库
                databases = sorted([db1, db2, db3, db4])
                if len(set(databases)) != 4:
                    continue
                
                # 构建4数据库JOIN骨架
                table1 = pair1['table1']
                table2 = pair1['table2']
                table3 = pair2['table2'] if pair2['db1'] == db2 else pair2['table1']
                table4 = pair3['table2'] if pair3['db1'] == db3 else pair3['table1']
                
                # 获取JOIN列对
                best_col_pair1 = pair1['column_pairs'][0]
                best_col_pair2 = pair2['column_pairs'][0]
                best_col_pair3 = pair3['column_pairs'][0]
                
                # 决定是否使用聚合函数和ORDER BY
                use_aggregate = random.random() < 0.4
                use_order_by = random.random() < 0.3
                
                # 生成SQL骨架：table1 JOIN table2 ON ... JOIN table3 ON ... JOIN table4 ON ...
                if use_aggregate:
                    skeleton = f"SELECT _ ._ , COUNT(_ ._ ) AS _ FROM {table1} JOIN {table2} ON {table1}._ = {table2}._ JOIN {table3} ON {table2}._ = {table3}._ JOIN {table4} ON {table3}._ = {table4}._ GROUP BY _ ._ "
                    if use_order_by:
                        skeleton += "ORDER BY _ ._ "
                else:
                    skeleton = f"SELECT _ ._ , _ ._ , _ ._ , _ ._ FROM {table1} JOIN {table2} ON {table1}._ = {table2}._ JOIN {table3} ON {table2}._ = {table3}._ JOIN {table4} ON {table3}._ = {table4}._ WHERE _ ._ IS NOT NULL "
                    if use_order_by:
                        skeleton += "ORDER BY _ ._ "
                
                skeleton = convert_to_skeleton(skeleton)
                
                # 构建表到数据库的映射
                table_database_mapping = {
                    table1: db1,
                    table2: db2,
                    table3: db3,
                    table4: db4
                }
                
                # 记录推荐的JOIN列对
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
            
            print(f"  成功生成 {generated_4db} 个4数据库骨架")
            if generated_4db < num_skeletons:
                print(f"  警告: 只生成了 {generated_4db}/{num_skeletons} 个4数据库骨架")
    
    # 4. 保存骨架
    print(f"\n保存SQL骨架...")
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(all_selected_skeletons, f, ensure_ascii=False, indent=2)
    
    print(f"已生成 {len(all_selected_skeletons)} 个跨数据库SQL骨架（JOIN版本）")
    
    # 5. 统计
    print("\n统计信息:")
    similarity_dist = defaultdict(int)
    aggregate_count = sum(1 for s in all_selected_skeletons if s.get('use_aggregate', False))
    order_by_count = sum(1 for s in all_selected_skeletons if s.get('use_order_by', False))
    
    for skeleton in all_selected_skeletons:
        sim = skeleton.get('similarity', 0)
        sim_range = int(sim // 2) * 2
        similarity_dist[sim_range] += 1
    
    if all_selected_skeletons:
        print(f"  使用聚合函数: {aggregate_count} ({aggregate_count/len(all_selected_skeletons)*100:.1f}%)")
        print(f"  使用ORDER BY: {order_by_count} ({order_by_count/len(all_selected_skeletons)*100:.1f}%)")
        print(f"  相似度分布:")
        for sim_range in sorted(similarity_dist.keys(), reverse=True):
            print(f"    {sim_range}-{sim_range+2}: {similarity_dist[sim_range]} 个")
    
    # 6. 按数据库组合统计
    db_combo_dist = defaultdict(int)
    for skeleton in all_selected_skeletons:
        dbs = sorted(skeleton.get('databases', []))
        if len(dbs) >= 2:
            key = f"{dbs[0]} + {dbs[1]}"
            db_combo_dist[key] += 1
    
    if db_combo_dist:
        print(f"\n  按数据库组合分布（前10个）:")
        for combo, count in sorted(db_combo_dist.items(), key=lambda x: x[1], reverse=True)[:10]:
            print(f"    {combo}: {count} 个")

def main():
    parser = argparse.ArgumentParser(description='基于可JOIN表对生成跨数据库SQL骨架（JOIN版本）')
    parser.add_argument('--joinable_pairs_file', type=str,
                       default='benchmark/generation/cross_database/joinable_table_pairs.json',
                       help='可JOIN表对文件')
    parser.add_argument('--output_file', type=str,
                       default='benchmark/generation/cross_database/cross_db_skeletons_join.json',
                       help='输出文件')
    parser.add_argument('--num_skeletons_2db', type=int, default=200,
                       help='2数据库的SQL骨架数量')
    parser.add_argument('--num_skeletons_3db', type=int, default=0,
                       help='3数据库的SQL骨架数量')
    parser.add_argument('--num_skeletons_4db', type=int, default=0,
                       help='4数据库的SQL骨架数量')
    parser.add_argument('--min_similarity', type=float, default=10.0,
                       help='最小相似度阈值')
    
    args = parser.parse_args()
    
    num_skeletons_by_db = {}
    if args.num_skeletons_2db > 0:
        num_skeletons_by_db[2] = args.num_skeletons_2db
    if args.num_skeletons_3db > 0:
        num_skeletons_by_db[3] = args.num_skeletons_3db
    if args.num_skeletons_4db > 0:
        num_skeletons_by_db[4] = args.num_skeletons_4db
    
    print("=" * 70)
    print("生成跨数据库SQL骨架（JOIN版本）")
    print("=" * 70)
    print(f"\n可JOIN表对文件: {args.joinable_pairs_file}")
    print(f"输出文件: {args.output_file}")
    print(f"目标数量: {num_skeletons_by_db}")
    print(f"最小相似度: {args.min_similarity}")
    print()
    
    generate_cross_database_skeletons_join(
        args.joinable_pairs_file,
        args.output_file,
        num_skeletons_by_db,
        args.min_similarity
    )
    
    print("\n" + "=" * 70)
    print("完成！")
    print("=" * 70)

if __name__ == '__main__':
    main()

