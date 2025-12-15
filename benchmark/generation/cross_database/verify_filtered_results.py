#!/usr/bin/env python3
"""
验证筛选后的SQL质量
"""

import os
import json
from collections import defaultdict

def verify_filtered_sqls(final_dir):
    """验证筛选后的SQL质量"""
    stats = {
        'total': 0,
        'by_db_count': defaultdict(lambda: {'total': 0, 'with_results': 0, 'null_issues': 0}),
        'db_combinations': set(),
        'sql_structures': defaultdict(int)
    }
    
    if not os.path.exists(final_dir):
        print(f"错误: 目录不存在 {final_dir}")
        return
    
    for filename in sorted(os.listdir(final_dir)):
        if not filename.startswith('cross_db_generated_sql_') or not filename.endswith('.json'):
            continue
        
        file_path = os.path.join(final_dir, filename)
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            databases = data.get('databases', [])
            num_databases = len(databases)
            metadata = data.get('metadata', {})
            if 'num_databases' in metadata:
                num_databases = metadata['num_databases']
            
            if num_databases < 2 or num_databases > 4:
                continue
            
            results = data.get('results', [])
            sql = data.get('sql', '')
            
            stats['total'] += 1
            stats['by_db_count'][num_databases]['total'] += 1
            
            if results and len(results) > 0:
                stats['by_db_count'][num_databases]['with_results'] += 1
                
                # 检查null问题
                has_null_issue = False
                for row in results:
                    if isinstance(row, (list, tuple)):
                        if all(cell is None or (isinstance(cell, str) and cell.upper() in ['NULL', 'NONE', '']) for cell in row):
                            has_null_issue = True
                            break
                    else:
                        if row is None or (isinstance(row, str) and row.upper() in ['NULL', 'NONE', '']):
                            has_null_issue = True
                            break
                
                if has_null_issue:
                    stats['by_db_count'][num_databases]['null_issues'] += 1
            
            # 统计数据库组合
            db_combo = tuple(sorted(databases))
            stats['db_combinations'].add(db_combo)
            
            # 统计SQL结构
            sql_upper = sql.upper()
            structure = []
            if 'JOIN' in sql_upper:
                structure.append('JOIN')
            if 'UNION' in sql_upper:
                structure.append('UNION')
            if 'GROUP BY' in sql_upper:
                structure.append('GROUP_BY')
            if 'ORDER BY' in sql_upper:
                structure.append('ORDER_BY')
            if any(func in sql_upper for func in ['COUNT', 'SUM', 'AVG', 'MAX', 'MIN']):
                structure.append('AGGREGATE')
            if 'WHERE' in sql_upper:
                structure.append('WHERE')
            
            structure_key = '+'.join(sorted(structure)) if structure else 'SIMPLE'
            stats['sql_structures'][structure_key] += 1
            
        except Exception as e:
            print(f"警告: 无法读取文件 {filename}: {e}")
            continue
    
    return stats

def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(script_dir)))
    
    final_dir = os.path.join(project_root, 'benchmark', 'data', 'beijing', 'output', 'cross_db_final')
    
    print("=" * 80)
    print("验证筛选后的SQL质量")
    print("=" * 80)
    print()
    
    stats = verify_filtered_sqls(final_dir)
    
    if not stats:
        return
    
    print("【统计结果】")
    print("-" * 80)
    for db_count in [2, 3, 4]:
        total = stats['by_db_count'][db_count]['total']
        with_results = stats['by_db_count'][db_count]['with_results']
        null_issues = stats['by_db_count'][db_count]['null_issues']
        print(f"{db_count}个数据库: 总数={total}, 有结果={with_results}, null问题={null_issues}")
    print()
    
    print("【数据库组合多样性】")
    print(f"不同的数据库组合数: {len(stats['db_combinations'])}")
    print()
    
    print("【SQL结构多样性】")
    print(f"不同的SQL结构数: {len(stats['sql_structures'])}")
    print("结构分布:")
    for structure, count in sorted(stats['sql_structures'].items(), key=lambda x: x[1], reverse=True):
        print(f"  {structure}: {count}")
    print()
    
    print("【质量评估】")
    total = stats['total']
    total_with_results = sum(stats['by_db_count'][db]['with_results'] for db in [2, 3, 4])
    total_null_issues = sum(stats['by_db_count'][db]['null_issues'] for db in [2, 3, 4])
    
    print(f"总SQL数: {total}")
    print(f"有结果的SQL: {total_with_results} ({total_with_results/total*100:.1f}%)")
    print(f"有null问题的SQL: {total_null_issues} ({total_null_issues/total*100:.1f}%)")
    print(f"数据库组合数: {len(stats['db_combinations'])}")
    print(f"SQL结构类型数: {len(stats['sql_structures'])}")
    print()

if __name__ == '__main__':
    main()



