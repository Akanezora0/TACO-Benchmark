#!/usr/bin/env python3
"""
筛选最优的跨数据库SQL
筛选标准：
1. SQL多样性（不同的数据库组合、SQL结构、表组合等）
2. 结果质量（避免全是null的结果）
3. 结果数量（有合理数量的结果）
"""

import os
import json
import shutil
from collections import defaultdict
from typing import Dict, List, Tuple
import re

# 目标数量
TARGET_COUNTS = {
    2: 359,  # 跨2个数据库
    3: 105,  # 跨3个数据库
    4: 2     # 跨4个数据库
}

def analyze_sql_structure(sql: str) -> Dict:
    """分析SQL结构特征"""
    sql_upper = sql.upper()
    
    features = {
        'has_join': 'JOIN' in sql_upper,
        'has_union': 'UNION' in sql_upper,
        'has_group_by': 'GROUP BY' in sql_upper,
        'has_order_by': 'ORDER BY' in sql_upper,
        'has_aggregate': any(func in sql_upper for func in ['COUNT', 'SUM', 'AVG', 'MAX', 'MIN']),
        'has_where': 'WHERE' in sql_upper,
        'has_having': 'HAVING' in sql_upper,
        'has_limit': 'LIMIT' in sql_upper,
        'has_subquery': bool(re.search(r'\([^)]*SELECT[^)]*\)', sql_upper)),
        'join_count': len(re.findall(r'\bJOIN\b', sql_upper)),
        'union_count': len(re.findall(r'\bUNION\b', sql_upper)),
    }
    
    return features

def calculate_result_quality(results: List) -> float:
    """计算结果质量分数
    返回0-1之间的分数，1表示质量最好
    """
    if not results or len(results) == 0:
        return 0.0
    
    # 检查结果数量
    result_count = len(results)
    if result_count == 0:
        return 0.0
    
    # 计算null值的比例
    null_count = 0
    total_cells = 0
    
    for row in results:
        if isinstance(row, (list, tuple)):
            for cell in row:
                total_cells += 1
                if cell is None or (isinstance(cell, str) and cell.upper() in ['NULL', 'NONE', '']):
                    null_count += 1
        else:
            total_cells += 1
            if row is None or (isinstance(row, str) and row.upper() in ['NULL', 'NONE', '']):
                null_count += 1
    
    if total_cells == 0:
        return 0.0
    
    null_ratio = null_count / total_cells
    
    # 质量分数：结果数量越多越好，null比例越低越好
    # 结果数量分数（0-0.5）：log(result_count + 1) / log(100) * 0.5，最多0.5分
    import math
    count_score = min(0.5, math.log(result_count + 1) / math.log(100) * 0.5)
    
    # null比例分数（0-0.5）：(1 - null_ratio) * 0.5
    null_score = (1 - null_ratio) * 0.5
    
    quality_score = count_score + null_score
    
    return quality_score

def calculate_diversity_score(sql_data: Dict, existing_features: List[Dict], 
                            existing_db_combos: set, existing_tables: set) -> float:
    """计算SQL的多样性分数
    返回0-1之间的分数，1表示最独特
    """
    diversity_score = 0.0
    
    # 1. 数据库组合多样性（0.3分）
    databases = tuple(sorted(sql_data.get('databases', [])))
    if databases not in existing_db_combos:
        diversity_score += 0.3
    else:
        # 如果组合已存在，但表组合不同，给部分分数
        diversity_score += 0.1
    
    # 2. 表组合多样性（0.3分）
    table_mapping = sql_data.get('table_database_mapping', {})
    tables = tuple(sorted(table_mapping.keys()))
    if tables not in existing_tables:
        diversity_score += 0.3
    else:
        diversity_score += 0.05
    
    # 3. SQL结构多样性（0.4分）
    sql = sql_data.get('sql', '')
    sql_features = analyze_sql_structure(sql)
    
    # 检查与已有SQL结构的差异
    max_structure_similarity = 0.0
    for existing_feat in existing_features:
        similarity = 0.0
        for key in sql_features:
            if existing_feat.get(key) == sql_features[key]:
                similarity += 1.0 / len(sql_features)
        max_structure_similarity = max(max_structure_similarity, similarity)
    
    structure_diversity = 1.0 - max_structure_similarity
    diversity_score += structure_diversity * 0.4
    
    return min(1.0, diversity_score)

def score_sql(sql_data: Dict, existing_features: List[Dict], 
              existing_db_combos: set, existing_tables: set) -> Tuple[float, Dict]:
    """计算SQL的综合分数
    
    Returns:
        (总分, 分数详情)
    """
    results = sql_data.get('results', [])
    
    # 1. 结果质量分数（0.5分）
    quality_score = calculate_result_quality(results)
    
    # 2. 多样性分数（0.5分）
    diversity_score = calculate_diversity_score(sql_data, existing_features, 
                                                existing_db_combos, existing_tables)
    
    # 综合分数
    total_score = quality_score * 0.5 + diversity_score * 0.5
    
    score_details = {
        'quality_score': quality_score,
        'diversity_score': diversity_score,
        'total_score': total_score,
        'result_count': len(results) if results else 0,
        'null_ratio': calculate_null_ratio(results)
    }
    
    return total_score, score_details

def calculate_null_ratio(results: List) -> float:
    """计算结果中null的比例"""
    if not results or len(results) == 0:
        return 1.0
    
    null_count = 0
    total_cells = 0
    
    for row in results:
        if isinstance(row, (list, tuple)):
            for cell in row:
                total_cells += 1
                if cell is None or (isinstance(cell, str) and cell.upper() in ['NULL', 'NONE', '']):
                    null_count += 1
        else:
            total_cells += 1
            if row is None or (isinstance(row, str) and row.upper() in ['NULL', 'NONE', '']):
                null_count += 1
    
    if total_cells == 0:
        return 1.0
    
    return null_count / total_cells

def load_all_sqls(join_dir: str, join_backup_dir: str, union_dir: str) -> Dict[int, List[Dict]]:
    """加载所有SQL文件，按数据库数量分类"""
    sqls_by_db_count = defaultdict(list)
    
    directories = [
        (join_dir, 'join'),
        (join_backup_dir, 'join_backup'),
        (union_dir, 'union')
    ]
    
    for sql_dir, sql_type in directories:
        if not os.path.exists(sql_dir):
            continue
        
        for filename in os.listdir(sql_dir):
            if not filename.startswith('cross_db_generated_sql_') or not filename.endswith('.json'):
                continue
            
            file_path = os.path.join(sql_dir, filename)
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                databases = data.get('databases', [])
                num_databases = len(databases)
                
                # 如果metadata中有num_databases，优先使用
                metadata = data.get('metadata', {})
                if 'num_databases' in metadata:
                    num_databases = metadata['num_databases']
                
                if num_databases < 2 or num_databases > 4:
                    continue
                
                results = data.get('results', [])
                # 只保留有结果的SQL
                if results is not None and len(results) > 0:
                    data['_source_file'] = file_path
                    data['_source_type'] = sql_type
                    sqls_by_db_count[num_databases].append(data)
                    
            except Exception as e:
                print(f"警告: 无法读取文件 {filename}: {e}")
                continue
    
    return sqls_by_db_count

def filter_best_sqls(sqls: List[Dict], target_count: int) -> List[Dict]:
    """筛选最优的SQL"""
    if len(sqls) <= target_count:
        return sqls
    
    # 计算每个SQL的分数
    scored_sqls = []
    existing_features = []
    existing_db_combos = set()
    existing_tables = set()
    
    for sql_data in sqls:
        score, score_details = score_sql(sql_data, existing_features, 
                                         existing_db_combos, existing_tables)
        
        scored_sqls.append({
            'sql_data': sql_data,
            'score': score,
            'score_details': score_details
        })
    
    # 按分数排序
    scored_sqls.sort(key=lambda x: x['score'], reverse=True)
    
    # 选择前target_count个，同时确保多样性
    selected_sqls = []
    selected_features = []
    selected_db_combos = set()
    selected_tables = set()
    
    for item in scored_sqls:
        if len(selected_sqls) >= target_count:
            break
        
        sql_data = item['sql_data']
        databases = tuple(sorted(sql_data.get('databases', [])))
        table_mapping = sql_data.get('table_database_mapping', {})
        tables = tuple(sorted(table_mapping.keys()))
        sql = sql_data.get('sql', '')
        sql_features = analyze_sql_structure(sql)
        
        # 检查是否已经选择了太多相似的SQL
        # 如果数据库组合和表组合都相同，最多选择2个
        similar_count = sum(1 for s in selected_sqls 
                           if tuple(sorted(s.get('databases', []))) == databases and
                           tuple(sorted(s.get('table_database_mapping', {}).keys())) == tables)
        
        if similar_count < 2:  # 允许最多2个相似的
            selected_sqls.append(sql_data)
            selected_features.append(sql_features)
            selected_db_combos.add(databases)
            selected_tables.add(tables)
    
    # 如果还没达到目标数量，继续添加
    if len(selected_sqls) < target_count:
        for item in scored_sqls:
            if len(selected_sqls) >= target_count:
                break
            if item['sql_data'] not in selected_sqls:
                selected_sqls.append(item['sql_data'])
    
    return selected_sqls[:target_count]

def main():
    # 获取脚本所在目录和项目根目录
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(script_dir)))
    
    # 定义目录路径
    base_output_dir = os.path.join(project_root, 'benchmark', 'data', 'beijing', 'output')
    
    # 输入目录
    join_dir = os.path.join(base_output_dir, 'cross_db_single_join')
    join_backup_dir = os.path.join(base_output_dir, 'cross_db_single_join_backup_51')
    union_dir = os.path.join(base_output_dir, 'cross_db_single_union_version')
    
    # 输出目录
    final_output_dir = os.path.join(base_output_dir, 'cross_db_final')
    os.makedirs(final_output_dir, exist_ok=True)
    
    print("=" * 80)
    print("筛选最优跨数据库SQL")
    print("=" * 80)
    print()
    
    # 加载所有SQL
    print("📊 加载所有SQL文件...")
    sqls_by_db_count = load_all_sqls(join_dir, join_backup_dir, union_dir)
    
    for db_count in [2, 3, 4]:
        print(f"  {db_count}个数据库: {len(sqls_by_db_count[db_count])} 个")
    print()
    
    # 筛选最优SQL
    all_selected_sqls = []
    
    for db_count in [2, 3, 4]:
        target = TARGET_COUNTS[db_count]
        sqls = sqls_by_db_count[db_count]
        
        print(f"【{db_count}个数据库】")
        print(f"  候选数量: {len(sqls)}")
        print(f"  目标数量: {target}")
        
        if len(sqls) == 0:
            print(f"  ⚠️  没有可用的SQL")
            continue
        
        if len(sqls) <= target:
            print(f"  ✅ 数量已满足，全部选择")
            selected_sqls = sqls
        else:
            print(f"  🔍 开始筛选最优的 {target} 个...")
            selected_sqls = filter_best_sqls(sqls, target)
            print(f"  ✅ 已筛选出 {len(selected_sqls)} 个最优SQL")
        
        all_selected_sqls.extend(selected_sqls)
        print()
    
    # 保存筛选结果
    print("=" * 80)
    print("保存筛选结果")
    print("=" * 80)
    print()
    
    # 按数据库数量分别保存
    sqls_by_db_count_selected = defaultdict(list)
    for sql_data in all_selected_sqls:
        databases = sql_data.get('databases', [])
        num_databases = len(databases)
        metadata = sql_data.get('metadata', {})
        if 'num_databases' in metadata:
            num_databases = metadata['num_databases']
        sqls_by_db_count_selected[num_databases].append(sql_data)
    
    file_counter = 0
    for db_count in [2, 3, 4]:
        sqls = sqls_by_db_count_selected[db_count]
        for sql_data in sqls:
            # 移除临时字段
            sql_data_copy = sql_data.copy()
            sql_data_copy.pop('_source_file', None)
            sql_data_copy.pop('_source_type', None)
            
            output_file = os.path.join(final_output_dir, f'cross_db_generated_sql_{file_counter}.json')
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(sql_data_copy, f, ensure_ascii=False, indent=2)
            file_counter += 1
    
    print(f"✅ 已保存 {len(all_selected_sqls)} 个最优SQL到: {final_output_dir}")
    print()
    
    # 统计信息
    print("=" * 80)
    print("筛选结果统计")
    print("=" * 80)
    print()
    print(f"{'数据库数量':<15} {'目标':<10} {'筛选结果':<10}")
    print("-" * 40)
    for db_count in [2, 3, 4]:
        target = TARGET_COUNTS[db_count]
        selected_count = len(sqls_by_db_count_selected[db_count])
        print(f"{db_count}个数据库{'':<6} {target:<10} {selected_count:<10}")
    print("-" * 40)
    print(f"{'总计':<15} {sum(TARGET_COUNTS.values()):<10} {len(all_selected_sqls):<10}")
    print()

if __name__ == '__main__':
    main()



