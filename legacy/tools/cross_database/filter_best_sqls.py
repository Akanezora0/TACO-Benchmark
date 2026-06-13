#!/usr/bin/env python3
"""
Filter best cross-database SQL
Filter criteria:
1. SQL diversity (different database combinations, SQL structures, table combinations, etc.)
2. Result quality (avoid all-null results)
3. Result count (reasonable number of results)
"""

import os
import json
import shutil
from collections import defaultdict
from typing import Dict, List, Tuple
import re

# Target counts
TARGET_COUNTS = {
    2: 359,  # cross 2 databases
    3: 105,  # cross 3 databases
    4: 2     # cross 4 databases
}

def analyze_sql_structure(sql: str) -> Dict:
    """Analyze SQL structure features"""
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
    """Calculate result quality score
    Return score between 0-1; 1 means best quality
    """
    if not results or len(results) == 0:
        return 0.0
    
    # Check result count
    result_count = len(results)
    if result_count == 0:
        return 0.0
    
    # Calculate null value ratio
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
    
    # Quality score: more results is better, lower null ratio is better
    # Result count score (0-0.5): log(result_count + 1) / log(100) * 0.5, max 0.5 points
    import math
    count_score = min(0.5, math.log(result_count + 1) / math.log(100) * 0.5)
    
    # Null ratio score (0-0.5): (1 - null_ratio) * 0.5
    null_score = (1 - null_ratio) * 0.5
    
    quality_score = count_score + null_score
    
    return quality_score

def calculate_diversity_score(sql_data: Dict, existing_features: List[Dict], 
                            existing_db_combos: set, existing_tables: set) -> float:
    """Calculate SQL diversity score
    Return score between 0-1; 1 means most unique
    """
    diversity_score = 0.0
    
    # 1. Database combination diversity (0.3 points)
    databases = tuple(sorted(sql_data.get('databases', [])))
    if databases not in existing_db_combos:
        diversity_score += 0.3
    else:
        # If combination exists but table combination differs, give partial score
        diversity_score += 0.1
    
    # 2. Table combination diversity (0.3 points)
    table_mapping = sql_data.get('table_database_mapping', {})
    tables = tuple(sorted(table_mapping.keys()))
    if tables not in existing_tables:
        diversity_score += 0.3
    else:
        diversity_score += 0.05
    
    # 3. SQL structure diversity (0.4 points)
    sql = sql_data.get('sql', '')
    sql_features = analyze_sql_structure(sql)
    
    # Check difference from existing SQL structures
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
    """Calculate overall SQL score
    
    Returns:
        (total score, score details)
    """
    results = sql_data.get('results', [])
    
    # 1. Result quality score (0.5 points)
    quality_score = calculate_result_quality(results)
    
    # 2. Diversity score (0.5 points)
    diversity_score = calculate_diversity_score(sql_data, existing_features, 
                                                existing_db_combos, existing_tables)
    
    # Overall score
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
    """Calculate null ratio in results"""
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
    """Load all SQL files, classified by database count"""
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
                
                # If num_databases in metadata, prefer that
                metadata = data.get('metadata', {})
                if 'num_databases' in metadata:
                    num_databases = metadata['num_databases']
                
                if num_databases < 2 or num_databases > 4:
                    continue
                
                results = data.get('results', [])
                # Only keep SQL with results
                if results is not None and len(results) > 0:
                    data['_source_file'] = file_path
                    data['_source_type'] = sql_type
                    sqls_by_db_count[num_databases].append(data)
                    
            except Exception as e:
                print(f"Warning: cannot read file {filename}: {e}")
                continue
    
    return sqls_by_db_count

def filter_best_sqls(sqls: List[Dict], target_count: int) -> List[Dict]:
    """Filter best SQL"""
    if len(sqls) <= target_count:
        return sqls
    
    # Calculate score for each SQL
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
    
    # Sort by score
    scored_sqls.sort(key=lambda x: x['score'], reverse=True)
    
    # Select top target_count while ensuring diversity
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
        
        # Check if too many similar SQL already selected
        # If database and table combinations are same, select at most 2
        similar_count = sum(1 for s in selected_sqls 
                           if tuple(sorted(s.get('databases', []))) == databases and
                           tuple(sorted(s.get('table_database_mapping', {}).keys())) == tables)
        
        if similar_count < 2:  # Allow at most 2 similar ones
            selected_sqls.append(sql_data)
            selected_features.append(sql_features)
            selected_db_combos.add(databases)
            selected_tables.add(tables)
    
    # If target count not reached, continue adding
    if len(selected_sqls) < target_count:
        for item in scored_sqls:
            if len(selected_sqls) >= target_count:
                break
            if item['sql_data'] not in selected_sqls:
                selected_sqls.append(item['sql_data'])
    
    return selected_sqls[:target_count]

def main():
    # Get script directory and project root
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(script_dir)))
    
    # Define directory paths
    base_output_dir = os.path.join(project_root, 'benchmark', 'data', 'beijing', 'output')
    
    # Input directories
    join_dir = os.path.join(base_output_dir, 'cross_db_single_join')
    join_backup_dir = os.path.join(base_output_dir, 'cross_db_single_join_backup_51')
    union_dir = os.path.join(base_output_dir, 'cross_db_single_union_version')
    
    # Output directory
    final_output_dir = os.path.join(base_output_dir, 'cross_db_final')
    os.makedirs(final_output_dir, exist_ok=True)
    
    print("=" * 80)
    print("Filter best cross-database SQL")
    print("=" * 80)
    print()
    
    # Load all SQL
    print("Loading all SQL files...")
    sqls_by_db_count = load_all_sqls(join_dir, join_backup_dir, union_dir)
    
    for db_count in [2, 3, 4]:
        print(f"  {db_count} databases: {len(sqls_by_db_count[db_count])}")
    print()
    
    # Filter best SQL
    all_selected_sqls = []
    
    for db_count in [2, 3, 4]:
        target = TARGET_COUNTS[db_count]
        sqls = sqls_by_db_count[db_count]
        
        print(f"[{db_count} databases]")
        print(f"  Candidate count: {len(sqls)}")
        print(f"  Target count: {target}")
        
        if len(sqls) == 0:
            print(f"  Warning: no available SQL")
            continue
        
        if len(sqls) <= target:
            print(f"  Count satisfied, select all")
            selected_sqls = sqls
        else:
            print(f"  Start filtering best {target}...")
            selected_sqls = filter_best_sqls(sqls, target)
            print(f"  Filtered {len(selected_sqls)} best SQL")
        
        all_selected_sqls.extend(selected_sqls)
        print()
    
    # Save filter results
    print("=" * 80)
    print("Save filter results")
    print("=" * 80)
    print()
    
    # Save separately by database count
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
            # Remove temporary fields
            sql_data_copy = sql_data.copy()
            sql_data_copy.pop('_source_file', None)
            sql_data_copy.pop('_source_type', None)
            
            output_file = os.path.join(final_output_dir, f'cross_db_generated_sql_{file_counter}.json')
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(sql_data_copy, f, ensure_ascii=False, indent=2)
            file_counter += 1
    
    print(f"Saved {len(all_selected_sqls)} best SQL to: {final_output_dir}")
    print()
    
    # Statistics
    print("=" * 80)
    print("Filter results statistics")
    print("=" * 80)
    print()
    print(f"{'Database count':<15} {'Target':<10} {'Filtered':<10}")
    print("-" * 40)
    for db_count in [2, 3, 4]:
        target = TARGET_COUNTS[db_count]
        selected_count = len(sqls_by_db_count_selected[db_count])
        print(f"{db_count} databases{'':<6} {target:<10} {selected_count:<10}")
    print("-" * 40)
    print(f"{'Total':<15} {sum(TARGET_COUNTS.values()):<10} {len(all_selected_sqls):<10}")
    print()

if __name__ == '__main__':
    main()
