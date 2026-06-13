#!/usr/bin/env python3
"""
Analyze joinable table pairs across databases

Efficient strategy:
1. Check only semantically related database combinations (from database_combinations.json)
2. Build a column-name index (grouped by keywords)
3. Match columns by name similarity (keyword matching, semantic similarity)
4. Prioritize common JOIN column types (ID, name, code, etc.)
"""

import json
import os
import re
from collections import defaultdict
from tqdm import tqdm
import argparse

# Common JOIN column keywords (sorted by priority)
JOIN_KEYWORDS = {
    'id': ['id', 'ID', 'Id', '编号', '序号', '标识', '标识符'],
    'code': ['代码', '编码', 'code', 'Code', 'CODE', '统一社会信用代码', '组织机构代码', '信用代码'],
    'name': ['名称', 'name', 'Name', 'NAME', '企业名称', '机构名称', '单位名称', '公司名称', '单位', '机构'],
    'number': ['号码', 'number', 'Number', 'NUMBER', '电话', '手机', '联系电话'],
    'date': ['日期', 'date', 'Date', 'DATE', '时间', 'time', 'Time', 'TIME', '年月', '年份'],
    'type': ['类型', 'type', 'Type', 'TYPE', '类别', '分类', '种类'],
    'status': ['状态', 'status', 'Status', 'STATUS', '状态码'],
    'area': ['地区', '区域', 'area', 'Area', 'AREA', '区', '市', '省', '县', '街道', '地址'],
}

def load_schema(schema_file):
    """Load database schema."""
    try:
        with open(schema_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        if 'tables' in data:
            return data
        return None
    except:
        return None

def extract_column_keywords(column_name):
    """Extract keywords from a column name."""
    keywords = set()
    column_lower = column_name.lower()
    
    # Check for common JOIN keywords
    for keyword_type, patterns in JOIN_KEYWORDS.items():
        for pattern in patterns:
            if pattern.lower() in column_name or pattern in column_name:
                keywords.add(keyword_type)
                break
    
    # Extract digits (may be part of an ID)
    if re.search(r'\d+', column_name):
        keywords.add('has_number')
    
    # Extract Chinese keywords (common Chinese terms)
    chinese_keywords = ['企业', '机构', '单位', '公司', '组织', '部门', '人员', '用户', '客户']
    for kw in chinese_keywords:
        if kw in column_name:
            keywords.add(f'chinese_{kw}')
    
    return keywords

def build_column_index(schemas):
    """Build a column-name index for all databases."""
    # Index structure: {database_name: {table_name: {column_name: {keywords: set, data_type: str}}}}
    index = defaultdict(lambda: defaultdict(dict))
    
    for db_name, schema in schemas.items():
        if schema is None:
            continue
        
        for table_info in schema.get('tables', []):
            table_name = table_info.get('table_name', '')
            for col_info in table_info.get('columns', []):
                col_name = col_info.get('column_name', '')
                data_type = col_info.get('data_type', 'TEXT')
                
                keywords = extract_column_keywords(col_name)
                index[db_name][table_name][col_name] = {
                    'keywords': keywords,
                    'data_type': data_type
                }
    
    return index

def calculate_column_similarity(col1_info, col2_info):
    """Calculate similarity score between two columns."""
    keywords1 = col1_info['keywords']
    keywords2 = col2_info['keywords']
    data_type1 = col1_info['data_type']
    data_type2 = col2_info['data_type']
    
    score = 0.0
    
    # 1. Keyword matching (highest weight)
    common_keywords = keywords1 & keywords2
    if 'id' in common_keywords:
        score += 10.0  # ID match has highest weight
    elif 'code' in common_keywords:
        score += 8.0   # Code match has very high weight
    elif 'name' in common_keywords:
        score += 6.0   # Name match has high weight
    elif 'number' in common_keywords:
        score += 5.0
    elif 'date' in common_keywords:
        score += 4.0
    elif 'type' in common_keywords or 'status' in common_keywords:
        score += 3.0
    elif 'area' in common_keywords:
        score += 3.0
    else:
        # Other keyword matches
        score += len(common_keywords) * 1.0
    
    # 2. Data type matching (bonus if types match)
    if data_type1 == data_type2 and data_type1 != 'TEXT':
        score += 2.0
    
    # 3. Chinese keyword matching (semantic similarity)
    chinese_kw1 = {kw for kw in keywords1 if kw.startswith('chinese_')}
    chinese_kw2 = {kw for kw in keywords2 if kw.startswith('chinese_')}
    if chinese_kw1 & chinese_kw2:
        score += 2.0
    
    return score

def find_joinable_table_pairs(index, db1_name, db2_name, min_similarity=5.0):
    """Find joinable table pairs between two databases."""
    joinable_pairs = []
    
    tables1 = index.get(db1_name, {})
    tables2 = index.get(db2_name, {})
    
    if not tables1 or not tables2:
        return joinable_pairs
    
    # Iterate over all table pairs
    for table1_name, cols1 in tables1.items():
        for table2_name, cols2 in tables2.items():
            # Find all possible column pairs
            column_pairs = []
            for col1_name, col1_info in cols1.items():
                for col2_name, col2_info in cols2.items():
                    similarity = calculate_column_similarity(col1_info, col2_info)
                    if similarity >= min_similarity:
                        column_pairs.append({
                            'col1': col1_name,
                            'col2': col2_name,
                            'similarity': similarity,
                            'keywords1': list(col1_info['keywords']),  # Convert to list for JSON serialization
                            'keywords2': list(col2_info['keywords'])   # Convert to list for JSON serialization
                        })
            
            # If joinable column pairs exist, record this table pair
            if column_pairs:
                # Sort by similarity and keep top 3 column pairs
                column_pairs.sort(key=lambda x: x['similarity'], reverse=True)
                joinable_pairs.append({
                    'db1': db1_name,
                    'db2': db2_name,
                    'table1': table1_name,
                    'table2': table2_name,
                    'column_pairs': column_pairs[:3],  # Keep only top 3 column pairs
                    'best_similarity': column_pairs[0]['similarity'],
                    'num_columns1': len(cols1),
                    'num_columns2': len(cols2)
                })
    
    return joinable_pairs

def analyze_all_combinations(database_dir, combinations_file, output_file, min_similarity=5.0):
    """Analyze all semantically related database combinations."""
    print("=" * 70)
    print("Analyze joinable table pairs across databases")
    print("=" * 70)
    
    # 1. Load database combinations
    print("\n1. Loading database combinations...")
    with open(combinations_file, 'r', encoding='utf-8') as f:
        combinations_data = json.load(f)
    
    all_combinations = []
    all_combinations.extend(combinations_data.get('2db_combinations', []))
    all_combinations.extend(combinations_data.get('3db_combinations', []))
    all_combinations.extend(combinations_data.get('4db_combinations', []))
    
    # Collect all involved databases
    all_databases = set()
    for combo in all_combinations:
        all_databases.update(combo)
    
    print(f"  Database combinations: {len(all_combinations)}")
    print(f"  Databases involved: {len(all_databases)}")
    
    # 2. Load schemas for all databases
    print("\n2. Loading database schemas...")
    schemas = {}
    for db_name in tqdm(sorted(all_databases), desc="Loading schemas"):
        schema_file = os.path.join(database_dir, db_name, f"{db_name}.json")
        if os.path.exists(schema_file):
            schema = load_schema(schema_file)
            if schema:
                schemas[db_name] = schema
        else:
            print(f"  Warning: schema file not found: {schema_file}")
    
    print(f"  Successfully loaded schemas for {len(schemas)} databases")
    
    # 3. Build column-name index
    print("\n3. Building column-name index...")
    index = build_column_index(schemas)
    
    total_tables = sum(len(tables) for tables in index.values())
    total_columns = sum(
        len(cols) 
        for db_tables in index.values() 
        for cols in db_tables.values()
    )
    print(f"  Total tables: {total_tables}")
    print(f"  Total columns: {total_columns}")
    
    # 4. Analyze each database combination
    print("\n4. Analyzing joinable table pairs...")
    all_joinable_pairs = []
    
    # For 2-database combinations, analyze directly
    for combo in tqdm(all_combinations, desc="Analyzing combinations"):
        if len(combo) == 2:
            db1, db2 = combo
            pairs = find_joinable_table_pairs(index, db1, db2, min_similarity)
            all_joinable_pairs.extend(pairs)
        elif len(combo) >= 3:
            # For 3+ database combinations, analyze all 2-database sub-combinations
            for i in range(len(combo)):
                for j in range(i + 1, len(combo)):
                    db1, db2 = combo[i], combo[j]
                    pairs = find_joinable_table_pairs(index, db1, db2, min_similarity)
                    all_joinable_pairs.extend(pairs)
    
    # Deduplicate (keep each table pair only once)
    seen_pairs = set()
    unique_pairs = []
    for pair in all_joinable_pairs:
        key = (pair['db1'], pair['db2'], pair['table1'], pair['table2'])
        if key not in seen_pairs:
            seen_pairs.add(key)
            unique_pairs.append(pair)
    
    print(f"\n  Found {len(unique_pairs)} joinable table pairs")
    
    # 5. Sort by similarity
    unique_pairs.sort(key=lambda x: x['best_similarity'], reverse=True)
    
    # 6. Statistics
    print("\n5. Statistics...")
    by_db_combo = defaultdict(int)
    by_similarity = defaultdict(int)
    by_keyword_type = defaultdict(int)
    
    for pair in unique_pairs:
        key = f"{pair['db1']} + {pair['db2']}"
        by_db_combo[key] += 1
        
        similarity_range = int(pair['best_similarity'] // 2) * 2
        by_similarity[similarity_range] += 1
        
        # Count keyword types
        if pair['column_pairs']:
            best_pair = pair['column_pairs'][0]
            keywords1 = set(best_pair['keywords1'])  # Convert to set
            keywords2 = set(best_pair['keywords2'])  # Convert to set
            common = keywords1 & keywords2
            if 'id' in common:
                by_keyword_type['ID'] += 1
            elif 'code' in common:
                by_keyword_type['Code'] += 1
            elif 'name' in common:
                by_keyword_type['Name'] += 1
            else:
                by_keyword_type['Other'] += 1
    
    print(f"\n  Distribution by database combination (top 10):")
    for combo, count in sorted(by_db_combo.items(), key=lambda x: x[1], reverse=True)[:10]:
        print(f"    {combo}: {count} table pairs")
    
    print(f"\n  Distribution by similarity:")
    for sim_range in sorted(by_similarity.keys(), reverse=True):
        print(f"    {sim_range}-{sim_range+2}: {by_similarity[sim_range]} table pairs")
    
    print(f"\n  Distribution by keyword type:")
    for kw_type, count in sorted(by_keyword_type.items(), key=lambda x: x[1], reverse=True):
        print(f"    {kw_type}: {count} table pairs")
    
    # 7. Save results
    print("\n6. Saving results...")
    result = {
        'total_pairs': len(unique_pairs),
        'min_similarity': min_similarity,
        'joinable_pairs': unique_pairs,
        'statistics': {
            'by_db_combo': dict(by_db_combo),
            'by_similarity': dict(by_similarity),
            'by_keyword_type': dict(by_keyword_type)
        }
    }
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    
    print(f"  Results saved to: {output_file}")
    
    # 8. Show top 20 best table pairs
    print("\n7. Top 20 best joinable table pairs:")
    for i, pair in enumerate(unique_pairs[:20], 1):
        best_col_pair = pair['column_pairs'][0]
        print(f"\n  {i}. {pair['db1']}.{pair['table1']} <-> {pair['db2']}.{pair['table2']}")
        print(f"     Similarity: {pair['best_similarity']:.1f}")
        print(f"     Best column pair: {best_col_pair['col1']} <-> {best_col_pair['col2']}")
        keywords1_set = set(best_col_pair['keywords1'])
        keywords2_set = set(best_col_pair['keywords2'])
        common_keywords = keywords1_set & keywords2_set
        print(f"     Common keywords: {list(common_keywords)}")
    
    print("\n" + "=" * 70)
    print("Analysis complete!")
    print("=" * 70)
    
    return result

def main():
    parser = argparse.ArgumentParser(description='Analyze joinable table pairs across databases')
    parser.add_argument('--database_dir', type=str,
                       default='benchmark/data/beijing/database_chinese',
                       help='Database directory')
    parser.add_argument('--combinations_file', type=str,
                       default='benchmark/generation/cross_database/database_combinations.json',
                       help='Database combinations file')
    parser.add_argument('--output_file', type=str,
                       default='benchmark/generation/cross_database/joinable_table_pairs.json',
                       help='Output file')
    parser.add_argument('--min_similarity', type=float, default=5.0,
                       help='Minimum similarity threshold (default 5.0)')
    
    args = parser.parse_args()
    
    analyze_all_combinations(
        args.database_dir,
        args.combinations_file,
        args.output_file,
        args.min_similarity
    )

if __name__ == '__main__':
    main()
