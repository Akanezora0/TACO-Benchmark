#!/usr/bin/env python3
"""
分析跨数据库可JOIN的表对

高效策略：
1. 只检查语义相关的数据库组合（根据database_combinations.json）
2. 建立列名索引（按关键词分类）
3. 使用列名相似性匹配（关键词匹配、语义相似）
4. 优先考虑常见的JOIN列类型（ID、名称、代码等）
"""

import json
import os
import re
from collections import defaultdict
from tqdm import tqdm
import argparse

# 常见的JOIN列关键词（按优先级排序）
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
    """加载数据库schema"""
    try:
        with open(schema_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        if 'tables' in data:
            return data
        return None
    except:
        return None

def extract_column_keywords(column_name):
    """提取列名的关键词"""
    keywords = set()
    column_lower = column_name.lower()
    
    # 检查是否包含常见JOIN关键词
    for keyword_type, patterns in JOIN_KEYWORDS.items():
        for pattern in patterns:
            if pattern.lower() in column_name or pattern in column_name:
                keywords.add(keyword_type)
                break
    
    # 提取数字（可能是ID的一部分）
    if re.search(r'\d+', column_name):
        keywords.add('has_number')
    
    # 提取中文关键词（常见的中文词汇）
    chinese_keywords = ['企业', '机构', '单位', '公司', '组织', '部门', '人员', '用户', '客户']
    for kw in chinese_keywords:
        if kw in column_name:
            keywords.add(f'chinese_{kw}')
    
    return keywords

def build_column_index(schemas):
    """为所有数据库建立列名索引"""
    # 索引结构：{数据库名: {表名: {列名: {keywords: set, data_type: str}}}}
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
    """计算两个列的相似度分数"""
    keywords1 = col1_info['keywords']
    keywords2 = col2_info['keywords']
    data_type1 = col1_info['data_type']
    data_type2 = col2_info['data_type']
    
    score = 0.0
    
    # 1. 关键词匹配（权重最高）
    common_keywords = keywords1 & keywords2
    if 'id' in common_keywords:
        score += 10.0  # ID匹配权重最高
    elif 'code' in common_keywords:
        score += 8.0   # 代码匹配权重很高
    elif 'name' in common_keywords:
        score += 6.0   # 名称匹配权重较高
    elif 'number' in common_keywords:
        score += 5.0
    elif 'date' in common_keywords:
        score += 4.0
    elif 'type' in common_keywords or 'status' in common_keywords:
        score += 3.0
    elif 'area' in common_keywords:
        score += 3.0
    else:
        # 其他关键词匹配
        score += len(common_keywords) * 1.0
    
    # 2. 数据类型匹配（如果类型相同，加分）
    if data_type1 == data_type2 and data_type1 != 'TEXT':
        score += 2.0
    
    # 3. 中文关键词匹配（语义相似）
    chinese_kw1 = {kw for kw in keywords1 if kw.startswith('chinese_')}
    chinese_kw2 = {kw for kw in keywords2 if kw.startswith('chinese_')}
    if chinese_kw1 & chinese_kw2:
        score += 2.0
    
    return score

def find_joinable_table_pairs(index, db1_name, db2_name, min_similarity=5.0):
    """找出两个数据库之间可JOIN的表对"""
    joinable_pairs = []
    
    tables1 = index.get(db1_name, {})
    tables2 = index.get(db2_name, {})
    
    if not tables1 or not tables2:
        return joinable_pairs
    
    # 遍历所有表对
    for table1_name, cols1 in tables1.items():
        for table2_name, cols2 in tables2.items():
            # 找出所有可能的列对
            column_pairs = []
            for col1_name, col1_info in cols1.items():
                for col2_name, col2_info in cols2.items():
                    similarity = calculate_column_similarity(col1_info, col2_info)
                    if similarity >= min_similarity:
                        column_pairs.append({
                            'col1': col1_name,
                            'col2': col2_name,
                            'similarity': similarity,
                            'keywords1': list(col1_info['keywords']),  # 转换为list以便JSON序列化
                            'keywords2': list(col2_info['keywords'])   # 转换为list以便JSON序列化
                        })
            
            # 如果找到可JOIN的列对，记录这个表对
            if column_pairs:
                # 按相似度排序，取前3个最好的列对
                column_pairs.sort(key=lambda x: x['similarity'], reverse=True)
                joinable_pairs.append({
                    'db1': db1_name,
                    'db2': db2_name,
                    'table1': table1_name,
                    'table2': table2_name,
                    'column_pairs': column_pairs[:3],  # 只保留前3个最好的列对
                    'best_similarity': column_pairs[0]['similarity'],
                    'num_columns1': len(cols1),
                    'num_columns2': len(cols2)
                })
    
    return joinable_pairs

def analyze_all_combinations(database_dir, combinations_file, output_file, min_similarity=5.0):
    """分析所有语义相关的数据库组合"""
    print("=" * 70)
    print("分析跨数据库可JOIN的表对")
    print("=" * 70)
    
    # 1. 加载数据库组合
    print("\n1. 加载数据库组合...")
    with open(combinations_file, 'r', encoding='utf-8') as f:
        combinations_data = json.load(f)
    
    all_combinations = []
    all_combinations.extend(combinations_data.get('2db_combinations', []))
    all_combinations.extend(combinations_data.get('3db_combinations', []))
    all_combinations.extend(combinations_data.get('4db_combinations', []))
    
    # 提取所有涉及的数据库
    all_databases = set()
    for combo in all_combinations:
        all_databases.update(combo)
    
    print(f"  数据库组合数: {len(all_combinations)}")
    print(f"  涉及数据库数: {len(all_databases)}")
    
    # 2. 加载所有数据库的schema
    print("\n2. 加载数据库schema...")
    schemas = {}
    for db_name in tqdm(sorted(all_databases), desc="加载schema"):
        schema_file = os.path.join(database_dir, db_name, f"{db_name}.json")
        if os.path.exists(schema_file):
            schema = load_schema(schema_file)
            if schema:
                schemas[db_name] = schema
        else:
            print(f"  警告: 找不到schema文件 {schema_file}")
    
    print(f"  成功加载 {len(schemas)} 个数据库的schema")
    
    # 3. 建立列名索引
    print("\n3. 建立列名索引...")
    index = build_column_index(schemas)
    
    total_tables = sum(len(tables) for tables in index.values())
    total_columns = sum(
        len(cols) 
        for db_tables in index.values() 
        for cols in db_tables.values()
    )
    print(f"  总表数: {total_tables}")
    print(f"  总列数: {total_columns}")
    
    # 4. 分析每个数据库组合
    print("\n4. 分析可JOIN的表对...")
    all_joinable_pairs = []
    
    # 对于2数据库组合，直接分析
    for combo in tqdm(all_combinations, desc="分析组合"):
        if len(combo) == 2:
            db1, db2 = combo
            pairs = find_joinable_table_pairs(index, db1, db2, min_similarity)
            all_joinable_pairs.extend(pairs)
        elif len(combo) >= 3:
            # 对于3+数据库组合，分析所有2数据库子组合
            for i in range(len(combo)):
                for j in range(i + 1, len(combo)):
                    db1, db2 = combo[i], combo[j]
                    pairs = find_joinable_table_pairs(index, db1, db2, min_similarity)
                    all_joinable_pairs.extend(pairs)
    
    # 去重（相同的表对只保留一次）
    seen_pairs = set()
    unique_pairs = []
    for pair in all_joinable_pairs:
        key = (pair['db1'], pair['db2'], pair['table1'], pair['table2'])
        if key not in seen_pairs:
            seen_pairs.add(key)
            unique_pairs.append(pair)
    
    print(f"\n  找到 {len(unique_pairs)} 个可JOIN的表对")
    
    # 5. 按相似度排序
    unique_pairs.sort(key=lambda x: x['best_similarity'], reverse=True)
    
    # 6. 统计信息
    print("\n5. 统计信息...")
    by_db_combo = defaultdict(int)
    by_similarity = defaultdict(int)
    by_keyword_type = defaultdict(int)
    
    for pair in unique_pairs:
        key = f"{pair['db1']} + {pair['db2']}"
        by_db_combo[key] += 1
        
        similarity_range = int(pair['best_similarity'] // 2) * 2
        by_similarity[similarity_range] += 1
        
        # 统计关键词类型
        if pair['column_pairs']:
            best_pair = pair['column_pairs'][0]
            keywords1 = set(best_pair['keywords1'])  # 转换为set
            keywords2 = set(best_pair['keywords2'])  # 转换为set
            common = keywords1 & keywords2
            if 'id' in common:
                by_keyword_type['ID'] += 1
            elif 'code' in common:
                by_keyword_type['代码'] += 1
            elif 'name' in common:
                by_keyword_type['名称'] += 1
            else:
                by_keyword_type['其他'] += 1
    
    print(f"\n  按数据库组合分布（前10个）:")
    for combo, count in sorted(by_db_combo.items(), key=lambda x: x[1], reverse=True)[:10]:
        print(f"    {combo}: {count} 个表对")
    
    print(f"\n  按相似度分布:")
    for sim_range in sorted(by_similarity.keys(), reverse=True):
        print(f"    {sim_range}-{sim_range+2}: {by_similarity[sim_range]} 个表对")
    
    print(f"\n  按关键词类型分布:")
    for kw_type, count in sorted(by_keyword_type.items(), key=lambda x: x[1], reverse=True):
        print(f"    {kw_type}: {count} 个表对")
    
    # 7. 保存结果
    print("\n6. 保存结果...")
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
    
    print(f"  结果已保存到: {output_file}")
    
    # 8. 显示前20个最好的表对
    print("\n7. 前20个最好的可JOIN表对:")
    for i, pair in enumerate(unique_pairs[:20], 1):
        best_col_pair = pair['column_pairs'][0]
        print(f"\n  {i}. {pair['db1']}.{pair['table1']} <-> {pair['db2']}.{pair['table2']}")
        print(f"     相似度: {pair['best_similarity']:.1f}")
        print(f"     最佳列对: {best_col_pair['col1']} <-> {best_col_pair['col2']}")
        keywords1_set = set(best_col_pair['keywords1'])
        keywords2_set = set(best_col_pair['keywords2'])
        common_keywords = keywords1_set & keywords2_set
        print(f"     共同关键词: {list(common_keywords)}")
    
    print("\n" + "=" * 70)
    print("分析完成！")
    print("=" * 70)
    
    return result

def main():
    parser = argparse.ArgumentParser(description='分析跨数据库可JOIN的表对')
    parser.add_argument('--database_dir', type=str,
                       default='benchmark/data/beijing/database_chinese',
                       help='数据库目录')
    parser.add_argument('--combinations_file', type=str,
                       default='benchmark/generation/cross_database/database_combinations.json',
                       help='数据库组合文件')
    parser.add_argument('--output_file', type=str,
                       default='benchmark/generation/cross_database/joinable_table_pairs.json',
                       help='输出文件')
    parser.add_argument('--min_similarity', type=float, default=5.0,
                       help='最小相似度阈值（默认5.0）')
    
    args = parser.parse_args()
    
    analyze_all_combinations(
        args.database_dir,
        args.combinations_file,
        args.output_file,
        args.min_similarity
    )

if __name__ == '__main__':
    main()

