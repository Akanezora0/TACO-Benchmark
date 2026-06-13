#!/usr/bin/env python3
"""
Analyze real data (SmartCity) characteristics and extract templates
"""

import json
import os
import re
from collections import defaultdict, Counter
from typing import List, Dict

def count_tokens(text: str) -> int:
    """Count tokens (Chinese characters + English words)"""
    chinese_chars = len([c for c in text if '\u4e00' <= c <= '\u9fff'])
    english_words = len([w for w in text.split() if any(c.isalpha() for c in w)])
    return chinese_chars + english_words

def extract_opening_style(query: str) -> str:
    """Extract opening style"""
    query_clean = query.strip()
    
    if query_clean.startswith('市民反映'):
        return '市民反映'
    elif query_clean.startswith('市民投诉'):
        return '市民投诉'
    elif query_clean.startswith('市民咨询'):
        return '市民咨询'
    elif query_clean.startswith('工单来源'):
        return '工单来源'
    elif query_clean.startswith('企业反映'):
        return '企业反映'
    elif re.match(r'^\d+[\.、]', query_clean):
        return '结构化列表'
    elif query_clean.startswith('市民'):
        return '其他市民开头'
    else:
        return '其他'

def extract_business_scenario(query: str) -> str:
    """Extract business scenario (keyword matching)"""
    query_lower = query.lower()
    
    scenarios = {
        '发票问题': ['发票', '开发票', '开票', '专票', '增值税'],
        '税务问题': ['税务', '偷税', '漏税', '纳税', '个人所得税', '滞纳金'],
        '交通问题': ['交通', '违章', '停车', '驾驶证', '车牌', '进京证'],
        '医疗问题': ['医院', '医疗', '就诊', '医生', '药品', '医保'],
        '教育问题': ['教育', '学校', '培训', '课程', '学历'],
        '住房问题': ['住房', '租房', '房产', '居住证', '户口'],
        '企业问题': ['企业', '公司', '营业执照', '注册'],
        '社保问题': ['社保', '公积金', '养老保险'],
        '消费问题': ['消费', '购买', '订单', '退款', '赔偿'],
        '其他': []
    }
    
    for scenario, keywords in scenarios.items():
        if scenario == '其他':
            continue
        if any(keyword in query for keyword in keywords):
            return scenario
    
    return '其他'

def analyze_query_structure(query: str) -> Dict:
    """Analyze query structure features"""
    features = {
        'has_time': bool(re.search(r'\d{4}年\d{1,2}月\d{1,2}日|\d{4}-\d{1,2}-\d{1,2}|\d{1,2}月\d{1,2}日', query)),
        'has_location': bool(re.search(r'[区县]|地址|位置|地点', query)),
        'has_amount': bool(re.search(r'\d+元|\d+万元|\d+\.\d+元', query)),
        'has_phone': bool(re.search(r'mobile_|电话|手机', query)),
        'has_id': bool(re.search(r'sfzh_|身份证|证件号', query)),
        'has_name': bool(re.search(r'xm_|姓名|名称', query)),
        'has_address': bool(re.search(r'dz_|地址', query)),
        'has_emotion': bool(re.search(r'不满|不认可|希望|要求|投诉|反映', query)),
        'has_background': len(query) > 200,  # contains background information
        'has_demand': bool(re.search(r'希望|要求|诉求|需要|想', query)),
    }
    return features

def analyze_real_data(data_file: str):
    """Analyze real data"""
    with open(data_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    print("=" * 80)
    print("Real data (SmartCity) feature analysis")
    print("=" * 80)
    print()
    
    queries = [item['query'] for item in data]
    sqls = [item['sql'] for item in data]
    
    # 1. Length statistics
    token_counts = [count_tokens(q) for q in queries]
    print("[Length statistics]")
    print(f"  Total: {len(queries)}")
    print(f"  Average tokens: {sum(token_counts)/len(token_counts):.1f}")
    print(f"  Median: {sorted(token_counts)[len(token_counts)//2]}")
    print(f"  Min: {min(token_counts)}")
    print(f"  Max: {max(token_counts)}")
    print()
    
    # 2. Opening style statistics
    opening_styles = [extract_opening_style(q) for q in queries]
    style_counter = Counter(opening_styles)
    print("[Opening style distribution]")
    for style, count in style_counter.most_common():
        print(f"  {style}: {count} ({count/len(queries)*100:.1f}%)")
    print()
    
    # 3. Business scenario statistics
    scenarios = [extract_business_scenario(q) for q in queries]
    scenario_counter = Counter(scenarios)
    print("[Business scenario distribution]")
    for scenario, count in scenario_counter.most_common():
        print(f"  {scenario}: {count} ({count/len(queries)*100:.1f}%)")
    print()
    
    # 4. Structure feature statistics
    all_features = defaultdict(int)
    for q in queries:
        features = analyze_query_structure(q)
        for key, value in features.items():
            if value:
                all_features[key] += 1
    
    print("[Structure feature distribution]")
    for feature, count in sorted(all_features.items(), key=lambda x: x[1], reverse=True):
        print(f"  {feature}: {count} ({count/len(queries)*100:.1f}%)")
    print()
    
    # 5. Classify by style and scenario
    classified_queries = defaultdict(lambda: defaultdict(list))
    for i, (query, sql) in enumerate(zip(queries, sqls)):
        style = extract_opening_style(query)
        scenario = extract_business_scenario(query)
        token_count = count_tokens(query)
        
        classified_queries[style][scenario].append({
            'query': query,
            'sql': sql,
            'tokens': token_count,
            'index': i
        })
    
    print("[Classification statistics]")
    for style in sorted(classified_queries.keys()):
        print(f"  {style}:")
        for scenario in sorted(classified_queries[style].keys()):
            count = len(classified_queries[style][scenario])
            avg_tokens = sum(q['tokens'] for q in classified_queries[style][scenario]) / count if count > 0 else 0
            print(f"    {scenario}: {count} items (avg {avg_tokens:.1f} tokens)")
    print()
    
    return classified_queries, queries, sqls

def extract_templates(classified_queries: Dict, min_tokens: int = 80, max_templates: int = 100) -> List[Dict]:
    """Extract templates from classified data"""
    templates = []
    
    # Prefer queries with suitable length (80-200 tokens)
    for style in classified_queries.keys():
        for scenario in classified_queries[style].keys():
            queries_list = classified_queries[style][scenario]
            
            # Filter by length
            suitable_queries = [q for q in queries_list if min_tokens <= q['tokens'] <= 200]
            
            # Relax criteria if not enough suitable queries
            if len(suitable_queries) < 2:
                suitable_queries = queries_list
            
            # Select representative queries (prefer medium length)
            suitable_queries.sort(key=lambda x: abs(x['tokens'] - 120))  # prefer ~120 tokens
            
            # Select 1-3 per style-scenario combination
            selected = suitable_queries[:min(3, len(suitable_queries))]
            
            for q in selected:
                features = analyze_query_structure(q['query'])
                template = {
                    'query': q['query'],
                    'sql': q['sql'],
                    'style': style,
                    'scenario': scenario,
                    'tokens': q['tokens'],
                    'features': features,
                    'index': q['index']
                }
                templates.append(template)
                
                if len(templates) >= max_templates:
                    break
        
        if len(templates) >= max_templates:
            break
    
    return templates

def main():
    try:
        from taco.core.paths import NL_QUERY_DIR, PROJECT_ROOT
    except ImportError:
        from pathlib import Path
        PROJECT_ROOT = Path(__file__).resolve().parents[3]
        NL_QUERY_DIR = PROJECT_ROOT / "benchmark" / "generation" / "nl_query"

    data_file = os.environ.get(
        "TACO_REAL_DATA_FILE",
        str(PROJECT_ROOT / "benchmark" / "generation" / "nl_query" / "sample_source.json"),
    )
    
    print("Starting real data analysis...")
    print()
    
    # Analyze data
    classified_queries, all_queries, all_sqls = analyze_real_data(data_file)
    
    # Extract templates
    print("=" * 80)
    print("Extract templates")
    print("=" * 80)
    print()
    
    templates = extract_templates(classified_queries, min_tokens=80, max_templates=100)
    
    print(f"Extracted {len(templates)} templates")
    print()
    
    # Template distribution statistics
    style_counter = Counter(t['style'] for t in templates)
    scenario_counter = Counter(t['scenario'] for t in templates)
    
    print("[Template style distribution]")
    for style, count in style_counter.most_common():
        print(f"  {style}: {count}")
    print()
    
    print("[Template scenario distribution]")
    for scenario, count in scenario_counter.most_common():
        print(f"  {scenario}: {count}")
    print()
    
    # Save templates
    output_file = str(NL_QUERY_DIR / "real_data_templates.json")
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(templates, f, ensure_ascii=False, indent=2)
    
    print(f"✅ Templates saved to: {output_file}")
    print()
    
    # Show sample templates
    print("=" * 80)
    print("Template examples (first 5)")
    print("=" * 80)
    print()
    
    for i, template in enumerate(templates[:5], 1):
        print(f"[Template {i}]")
        print(f"  Style: {template['style']}")
        print(f"  Scenario: {template['scenario']}")
        print(f"  Tokens: {template['tokens']}")
        print(f"  Query: {template['query'][:150]}...")
        print()

if __name__ == '__main__':
    main()


