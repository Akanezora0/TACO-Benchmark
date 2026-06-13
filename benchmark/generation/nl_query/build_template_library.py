#!/usr/bin/env python3
"""
Build an expanded template library (50-100 templates) from real data analysis results.
Anonymize templates and create diverse variants.
"""

import json
import re
from typing import List, Dict
from collections import defaultdict

def anonymize_template(template: Dict) -> Dict:
    """Anonymize a template"""
    query = template['query']
    
    # Anonymization rules
    # 1. Remove or replace specific names (xm__-prefixed hashes, including escaped \\x)
    # Match xm__\x... or xm__... format
    query = re.sub(r'xm__\\?x?[a-f0-9]{20,}', '某市民', query)
    query = re.sub(r'患者姓名[：:]\s*xm__\\?x?[a-f0-9]+', '患者姓名：某市民', query)
    query = re.sub(r'患者是xm__\\?x?[a-f0-9]+', '患者是某市民', query)
    query = re.sub(r'姓名[：:]\s*xm__\\?x?[a-f0-9]+', '姓名：某市民', query)
    
    # 2. Remove or replace ID numbers (sfzh_-prefixed hashes, including escaped \\x)
    # Match sfzh__\x... or sfzh_... format
    query = re.sub(r'sfzh_+\\?x?[a-f0-9]{20,}', '某身份证号', query)
    query = re.sub(r'身份证号[码]?[：:]\s*sfzh_+\\?x?[a-f0-9]+', '身份证号：某身份证号', query)
    query = re.sub(r'证件号[：:]\s*sfzh_+\\?x?[a-f0-9]+', '证件号：某身份证号', query)
    
    # 3. Remove or replace phone numbers (mobile_-prefixed hashes)
    query = re.sub(r'mobile_\\?x?[a-f0-9]+', '某手机号', query)
    query = re.sub(r'电话[：:]\s*mobile_\\?x?[a-f0-9]+', '电话：某手机号', query)
    query = re.sub(r'手机号[：:]\s*mobile_\\?x?[a-f0-9]+', '手机号：某手机号', query)
    query = re.sub(r'联系方式为mobile_\\?x?[a-f0-9]+', '联系方式为某手机号', query)
    
    # 4. Remove or replace specific addresses (dz_-prefixed hashes)
    query = re.sub(r'dz_\\?x?[a-f0-9]+', '某地址', query)
    query = re.sub(r'地址[：:]\s*dz_\\?x?[a-f0-9]+', '地址：某地址', query)
    
    # 5. Replace specific license plate numbers (keep format, replace content)
    query = re.sub(r'车牌号[：:]\s*京[A-Z0-9]+', '车牌号：京XXXXX', query)
    query = re.sub(r'车牌[：:]\s*京[A-Z0-9]+', '车牌：京XXXXX', query)
    query = re.sub(r'京[A-Z0-9]{6,}', '京XXXXX', query)  # match license plate format directly
    
    # 6. Replace specific company names (keep Chinese placeholder in anonymized data)
    # Note: preserve company name semantics, but can simplify
    # query = re.sub(r'公司名称[：:]\s*[^，,。.]+', '公司名称：某公司', query)
    
    # 7. Remove or replace specific amounts (keep amount concept, can be blurred)
    # Keep amounts since they are business information
    
    # 8. Remove or replace specific dates (keep date format, can be blurred)
    # Keep dates since they are business information
    
    template['query'] = query
    template['anonymized'] = True
    
    return template

def create_template_variants(base_template: Dict) -> List[Dict]:
    """Create variants from a base template"""
    variants = []
    
    query = base_template['query']
    style = base_template['style']
    scenario = base_template['scenario']
    
    # Variant 1: keep as-is (already anonymized)
    variant1 = base_template.copy()
    variant1['variant_id'] = f"{base_template.get('index', 0)}_v1"
    variants.append(variant1)
    
    # Variant 2: change opening style (if possible)
    if style == '市民反映':
        # Can change opening style to other variants (e.g. citizen inquiry, complaint)
        variant2 = base_template.copy()
        variant2['query'] = query.replace('市民反映', '市民咨询', 1)
        variant2['style'] = '市民咨询'
        variant2['variant_id'] = f"{base_template.get('index', 0)}_v2"
        variants.append(variant2)
    
    # Variant 3: simplified version (if original query is long)
    if base_template['tokens'] > 120:
        variant3 = base_template.copy()
        # Simplify: remove redundant info, keep core query intent
        # Example only; can be adjusted as needed
        variant3['query'] = query  # keep as-is for now; can optimize later
        variant3['variant_id'] = f"{base_template.get('index', 0)}_v3"
        variant3['simplified'] = True
        variants.append(variant3)
    
    return variants

def expand_template_library(real_templates: List[Dict], target_count: int = 100) -> List[Dict]:
    """Expand template library to target count"""
    expanded_templates = []
    
    # 1. Anonymize original templates
    anonymized_templates = []
    for template in real_templates:
        anonymized = anonymize_template(template.copy())
        anonymized_templates.append(anonymized)
    
    # 2. Classify by style and scenario
    classified = defaultdict(list)
    for template in anonymized_templates:
        key = f"{template['style']}_{template['scenario']}"
        classified[key].append(template)
    
    # 3. Create variants for each base template
    for template in anonymized_templates:
        variants = create_template_variants(template)
        expanded_templates.extend(variants)
        
        if len(expanded_templates) >= target_count:
            break
    
    # 4. If still not enough, select more from different style-scenario combinations
    if len(expanded_templates) < target_count:
        # Sort by style-scenario combo, prefer groups with fewer samples
        sorted_keys = sorted(classified.keys(), key=lambda k: len(classified[k]))
        
        for key in sorted_keys:
            templates_in_group = classified[key]
            for template in templates_in_group:
                if template not in expanded_templates:
                    expanded_templates.append(template)
                    if len(expanded_templates) >= target_count:
                        break
            if len(expanded_templates) >= target_count:
                break
    
    # 5. If still not enough, randomly select from original templates (anonymized)
    if len(expanded_templates) < target_count:
        remaining = [t for t in anonymized_templates if t not in expanded_templates]
        needed = target_count - len(expanded_templates)
        expanded_templates.extend(remaining[:needed])
    
    return expanded_templates[:target_count]

def analyze_template_library(templates: List[Dict]):
    """Analyze template library distribution"""
    from collections import Counter
    
    print("=" * 80)
    print("Template library analysis")
    print("=" * 80)
    print()
    
    print(f"Total templates: {len(templates)}")
    print()
    
    # Style distribution
    styles = [t['style'] for t in templates]
    style_counter = Counter(styles)
    print("[Style distribution]")
    for style, count in style_counter.most_common():
        print(f"  {style}: {count} ({count/len(templates)*100:.1f}%)")
    print()
    
    # Scenario distribution
    scenarios = [t['scenario'] for t in templates]
    scenario_counter = Counter(scenarios)
    print("[Scenario distribution]")
    for scenario, count in scenario_counter.most_common():
        print(f"  {scenario}: {count} ({count/len(templates)*100:.1f}%)")
    print()
    
    # Token length distribution
    token_counts = [t['tokens'] for t in templates]
    print("[Token length distribution]")
    print(f"  Average: {sum(token_counts)/len(token_counts):.1f}")
    print(f"  Median: {sorted(token_counts)[len(token_counts)//2]}")
    print(f"  Min: {min(token_counts)}")
    print(f"  Max: {max(token_counts)}")
    print()
    
    # Feature distribution
    all_features = defaultdict(int)
    for t in templates:
        features = t.get('features', {})
        for key, value in features.items():
            if value:
                all_features[key] += 1
    
    print("[Feature distribution]")
    for feature, count in sorted(all_features.items(), key=lambda x: x[1], reverse=True):
        print(f"  {feature}: {count} ({count/len(templates)*100:.1f}%)")
    print()

def main():
    try:
        from taco.core.paths import NL_QUERY_DIR
    except ImportError:
        from pathlib import Path
        NL_QUERY_DIR = Path(__file__).resolve().parent

    # Load real data templates
    real_templates_file = str(NL_QUERY_DIR / "real_data_templates.json")
    
    print("Loading real data templates...")
    with open(real_templates_file, 'r', encoding='utf-8') as f:
        real_templates = json.load(f)
    
    print(f"Original template count: {len(real_templates)}")
    print()
    
    # Expand template library
    print("Expanding template library...")
    expanded_templates = expand_template_library(real_templates, target_count=100)
    
    print(f"Expanded template count: {len(expanded_templates)}")
    print()
    
    # Analyze template library
    analyze_template_library(expanded_templates)
    
    # Save template library
    output_file = str(NL_QUERY_DIR / "template_library.json")
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(expanded_templates, f, ensure_ascii=False, indent=2)
    
    print(f"✅ Template library saved to: {output_file}")
    print()
    
    # Show sample templates
    print("=" * 80)
    print("Template library examples (first 5)")
    print("=" * 80)
    print()
    
    for i, template in enumerate(expanded_templates[:5], 1):
        print(f"[Template {i}]")
        print(f"  Style: {template['style']}")
        print(f"  Scenario: {template['scenario']}")
        print(f"  Tokens: {template['tokens']}")
        print(f"  Query: {template['query'][:200]}...")
        print()

if __name__ == '__main__':
    main()
