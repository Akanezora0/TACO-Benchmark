#!/usr/bin/env python3
"""
基于真实数据分析结果，构建扩充的模板库（50-100个模板）
对模板进行脱敏处理，并创建多样化的变体
"""

import json
import re
from typing import List, Dict
from collections import defaultdict

def anonymize_template(template: Dict) -> Dict:
    """对模板进行脱敏处理"""
    query = template['query']
    
    # 脱敏规则
    # 1. 移除或替换具体姓名（xm__开头的哈希值，包括转义字符\\x）
    # 匹配 xm__\x... 或 xm__... 格式
    query = re.sub(r'xm__\\?x?[a-f0-9]{20,}', '某市民', query)
    query = re.sub(r'患者姓名[：:]\s*xm__\\?x?[a-f0-9]+', '患者姓名：某市民', query)
    query = re.sub(r'患者是xm__\\?x?[a-f0-9]+', '患者是某市民', query)
    query = re.sub(r'姓名[：:]\s*xm__\\?x?[a-f0-9]+', '姓名：某市民', query)
    
    # 2. 移除或替换身份证号（sfzh_开头的哈希值，包括转义字符\\x）
    # 匹配 sfzh__\x... 或 sfzh_... 格式
    query = re.sub(r'sfzh_+\\?x?[a-f0-9]{20,}', '某身份证号', query)
    query = re.sub(r'身份证号[码]?[：:]\s*sfzh_+\\?x?[a-f0-9]+', '身份证号：某身份证号', query)
    query = re.sub(r'证件号[：:]\s*sfzh_+\\?x?[a-f0-9]+', '证件号：某身份证号', query)
    
    # 3. 移除或替换手机号（mobile_开头的哈希值）
    query = re.sub(r'mobile_\\?x?[a-f0-9]+', '某手机号', query)
    query = re.sub(r'电话[：:]\s*mobile_\\?x?[a-f0-9]+', '电话：某手机号', query)
    query = re.sub(r'手机号[：:]\s*mobile_\\?x?[a-f0-9]+', '手机号：某手机号', query)
    query = re.sub(r'联系方式为mobile_\\?x?[a-f0-9]+', '联系方式为某手机号', query)
    
    # 4. 移除或替换具体地址（dz_开头的哈希值）
    query = re.sub(r'dz_\\?x?[a-f0-9]+', '某地址', query)
    query = re.sub(r'地址[：:]\s*dz_\\?x?[a-f0-9]+', '地址：某地址', query)
    
    # 5. 替换具体车牌号（保留格式，但替换具体内容）
    query = re.sub(r'车牌号[：:]\s*京[A-Z0-9]+', '车牌号：京XXXXX', query)
    query = re.sub(r'车牌[：:]\s*京[A-Z0-9]+', '车牌：京XXXXX', query)
    query = re.sub(r'京[A-Z0-9]{6,}', '京XXXXX', query)  # 直接匹配车牌号格式
    
    # 6. 替换具体公司名称（保留"某公司"）
    # 注意：这里保留公司名称的语义，但可以简化
    # query = re.sub(r'公司名称[：:]\s*[^，,。.]+', '公司名称：某公司', query)
    
    # 7. 移除或替换具体金额（保留金额概念，但可以模糊化）
    # 保留金额，因为这是业务信息
    
    # 8. 移除或替换具体日期（保留日期格式，但可以模糊化）
    # 保留日期，因为这是业务信息
    
    template['query'] = query
    template['anonymized'] = True
    
    return template

def create_template_variants(base_template: Dict) -> List[Dict]:
    """基于基础模板创建变体"""
    variants = []
    
    query = base_template['query']
    style = base_template['style']
    scenario = base_template['scenario']
    
    # 变体1：保持原样（已脱敏）
    variant1 = base_template.copy()
    variant1['variant_id'] = f"{base_template.get('index', 0)}_v1"
    variants.append(variant1)
    
    # 变体2：改变开头方式（如果可能）
    if style == '市民反映':
        # 可以改为"市民咨询"、"市民投诉"等
        variant2 = base_template.copy()
        variant2['query'] = query.replace('市民反映', '市民咨询', 1)
        variant2['style'] = '市民咨询'
        variant2['variant_id'] = f"{base_template.get('index', 0)}_v2"
        variants.append(variant2)
    
    # 变体3：简化版本（如果原查询较长）
    if base_template['tokens'] > 120:
        variant3 = base_template.copy()
        # 简化：移除部分冗余信息，保留核心查询意图
        # 这里只是示例，实际可以根据需要调整
        variant3['query'] = query  # 暂时保持原样，后续可以优化
        variant3['variant_id'] = f"{base_template.get('index', 0)}_v3"
        variant3['simplified'] = True
        variants.append(variant3)
    
    return variants

def expand_template_library(real_templates: List[Dict], target_count: int = 100) -> List[Dict]:
    """扩充模板库到目标数量"""
    expanded_templates = []
    
    # 1. 对原始模板进行脱敏
    anonymized_templates = []
    for template in real_templates:
        anonymized = anonymize_template(template.copy())
        anonymized_templates.append(anonymized)
    
    # 2. 按风格和场景分类
    classified = defaultdict(list)
    for template in anonymized_templates:
        key = f"{template['style']}_{template['scenario']}"
        classified[key].append(template)
    
    # 3. 为每个基础模板创建变体
    for template in anonymized_templates:
        variants = create_template_variants(template)
        expanded_templates.extend(variants)
        
        if len(expanded_templates) >= target_count:
            break
    
    # 4. 如果还不够，从不同风格-场景组合中选择更多
    if len(expanded_templates) < target_count:
        # 按风格-场景组合排序，优先选择样本少的组合
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
    
    # 5. 如果还不够，从原始模板中随机选择（已脱敏）
    if len(expanded_templates) < target_count:
        remaining = [t for t in anonymized_templates if t not in expanded_templates]
        needed = target_count - len(expanded_templates)
        expanded_templates.extend(remaining[:needed])
    
    return expanded_templates[:target_count]

def analyze_template_library(templates: List[Dict]):
    """分析模板库的分布"""
    from collections import Counter
    
    print("=" * 80)
    print("模板库分析")
    print("=" * 80)
    print()
    
    print(f"总模板数: {len(templates)}")
    print()
    
    # 风格分布
    styles = [t['style'] for t in templates]
    style_counter = Counter(styles)
    print("【风格分布】")
    for style, count in style_counter.most_common():
        print(f"  {style}: {count} ({count/len(templates)*100:.1f}%)")
    print()
    
    # 场景分布
    scenarios = [t['scenario'] for t in templates]
    scenario_counter = Counter(scenarios)
    print("【场景分布】")
    for scenario, count in scenario_counter.most_common():
        print(f"  {scenario}: {count} ({count/len(templates)*100:.1f}%)")
    print()
    
    # Token长度分布
    token_counts = [t['tokens'] for t in templates]
    print("【Token长度分布】")
    print(f"  平均: {sum(token_counts)/len(token_counts):.1f}")
    print(f"  中位数: {sorted(token_counts)[len(token_counts)//2]}")
    print(f"  最短: {min(token_counts)}")
    print(f"  最长: {max(token_counts)}")
    print()
    
    # 特征分布
    all_features = defaultdict(int)
    for t in templates:
        features = t.get('features', {})
        for key, value in features.items():
            if value:
                all_features[key] += 1
    
    print("【特征分布】")
    for feature, count in sorted(all_features.items(), key=lambda x: x[1], reverse=True):
        print(f"  {feature}: {count} ({count/len(templates)*100:.1f}%)")
    print()

def main():
    # 加载真实数据模板
    real_templates_file = '/home/u2023103807/TACO/benchmark/generation/nl_query/real_data_templates.json'
    
    print("加载真实数据模板...")
    with open(real_templates_file, 'r', encoding='utf-8') as f:
        real_templates = json.load(f)
    
    print(f"原始模板数: {len(real_templates)}")
    print()
    
    # 扩充模板库
    print("扩充模板库...")
    expanded_templates = expand_template_library(real_templates, target_count=100)
    
    print(f"扩充后模板数: {len(expanded_templates)}")
    print()
    
    # 分析模板库
    analyze_template_library(expanded_templates)
    
    # 保存模板库
    output_file = '/home/u2023103807/TACO/benchmark/generation/nl_query/template_library.json'
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(expanded_templates, f, ensure_ascii=False, indent=2)
    
    print(f"✅ 模板库已保存到: {output_file}")
    print()
    
    # 显示一些示例
    print("=" * 80)
    print("模板库示例（前5个）")
    print("=" * 80)
    print()
    
    for i, template in enumerate(expanded_templates[:5], 1):
        print(f"【模板 {i}】")
        print(f"  风格: {template['style']}")
        print(f"  场景: {template['scenario']}")
        print(f"  Token数: {template['tokens']}")
        print(f"  查询: {template['query'][:200]}...")
        print()

if __name__ == '__main__':
    main()

