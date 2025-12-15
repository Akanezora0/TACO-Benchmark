import os
import json
import matplotlib.pyplot as plt
from collections import Counter
import numpy as np

# 读取原始数据
def load_original_data(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)

# 读取生成的数据
def load_generated_data(directory):
    generated_data = []
    for i in range(1000):  # 假设生成文件从0到999
        file_path = os.path.join(directory, f'generated_nl_query_{i}.json')
        if os.path.exists(file_path):
            with open(file_path, 'r', encoding='utf-8') as f:
                generated_data.append(json.load(f))
    return generated_data

# 关键词提取
def extract_keywords(sql_statements):
    keywords = []
    for sql in sql_statements:
        keywords += [word.upper() for word in sql.split() if word.upper() in ['SELECT', 'FROM', 'WHERE', 'JOIN', 'ON', 'GROUP BY', 'ORDER BY', 'HAVING', 'AND', 'OR', 'LIMIT']]
    return keywords

# 判断查询复杂性
def is_complex_query(sql):
    # 判断复杂查询的规则
    if any(keyword in sql for keyword in ['JOIN', '(', 'UNION', 'HAVING', 'GROUP BY', 'ORDER BY']):  # 包含连接、子查询、合并、分组、排序等
        return True
    # if 'WHERE' in sql and ('>' in sql or '<' in sql or '=' in sql):  # 包含条件的查询
    #     return True
    return False

# 统计简单查询和复杂查询
def classify_queries(sql_statements):
    simple_count = 0
    complex_count = 0
    for sql in sql_statements:
        if is_complex_query(sql):
            complex_count += 1
        else:
            simple_count += 1
    return simple_count, complex_count

# 可视化关键词分布为饼图
def visualize_keyword_distribution(original_keywords, generated_keywords, save_dir):
    original_counter = Counter(original_keywords)
    generated_counter = Counter(generated_keywords)

    keywords = set(original_counter.keys()).union(set(generated_counter.keys()))
    
    original_counts = [original_counter[key] for key in keywords]
    generated_counts = [generated_counter[key] for key in keywords]

    plt.figure(figsize=(12, 6))
    
    # 饼图显示原始关键词分布
    plt.subplot(1, 2, 1)
    plt.pie(original_counts, labels=keywords, autopct='%1.1f%%', startangle=140)
    plt.title('Original SQL Keywords Distribution')

    # 饼图显示生成关键词分布
    plt.subplot(1, 2, 2)
    plt.pie(generated_counts, labels=keywords, autopct='%1.1f%%', startangle=140)
    plt.title('Generated SQL Keywords Distribution')

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'keyword_distribution_pie.png'))
    plt.close()

# 可视化简单查询和复杂查询的分布为饼图
def visualize_query_complexity(original_simple_count, original_complex_count, generated_simple_count, generated_complex_count, save_dir):
    labels = ['Simple', 'Complex']
    original_counts = [original_simple_count, original_complex_count]
    generated_counts = [generated_simple_count, generated_complex_count]

    plt.figure(figsize=(12, 6))

    # 原始数据饼图
    plt.subplot(1, 2, 1)
    plt.pie(original_counts, labels=labels, autopct='%1.1f%%', startangle=140)
    plt.title('Original Queries Complexity Distribution')

    # 生成数据饼图
    plt.subplot(1, 2, 2)
    plt.pie(generated_counts, labels=labels, autopct='%1.1f%%', startangle=140)
    plt.title('Generated Queries Complexity Distribution')

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'query_complexity_pie.png'))
    plt.close()

# 生成关键词统计表格并保存为图片
def visualize_keyword_table(original_keywords, generated_keywords, save_dir):
    original_counter = Counter(original_keywords)
    generated_counter = Counter(generated_keywords)

    keywords = set(original_counter.keys()).union(set(generated_counter.keys()))
    
    original_counts = [original_counter[key] for key in keywords]
    generated_counts = [generated_counter[key] for key in keywords]

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.axis('off')

    table_data = [['Keyword', 'Original Count', 'Generated Count']]
    for keyword, original_count, generated_count in zip(keywords, original_counts, generated_counts):
        table_data.append([keyword, original_count, generated_count])

    table = ax.table(cellText=table_data, colLabels=None, cellLoc='center', loc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(12)
    table.scale(1, 1.5)

    plt.savefig(os.path.join(save_dir, 'keyword_distribution_table.png'), bbox_inches='tight')
    plt.close()

# 生成查询复杂性统计表格并保存为图片
def visualize_query_complexity_table(original_simple_count, original_complex_count, generated_simple_count, generated_complex_count, save_dir):
    labels = ['Simple', 'Complex']
    original_counts = [original_simple_count, original_complex_count]
    generated_counts = [generated_simple_count, generated_complex_count]

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.axis('off')

    table_data = [['Query Type', 'Original Count', 'Generated Count']]
    for label, original_count, generated_count in zip(labels, original_counts, generated_counts):
        table_data.append([label, original_count, generated_count])

    table = ax.table(cellText=table_data, colLabels=None, cellLoc='center', loc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(12)
    table.scale(1, 1.5)

    plt.savefig(os.path.join(save_dir, 'query_complexity_table.png'), bbox_inches='tight')
    plt.close()

# 主程序
if __name__ == '__main__':
    original_data_file = '../../data/old_database/12345_200.json'
    generated_data_directory = '../../data/generated_nl_queries/'
    save_directory = '../../data/analysis_results/'  # 保存结果的目录

    os.makedirs(save_directory, exist_ok=True)  # 创建保存目录

    original_data = load_original_data(original_data_file)
    generated_data = load_generated_data(generated_data_directory)

    # 提取 SQL 语句
    original_sql_statements = [original['sql'] for original in original_data]
    generated_sql_statements = [gen['sql'] for gen in generated_data]

    # 关键词提取
    original_keywords = extract_keywords(original_sql_statements)
    generated_keywords = extract_keywords(generated_sql_statements)

    # 统计简单查询和复杂查询
    original_simple_count, original_complex_count = classify_queries(original_sql_statements)
    generated_simple_count, generated_complex_count = classify_queries(generated_sql_statements)

    # 可视化结果
    visualize_keyword_distribution(original_keywords, generated_keywords, save_directory)
    visualize_query_complexity(original_simple_count, original_complex_count, generated_simple_count, generated_complex_count, save_directory)

    # 生成并保存表格图片
    visualize_keyword_table(original_keywords, generated_keywords, save_directory)
    visualize_query_complexity_table(original_simple_count, original_complex_count, generated_simple_count, generated_complex_count, save_directory)