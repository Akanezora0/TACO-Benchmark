import os
import json
import matplotlib.pyplot as plt
from collections import Counter
import numpy as np

# Load original data
def load_original_data(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)

# Load generated data
def load_generated_data(directory):
    generated_data = []
    for i in range(1000):  # Assume generated files range from 0 to 999
        file_path = os.path.join(directory, f'generated_nl_query_{i}.json')
        if os.path.exists(file_path):
            with open(file_path, 'r', encoding='utf-8') as f:
                generated_data.append(json.load(f))
    return generated_data

# Extract keywords
def extract_keywords(sql_statements):
    keywords = []
    for sql in sql_statements:
        keywords += [word.upper() for word in sql.split() if word.upper() in ['SELECT', 'FROM', 'WHERE', 'JOIN', 'ON', 'GROUP BY', 'ORDER BY', 'HAVING', 'AND', 'OR', 'LIMIT']]
    return keywords

# Determine query complexity
def is_complex_query(sql):
    # Rules for identifying complex queries
    if any(keyword in sql for keyword in ['JOIN', '(', 'UNION', 'HAVING', 'GROUP BY', 'ORDER BY']):  # Contains joins, subqueries, unions, grouping, ordering, etc.
        return True
    # if 'WHERE' in sql and ('>' in sql or '<' in sql or '=' in sql):  # Queries with conditions
    #     return True
    return False

# Count simple and complex queries
def classify_queries(sql_statements):
    simple_count = 0
    complex_count = 0
    for sql in sql_statements:
        if is_complex_query(sql):
            complex_count += 1
        else:
            simple_count += 1
    return simple_count, complex_count

# Visualize keyword distribution as pie charts
def visualize_keyword_distribution(original_keywords, generated_keywords, save_dir):
    original_counter = Counter(original_keywords)
    generated_counter = Counter(generated_keywords)

    keywords = set(original_counter.keys()).union(set(generated_counter.keys()))
    
    original_counts = [original_counter[key] for key in keywords]
    generated_counts = [generated_counter[key] for key in keywords]

    plt.figure(figsize=(12, 6))
    
    # Pie chart for original keyword distribution
    plt.subplot(1, 2, 1)
    plt.pie(original_counts, labels=keywords, autopct='%1.1f%%', startangle=140)
    plt.title('Original SQL Keywords Distribution')

    # Pie chart for generated keyword distribution
    plt.subplot(1, 2, 2)
    plt.pie(generated_counts, labels=keywords, autopct='%1.1f%%', startangle=140)
    plt.title('Generated SQL Keywords Distribution')

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'keyword_distribution_pie.png'))
    plt.close()

# Visualize simple vs complex query distribution as pie charts
def visualize_query_complexity(original_simple_count, original_complex_count, generated_simple_count, generated_complex_count, save_dir):
    labels = ['Simple', 'Complex']
    original_counts = [original_simple_count, original_complex_count]
    generated_counts = [generated_simple_count, generated_complex_count]

    plt.figure(figsize=(12, 6))

    # Pie chart for original data
    plt.subplot(1, 2, 1)
    plt.pie(original_counts, labels=labels, autopct='%1.1f%%', startangle=140)
    plt.title('Original Queries Complexity Distribution')

    # Pie chart for generated data
    plt.subplot(1, 2, 2)
    plt.pie(generated_counts, labels=labels, autopct='%1.1f%%', startangle=140)
    plt.title('Generated Queries Complexity Distribution')

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'query_complexity_pie.png'))
    plt.close()

# Generate keyword statistics table and save as image
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

# Generate query complexity statistics table and save as image
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

# Main program
if __name__ == '__main__':
    original_data_file = '../../data/old_database/12345_200.json'
    generated_data_directory = '../../data/generated_nl_queries/'
    save_directory = '../../data/analysis_results/'  # Directory to save results

    os.makedirs(save_directory, exist_ok=True)  # Create output directory

    original_data = load_original_data(original_data_file)
    generated_data = load_generated_data(generated_data_directory)

    # Extract SQL statements
    original_sql_statements = [original['sql'] for original in original_data]
    generated_sql_statements = [gen['sql'] for gen in generated_data]

    # Extract keywords
    original_keywords = extract_keywords(original_sql_statements)
    generated_keywords = extract_keywords(generated_sql_statements)

    # Count simple and complex queries
    original_simple_count, original_complex_count = classify_queries(original_sql_statements)
    generated_simple_count, generated_complex_count = classify_queries(generated_sql_statements)

    # Visualize results
    visualize_keyword_distribution(original_keywords, generated_keywords, save_directory)
    visualize_query_complexity(original_simple_count, original_complex_count, generated_simple_count, generated_complex_count, save_directory)

    # Generate and save table images
    visualize_keyword_table(original_keywords, generated_keywords, save_directory)
    visualize_query_complexity_table(original_simple_count, original_complex_count, generated_simple_count, generated_complex_count, save_directory)
