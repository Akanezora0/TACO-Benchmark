import os
import re
import shutil
import json
from pypinyin import lazy_pinyin

# 判断是否包含中文字符
def contains_chinese(text):
    return any('\u4e00' <= char <= '\u9fff' for char in text)

def sanitize_table_name(name):
    # 如果文件名包含中文字符，则转化为拼音
    if contains_chinese(name):
        name_pinyin = ''.join(lazy_pinyin(name))
    else:
        name_pinyin = name
    
    # 替换表名中的空格为下划线
    name_pinyin = name_pinyin.replace(' ', '_')
    # 移除特殊字符，只保留字母、数字和下划线
    name_pinyin = re.sub(r'[^a-zA-Z0-9_]', '', name_pinyin)
    # 如果名字以数字开头，加上前缀
    if name_pinyin[0].isdigit():
        name_pinyin = '_' + name_pinyin
    return name_pinyin.lower()

def process_csv_folders(raw_data_dir, parse_data_dir, mappings_dir):
    mappings = {}
    
    for folder_name in os.listdir(raw_data_dir):
        folder_path = os.path.join(raw_data_dir, folder_name)
        if os.path.isdir(folder_path):
            # 创建对应的文件夹
            parse_folder_path = os.path.join(parse_data_dir, folder_name)
            os.makedirs(parse_folder_path, exist_ok=True)
            
            folder_mapping = {}
            
            for file_name in os.listdir(folder_path):
                if file_name.endswith('.csv'):
                    # 处理表格名
                    sanitized_name = sanitize_table_name(file_name.replace('.csv', ''))
                    sanitized_path = os.path.join(parse_folder_path, sanitized_name + '.csv')
                    
                    # 记录映射关系
                    folder_mapping[file_name] = sanitized_name
                    
                    # 复制原文件到新路径
                    original_csv_path = os.path.join(folder_path, file_name)
                    shutil.copy(original_csv_path, sanitized_path)
                    print(f"Processed: {file_name} -> {sanitized_name}.csv")
            
            if folder_mapping:
                # 将文件夹下所有文件的映射存储到 mappings 中
                mappings[folder_name] = folder_mapping
    
    # 保存映射文件
    with open(mappings_dir, 'w', encoding='utf-8') as f:
        json.dump(mappings, f, ensure_ascii=False, indent=4)
    print(f"Mappings saved to {mappings_dir}")

# 定义目录
raw_data_dir = '../../data/raw_csv_data'
parse_data_dir = '../../data/parsed_data'
mappings_dir = '../../data/table_name_mappings.json'

# 处理文件
process_csv_folders(raw_data_dir, parse_data_dir, mappings_dir)
