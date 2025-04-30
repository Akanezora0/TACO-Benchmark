import os
import re
import shutil
import json
from pypinyin import lazy_pinyin

# chinese character
def contains_chinese(text):
    return any('\u4e00' <= char <= '\u9fff' for char in text)

def sanitize_table_name(name):
    # pinyin
    if contains_chinese(name):
        name_pinyin = ''.join(lazy_pinyin(name))
    else:
        name_pinyin = name
    
    name_pinyin = name_pinyin.replace(' ', '_')

    name_pinyin = re.sub(r'[^a-zA-Z0-9_]', '', name_pinyin)

    if name_pinyin[0].isdigit():
        name_pinyin = '_' + name_pinyin
    return name_pinyin.lower()

def process_csv_folders(raw_data_dir, parse_data_dir, mappings_dir):
    mappings = {}
    
    for folder_name in os.listdir(raw_data_dir):
        folder_path = os.path.join(raw_data_dir, folder_name)
        if os.path.isdir(folder_path):

            parse_folder_path = os.path.join(parse_data_dir, folder_name)
            os.makedirs(parse_folder_path, exist_ok=True)
            
            folder_mapping = {}
            
            for file_name in os.listdir(folder_path):
                if file_name.endswith('.csv'):
                    # table name
                    sanitized_name = sanitize_table_name(file_name.replace('.csv', ''))
                    sanitized_path = os.path.join(parse_folder_path, sanitized_name + '.csv')
                    
                    # mapping
                    folder_mapping[file_name] = sanitized_name
                    
                    # copy
                    original_csv_path = os.path.join(folder_path, file_name)
                    shutil.copy(original_csv_path, sanitized_path)
                    print(f"Processed: {file_name} -> {sanitized_name}.csv")
            
            if folder_mapping:
                # mapping
                mappings[folder_name] = folder_mapping
    
    # mapping
    with open(mappings_dir, 'w', encoding='utf-8') as f:
        json.dump(mappings, f, ensure_ascii=False, indent=4)
    print(f"Mappings saved to {mappings_dir}")

# directory
raw_data_dir = '../../data/raw_csv_data'
parse_data_dir = '../../data/parsed_data'
mappings_dir = '../../data/table_name_mappings.json'

# process
process_csv_folders(raw_data_dir, parse_data_dir, mappings_dir)
