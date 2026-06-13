import os
import re
import shutil
import json
from pathlib import Path
from pypinyin import lazy_pinyin

_BENCHMARK_DATA = Path(__file__).resolve().parents[2] / "data"

# Check whether text contains Chinese characters
def contains_chinese(text):
    return any('\u4e00' <= char <= '\u9fff' for char in text)

def sanitize_table_name(name):
    # If the filename contains Chinese characters, convert to pinyin
    if contains_chinese(name):
        name_pinyin = ''.join(lazy_pinyin(name))
    else:
        name_pinyin = name
    
    # Replace spaces in table names with underscores
    name_pinyin = name_pinyin.replace(' ', '_')
    # Remove special characters; keep only letters, digits, and underscores
    name_pinyin = re.sub(r'[^a-zA-Z0-9_]', '', name_pinyin)
    # If the name starts with a digit, add a prefix
    if name_pinyin[0].isdigit():
        name_pinyin = '_' + name_pinyin
    return name_pinyin.lower()

def process_csv_folders(raw_data_dir, parse_data_dir, mappings_dir):
    mappings = {}
    
    for folder_name in os.listdir(raw_data_dir):
        folder_path = os.path.join(raw_data_dir, folder_name)
        if os.path.isdir(folder_path):
            # Create corresponding folder
            parse_folder_path = os.path.join(parse_data_dir, folder_name)
            os.makedirs(parse_folder_path, exist_ok=True)
            
            folder_mapping = {}
            
            for file_name in os.listdir(folder_path):
                if file_name.endswith('.csv'):
                    # Process table name
                    sanitized_name = sanitize_table_name(file_name.replace('.csv', ''))
                    sanitized_path = os.path.join(parse_folder_path, sanitized_name + '.csv')
                    
                    # Record mapping
                    folder_mapping[file_name] = sanitized_name
                    
                    # Copy original file to new path
                    original_csv_path = os.path.join(folder_path, file_name)
                    shutil.copy(original_csv_path, sanitized_path)
                    print(f"Processed: {file_name} -> {sanitized_name}.csv")
            
            if folder_mapping:
                # Store all file mappings for this folder
                mappings[folder_name] = folder_mapping
    
    # Save mapping file
    with open(mappings_dir, 'w', encoding='utf-8') as f:
        json.dump(mappings, f, ensure_ascii=False, indent=4)
    print(f"Mappings saved to {mappings_dir}")

# Define directories
raw_data_dir = str(_BENCHMARK_DATA / "raw_csv_data")
parse_data_dir = str(_BENCHMARK_DATA / "parsed_data")
mappings_dir = str(_BENCHMARK_DATA / "table_name_mappings.json")

# Process files
process_csv_folders(raw_data_dir, parse_data_dir, mappings_dir)
