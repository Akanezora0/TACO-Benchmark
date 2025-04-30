import os
import re
import shutil
import json
from pypinyin import lazy_pinyin


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
                    # parse the table name
                    # sanitized_name = sanitize_table_name(file_name.replace('.csv', ''))
                    sanitized_name = file_name.replace('.csv', '')
                    sanitized_path = os.path.join(parse_folder_path, sanitized_name + '.csv')
                    
                    # record the mapping relation
                    folder_mapping[file_name] = sanitized_name
                    
                    # copy the original file to the new path
                    original_csv_path = os.path.join(folder_path, file_name)
                    shutil.copy(original_csv_path, sanitized_path)
                    print(f"Processed: {file_name} -> {sanitized_name}.csv")
            
            if folder_mapping:
                # store the mapping of all files in the folder to the mappings
                mappings[folder_name] = folder_mapping
    
    # save the mapping file
    with open(mappings_dir, 'w', encoding='utf-8') as f:
        json.dump(mappings, f, ensure_ascii=False, indent=4)
    print(f"Mappings saved to {mappings_dir}")


raw_data_dir = '../../data/raw_csv_data'
parse_data_dir = '../../data/parsed_data'
mappings_dir = '../../data/table_name_mappings.json'


process_csv_folders(raw_data_dir, parse_data_dir, mappings_dir)
