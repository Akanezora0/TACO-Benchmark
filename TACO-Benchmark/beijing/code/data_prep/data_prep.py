import os
import shutil
import pandas as pd
import sqlite3
import json
import re
from tqdm import tqdm
from pypinyin import lazy_pinyin
from typing import Dict, List, Any

class DataPreparator:
    def __init__(self, base_data_dir: str = '../../data'):
        self.base_data_dir = base_data_dir
        self.raw_data_dir = os.path.join(base_data_dir, 'raw_data')
        self.raw_csv_data_dir = os.path.join(base_data_dir, 'raw_csv_data')
        self.parse_data_dir = os.path.join(base_data_dir, 'parsed_data')
        self.database_dir = os.path.join(base_data_dir, 'database')
        self.new_schema_dir = os.path.join(base_data_dir, 'new_schema')
        self.mappings_file = os.path.join(base_data_dir, 'table_name_mappings.json')
        self.table_description_file = os.path.join(base_data_dir, 'raw_data', 'table_description.csv')

    def _is_valid_excel(self, excel_path: str) -> bool:
        try:
            pd.read_excel(excel_path, engine='openpyxl')
            return True
        except Exception as e:
            print(f"Error reading {excel_path}: {e}")
            return False

    def _convert_excel_to_csv(self, excel_path: str, csv_path: str) -> bool:
        try:
            df = pd.read_excel(excel_path, engine='openpyxl', header=None)
            df.to_csv(csv_path, index=False, header=False, encoding='utf-8')
            print(f"Converted {excel_path} to {csv_path}")
            return True
        except Exception as e:
            print(f"Error converting {excel_path}: {e}")
            return False

    def _contains_chinese(self, text: str) -> bool:
        return any('\u4e00' <= char <= '\u9fff' for char in text)

    def _sanitize_table_name(self, name: str) -> str:
        if self._contains_chinese(name):
            name_pinyin = ''.join(lazy_pinyin(name))
        else:
            name_pinyin = name
        
        name_pinyin = name_pinyin.replace(' ', '_')
        name_pinyin = re.sub(r'[^a-zA-Z0-9_]', '', name_pinyin)

        if name_pinyin[0].isdigit():
            name_pinyin = '_' + name_pinyin
        return name_pinyin.lower()

    def transfer_to_csv(self) -> None:
        """excel to csv"""
        if not os.path.exists(self.raw_csv_data_dir):
            os.makedirs(self.raw_csv_data_dir)

        for folder_name in tqdm(os.listdir(self.raw_data_dir), desc='Processing folders'):
            folder_path = os.path.join(self.raw_data_dir, folder_name)
            if os.path.isdir(folder_path):
                raw_csv_folder_path = os.path.join(self.raw_csv_data_dir, folder_name)
                os.makedirs(raw_csv_folder_path, exist_ok=True)
                
                for file_name in os.listdir(folder_path):
                    original_file_path = os.path.join(folder_path, file_name)
                    if file_name.endswith(('.xls', '.xlsx')):
                        csv_file_name = file_name.replace('.xls', '.csv').replace('.xlsx', '.csv')
                        csv_file_name = csv_file_name.replace('.csvx', '.csv')
                        csv_file_path = os.path.join(raw_csv_folder_path, csv_file_name)
                        if self._is_valid_excel(original_file_path):
                            self._convert_excel_to_csv(original_file_path, csv_file_path)
                    elif file_name.endswith('.csv'):
                        csv_file_path = os.path.join(raw_csv_folder_path, file_name)
                        try:
                            with open(original_file_path, 'rb') as f:
                                raw_data = f.read()
                                detected_encoding = 'utf-8'
                                try:
                                    import chardet
                                    result = chardet.detect(raw_data)
                                    if result['encoding']:
                                        detected_encoding = result['encoding']
                                except ImportError:
                                    print("chardet not installed, using default UTF-8 encoding.")
                                
                                content = raw_data.decode(detected_encoding, errors='replace')
                                with open(csv_file_path, 'w', encoding='utf-8') as f_out:
                                    f_out.write(content)
                            print(f"Converted encoding and copied {file_name} to {csv_file_path}")
                        except Exception as e:
                            print(f"Error processing CSV file {file_name}: {e}")

    def parse_table_names(self) -> None:
        """process table names, generate normalized names"""
        mappings = {}
        
        for folder_name in os.listdir(self.raw_csv_data_dir):
            folder_path = os.path.join(self.raw_csv_data_dir, folder_name)
            if os.path.isdir(folder_path):
                parse_folder_path = os.path.join(self.parse_data_dir, folder_name)
                os.makedirs(parse_folder_path, exist_ok=True)
                
                folder_mapping = {}
                
                for file_name in os.listdir(folder_path):
                    if file_name.endswith('.csv'):
                        sanitized_name = self._sanitize_table_name(file_name.replace('.csv', ''))
                        sanitized_path = os.path.join(parse_folder_path, sanitized_name + '.csv')
                        
                        folder_mapping[file_name] = sanitized_name
                        
                        original_csv_path = os.path.join(folder_path, file_name)
                        shutil.copy(original_csv_path, sanitized_path)
                        print(f"Processed: {file_name} -> {sanitized_name}.csv")
                
                if folder_mapping:
                    mappings[folder_name] = folder_mapping
        
        with open(self.mappings_file, 'w', encoding='utf-8') as f:
            json.dump(mappings, f, ensure_ascii=False, indent=4)
        print(f"Mappings saved to {self.mappings_file}")

    def extract_schema(self) -> None:
        """extract data table structure, generate schema information"""
        os.makedirs(self.new_schema_dir, exist_ok=True)

        table_descriptions = {}
        with open(self.table_description_file, 'r', encoding='utf-8') as f:
            for line in f:
                parts = line.strip().split(',', 1)
                if len(parts) == 2:
                    table_name, description = parts
                    table_descriptions[table_name] = description

        with open(self.mappings_file, 'r', encoding='utf-8') as f:
            table_name_mappings = json.load(f)

        for folder_name in os.listdir(self.raw_csv_data_dir):
            folder_path = os.path.join(self.raw_csv_data_dir, folder_name)
            if os.path.isdir(folder_path):
                schema = {'tables': []}
                
                for file_name in os.listdir(folder_path):
                    if file_name.endswith('.csv'):
                        csv_path = os.path.join(folder_path, file_name)
                        try:
                            mapped_table_name = table_name_mappings.get(folder_name, {}).get(file_name, os.path.splitext(file_name)[0])
                            table_name_without_csv = os.path.splitext(file_name)[0]
                            table_description = table_descriptions.get(table_name_without_csv, "No description available.")
                            
                            df = pd.read_csv(csv_path)
                            columns = []
                            for col in df.columns:
                                data_type = str(df[col].dtype)
                                if 'int' in data_type:
                                    sql_type = 'INTEGER'
                                elif 'float' in data_type:
                                    sql_type = 'REAL'
                                else:
                                    sql_type = 'TEXT'
                                columns.append({'column_name': col, 'data_type': sql_type})

                            schema['tables'].append({
                                'table_name': mapped_table_name,
                                'table_comment': mapped_table_name,
                                'table_description': table_description,
                                'columns': columns,
                                'primary_keys': [],
                                'foreign_keys': []
                            })
                        except Exception as e:
                            print(f"Error processing file {csv_path}: {e}")
                            continue
                
                output_file = os.path.join(self.new_schema_dir, f"{folder_name}_schema.json")
                try:
                    with open(output_file, 'w', encoding='utf-8') as f:
                        json.dump(schema, f, ensure_ascii=False, indent=2)
                    print(f"{folder_name} schema saved to {output_file}")
                except Exception as e:
                    print(f"Error saving schema for {folder_name} to {output_file}: {e}")

    def create_database(self) -> None:
        """create SQLite database and import data"""
        os.makedirs(self.database_dir, exist_ok=True)

        for folder_name in os.listdir(self.parse_data_dir):
            folder_path = os.path.join(self.parse_data_dir, folder_name)
            if os.path.isdir(folder_path):
                try:
                    db_folder_path = os.path.join(self.database_dir, folder_name)
                    os.makedirs(db_folder_path, exist_ok=True)
                    
                    sqlite_db_path = os.path.join(db_folder_path, folder_name + '.db')
                    conn = sqlite3.connect(sqlite_db_path)
                    cursor = conn.cursor()

                    db_structure = {}

                    for file_name in os.listdir(folder_path):
                        if file_name.endswith('.csv'):
                            csv_path = os.path.join(folder_path, file_name)
                            try:
                                df = pd.read_csv(csv_path)
                                df = df.fillna('')
                                table_name = file_name.replace('.csv', '')
                                
                                df.to_sql(table_name, conn, if_exists='replace', index=False)
                                print(f"Added table: {table_name} from {csv_path}")
                                
                                db_structure[table_name] = {
                                    'columns': df.columns.tolist(),
                                    'data': df.to_dict(orient='records')
                                }
                            except Exception as e:
                                print(f"Error processing file {csv_path}: {e}")
                                continue

                    conn.commit()
                    conn.close()

                    json_file_path = os.path.join(db_folder_path, f"{folder_name}.json")
                    with open(json_file_path, 'w', encoding='utf-8') as json_file:
                        json.dump(db_structure, json_file, ensure_ascii=False, indent=4)
                    print(f"Database created and structure/data saved for {folder_name}")
                except Exception as e:
                    print(f"Error processing folder {folder_path}: {e}")
                    continue

    def prepare_all(self) -> None:
        """execute complete data preparation process"""
        print("Starting data preparation process...")
        self.transfer_to_csv()
        self.parse_table_names()
        self.extract_schema()
        self.create_database()
        print("Data preparation completed!")

if __name__ == "__main__":
    preparator = DataPreparator()
    preparator.prepare_all()