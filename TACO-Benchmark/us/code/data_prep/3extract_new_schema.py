import os
import pandas as pd
import json

raw_data_directory = os.path.join('..', '..', 'data', 'raw_csv_data')
output_directory = os.path.join('..', '..', 'data', 'new_schema')
table_description_file = os.path.join('..', '..', 'data', 'raw_data', 'table_description.csv')
table_name_mappings_file = os.path.join('..', '..', 'data', 'table_name_mappings.json')

os.makedirs(output_directory, exist_ok=True)

# table description file
table_descriptions = {}
with open(table_description_file, 'r', encoding='utf-8') as f:
    for line in f:
        parts = line.strip().split(',', 1)
        if len(parts) == 2:
            table_name, description = parts
            table_descriptions[table_name] = description

# table name mappings file
with open(table_name_mappings_file, 'r', encoding='utf-8') as f:
    table_name_mappings = json.load(f)

# raw_csv_data
for folder_name in os.listdir(raw_data_directory):
    folder_path = os.path.join(raw_data_directory, folder_name)

    if os.path.isdir(folder_path):
        schema = {'tables': []}
        
        # csv files
        for file_name in os.listdir(folder_path):
            if file_name.endswith('.csv'):
                csv_path = os.path.join(folder_path, file_name)
                
                try:
                    # table name mappings
                    mapped_table_name = table_name_mappings.get(folder_name, {}).get(file_name, os.path.splitext(file_name)[0])
                    
                    # table description
                    table_name_without_csv = os.path.splitext(file_name)[0]
                    table_description = table_descriptions.get(table_name_without_csv, "No description available.")
                    
                    # read
                    df = pd.read_csv(csv_path)

                    # columns and data types
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

                    # add to schema
                    schema['tables'].append({
                        'table_name': mapped_table_name,
                        'table_comment': mapped_table_name,
                        'table_description': table_description,
                        'columns': columns,
                        'primary_keys': [],
                        'foreign_keys': []   # if exists
                    })
                except Exception as e:
                    print(f"Error processing file {csv_path}: {e}")
                    continue
        
        output_file = os.path.join(output_directory, f"{folder_name}_schema.json")
        try:
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(schema, f, ensure_ascii=False, indent=2)
            print(f"{folder_name} schema saved to {output_file}")
        except Exception as e:
            print(f"Error saving schema for {folder_name} to {output_file}: {e}")
