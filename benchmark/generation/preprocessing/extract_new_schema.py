# import os
# import pandas as pd
# import json

# # Define paths
# raw_data_directory = os.path.join('..', '..', 'data', 'raw_csv_data')
# output_directory = os.path.join('..', '..', 'data', 'new_schema')
# table_description_file = os.path.join('..', '..', 'data', 'raw_data', 'table_description.csv')
# table_name_mappings_file = os.path.join('..', '..', 'data', 'table_name_mappings.json')

# # Create schema output directory if it does not exist
# os.makedirs(output_directory, exist_ok=True)

# # Read table description CSV file
# table_descriptions = {}
# with open(table_description_file, 'r', encoding='utf-8') as f:
#     for line in f:
#         parts = line.strip().split(',', 1)
#         # print(parts)
#         if len(parts) == 2:
#             table_name, description = parts
#             table_descriptions[table_name] = description

# # print(table_descriptions)

# # Read table name mapping JSON file
# with open(table_name_mappings_file, 'r', encoding='utf-8') as f:
#     table_name_mappings = json.load(f)
# # print(table_name_mappings)

# # Iterate all folders under raw_csv_data/
# for folder_name in os.listdir(raw_data_directory):
#     folder_path = os.path.join(raw_data_directory, folder_name)

#     if os.path.isdir(folder_path):  # each folder represents one database
#         schema = {'tables': []}
        
#         # Iterate all CSV files in this folder
#         for file_name in os.listdir(folder_path):

#             if file_name.endswith('.csv'):  # process CSV files only
#                 # print(file_name)
#                 csv_path = os.path.join(folder_path, file_name)
#                 # print(csv_path)
                
#                 # Use table_name_mappings to get mapped table name
#                 mapped_table_name = table_name_mappings.get(folder_name, {}).get(file_name, os.path.splitext(file_name)[0])
#                 # print(mapped_table_name)
                
#                 # Get table description
#                 table_name_without_csv = os.path.splitext(file_name)[0]  # remove .csv suffix
#                 table_description = table_descriptions.get(table_name_without_csv, "No description available.")
#                 # print(table_description)
                
#                 # Read CSV file as DataFrame
#                 df = pd.read_csv(csv_path)

#                 # Get column names and data types
#                 columns = []
#                 for col in df.columns:
#                     data_type = str(df[col].dtype)
#                     # Map pandas data types to SQL data types
#                     if 'int' in data_type:
#                         sql_type = 'INTEGER'
#                     elif 'float' in data_type:
#                         sql_type = 'REAL'
#                     else:
#                         sql_type = 'TEXT'
#                     columns.append({'column_name': col, 'data_type': sql_type})

#                 # Add table schema to schema list
#                 schema['tables'].append({
#                     'table_name': mapped_table_name,
#                     'table_comment': mapped_table_name,
#                     'table_description': table_description,  # add table description
#                     'columns': columns,
#                     'primary_keys': [],  # add primary key info here if available
#                     'foreign_keys': []   # add foreign key info here if available
#                 })
        
#         # Save each database schema to a JSON file
#         output_file = os.path.join(output_directory, f"{folder_name}_schema.json")
#         with open(output_file, 'w', encoding='utf-8') as f:
#             json.dump(schema, f, ensure_ascii=False, indent=2)

#         print(f"Schema for {folder_name} extracted and saved to {output_file}")


import os
import pandas as pd
import json

# Define paths
raw_data_directory = os.path.join('..', '..', 'data', 'raw_csv_data')
output_directory = os.path.join('..', '..', 'data', 'new_schema')
table_description_file = os.path.join('..', '..', 'data', 'raw_data', 'table_description.csv')
table_name_mappings_file = os.path.join('..', '..', 'data', 'table_name_mappings.json')

# Create schema output directory if it does not exist
os.makedirs(output_directory, exist_ok=True)

# Read table description CSV file
table_descriptions = {}
with open(table_description_file, 'r', encoding='utf-8') as f:
    for line in f:
        parts = line.strip().split(',', 1)
        if len(parts) == 2:
            table_name, description = parts
            table_descriptions[table_name] = description

# Read table name mapping JSON file
with open(table_name_mappings_file, 'r', encoding='utf-8') as f:
    table_name_mappings = json.load(f)

# Iterate all folders under raw_csv_data/
for folder_name in os.listdir(raw_data_directory):
    folder_path = os.path.join(raw_data_directory, folder_name)

    if os.path.isdir(folder_path):  # each folder represents one database
        schema = {'tables': []}
        
        # Iterate all CSV files in this folder
        for file_name in os.listdir(folder_path):
            if file_name.endswith('.csv'):  # process CSV files only
                csv_path = os.path.join(folder_path, file_name)
                
                try:
                    # Use table_name_mappings to get mapped table name
                    mapped_table_name = table_name_mappings.get(folder_name, {}).get(file_name, os.path.splitext(file_name)[0])
                    
                    # Get table description
                    table_name_without_csv = os.path.splitext(file_name)[0]  # remove .csv suffix
                    table_description = table_descriptions.get(table_name_without_csv, "No description available.")
                    
                    # Read CSV file as DataFrame
                    df = pd.read_csv(csv_path)

                    # Get column names and data types
                    columns = []
                    for col in df.columns:
                        data_type = str(df[col].dtype)
                        # Map pandas data types to SQL data types
                        if 'int' in data_type:
                            sql_type = 'INTEGER'
                        elif 'float' in data_type:
                            sql_type = 'REAL'
                        else:
                            sql_type = 'TEXT'
                        columns.append({'column_name': col, 'data_type': sql_type})

                    # Add table schema to schema list
                    schema['tables'].append({
                        'table_name': mapped_table_name,
                        'table_comment': mapped_table_name,
                        'table_description': table_description,  # add table description
                        'columns': columns,
                        'primary_keys': [],  # add primary key info here if available
                        'foreign_keys': []   # add foreign key info here if available
                    })
                except Exception as e:
                    # On CSV processing error, skip file and print error message
                    print(f"Error processing file {csv_path}: {e}")
                    continue  # skip current file and continue with next
        
        # Save each database schema to a JSON file
        output_file = os.path.join(output_directory, f"{folder_name}_schema.json")
        try:
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(schema, f, ensure_ascii=False, indent=2)
            print(f"Schema for {folder_name} extracted and saved to {output_file}")
        except Exception as e:
            # On JSON save error, print error message
            print(f"Error saving schema for {folder_name} to {output_file}: {e}")