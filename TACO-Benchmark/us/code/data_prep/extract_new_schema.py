# import os
# import pandas as pd
# import json

# # 定义路径
# raw_data_directory = os.path.join('..', '..', 'data', 'raw_csv_data')
# output_directory = os.path.join('..', '..', 'data', 'new_schema')
# # table_description_file = os.path.join('..', '..', 'data', 'raw_data', 'table_description.csv')
# table_name_mappings_file = os.path.join('..', '..', 'data', 'table_name_mappings.json')

# # 创建存储模式的文件夹（如果不存在）
# os.makedirs(output_directory, exist_ok=True)

# # 读取表格描述 CSV 文件
# table_descriptions = {}
# # with open(table_description_file, 'r', encoding='utf-8') as f:
# #     for line in f:
# #         parts = line.strip().split(',', 1)
# #         # print(parts)
# #         if len(parts) == 2:
# #             table_name, description = parts
# #             table_descriptions[table_name] = description

# # print(table_descriptions)

# # 读取表名映射 JSON 文件
# with open(table_name_mappings_file, 'r', encoding='utf-8') as f:
#     table_name_mappings = json.load(f)
# # print(table_name_mappings)

# # 遍历 raw_csv_data/ 下的所有文件夹
# for folder_name in os.listdir(raw_data_directory):
#     folder_path = os.path.join(raw_data_directory, folder_name)

#     if os.path.isdir(folder_path):  # 如果是文件夹（每个文件夹代表一个数据库）
#         schema = {'tables': []}
        
#         # 遍历该文件夹中的所有 CSV 文件
#         for file_name in os.listdir(folder_path):

#             if file_name.endswith('.csv'):  # 只处理 CSV 文件
#                 # print(file_name)
#                 csv_path = os.path.join(folder_path, file_name)
#                 # print(csv_path)
                
#                 # 使用 table_name_mappings 获取映射后的表格名
#                 mapped_table_name = table_name_mappings.get(folder_name, {}).get(file_name, os.path.splitext(file_name)[0])
#                 # print(mapped_table_name)
                
#                 # 获取表格描述
#                 table_name_without_csv = os.path.splitext(file_name)[0]  # 去掉 .csv 后缀
#                 table_description = table_descriptions.get(table_name_without_csv, "No description available.")
#                 # print(table_description)
                
#                 # 读取 CSV 文件为 DataFrame
#                 df = pd.read_csv(csv_path)

#                 # 获取列名和数据类型
#                 columns = []
#                 for col in df.columns:
#                     data_type = str(df[col].dtype)
#                     # 将 pandas 数据类型映射到 SQL 数据类型
#                     if 'int' in data_type:
#                         sql_type = 'INTEGER'
#                     elif 'float' in data_type:
#                         sql_type = 'REAL'
#                     else:
#                         sql_type = 'TEXT'
#                     columns.append({'column_name': col, 'data_type': sql_type})

#                 # 将表模式添加到 schema 列表中
#                 schema['tables'].append({
#                     'table_name': mapped_table_name,
#                     'table_comment': mapped_table_name,
#                     # 'table_description': table_description,  # 添加表格描述
#                     'table_description': "",  # 添加表格描述
#                     'columns': columns,
#                     'primary_keys': [],  # 如果有主键信息，可以在此添加
#                     'foreign_keys': []   # 如果有外键信息，可以在此添加
#                 })
        
#         # 保存每个数据库的 schema 到 JSON 文件
#         output_file = os.path.join(output_directory, f"{folder_name}_schema.json")
#         with open(output_file, 'w', encoding='utf-8') as f:
#             json.dump(schema, f, ensure_ascii=False, indent=2)

#         print(f"{folder_name} 的 schema 已提取并保存到 {output_file}")



import os
import pandas as pd
import json

# 定义路径
raw_data_directory = os.path.join('..', '..', 'data', 'raw_csv_data')
output_directory = os.path.join('..', '..', 'data', 'new_schema')
# table_description_file = os.path.join('..', '..', 'data', 'raw_data', 'table_description.csv')
table_name_mappings_file = os.path.join('..', '..', 'data', 'table_name_mappings.json')

# 创建存储模式的文件夹（如果不存在）
os.makedirs(output_directory, exist_ok=True)

# 读取表格描述 CSV 文件
table_descriptions = {}
# with open(table_description_file, 'r', encoding='utf-8') as f:
#     for line in f:
#         parts = line.strip().split(',', 1)
#         if len(parts) == 2:
#             table_name, description = parts
#             table_descriptions[table_name] = description

# 读取表名映射 JSON 文件
with open(table_name_mappings_file, 'r', encoding='utf-8') as f:
    table_name_mappings = json.load(f)

# 遍历 raw_csv_data/ 下的所有文件夹
for folder_name in os.listdir(raw_data_directory):
    folder_path = os.path.join(raw_data_directory, folder_name)

    if os.path.isdir(folder_path):  # 如果是文件夹（每个文件夹代表一个数据库）
        schema = {'tables': []}
        
        # 遍历该文件夹中的所有 CSV 文件
        for file_name in os.listdir(folder_path):
            if file_name.endswith('.csv'):  # 只处理 CSV 文件
                csv_path = os.path.join(folder_path, file_name)
                
                try:
                    # 使用 table_name_mappings 获取映射后的表格名
                    mapped_table_name = table_name_mappings.get(folder_name, {}).get(file_name, os.path.splitext(file_name)[0])
                    
                    # 获取表格描述
                    table_name_without_csv = os.path.splitext(file_name)[0]  # 去掉 .csv 后缀
                    table_description = table_descriptions.get(table_name_without_csv, "No description available.")
                    
                    # 读取 CSV 文件为 DataFrame
                    df = pd.read_csv(csv_path)

                    # 获取列名和数据类型
                    columns = []
                    for col in df.columns:
                        data_type = str(df[col].dtype)
                        # 将 pandas 数据类型映射到 SQL 数据类型
                        if 'int' in data_type:
                            sql_type = 'INTEGER'
                        elif 'float' in data_type:
                            sql_type = 'REAL'
                        else:
                            sql_type = 'TEXT'
                        columns.append({'column_name': col, 'data_type': sql_type})

                    # 将表模式添加到 schema 列表中
                    schema['tables'].append({
                        'table_name': mapped_table_name,
                        'table_comment': mapped_table_name,
                        # 'table_description': table_description,  # 添加表格描述
                        'table_description': "",  # 添加表格描述
                        'columns': columns,
                        'primary_keys': [],  # 如果有主键信息，可以在此添加
                        'foreign_keys': []   # 如果有外键信息，可以在此添加
                    })
                except Exception as e:
                    # 如果处理 CSV 文件时出错，跳过该文件并打印错误信息
                    print(f"Error processing file {csv_path}: {e}")
                    continue  # 跳过当前文件，继续处理下一个文件
        
        # 保存每个数据库的 schema 到 JSON 文件
        output_file = os.path.join(output_directory, f"{folder_name}_schema.json")
        try:
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(schema, f, ensure_ascii=False, indent=2)
            print(f"{folder_name} 的 schema 已提取并保存到 {output_file}")
        except Exception as e:
            # 如果保存 JSON 文件时出错，打印错误信息
            print(f"Error saving schema for {folder_name} to {output_file}: {e}")