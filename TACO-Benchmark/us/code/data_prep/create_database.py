# import os
# import sqlite3
# import pandas as pd
# import json

# # 将 CSV 文件转换为 SQLite 表
# def csv_to_sqlite(csv_folder_path, sqlite_db_path):
#     # 创建 SQLite 数据库连接
#     conn = sqlite3.connect(sqlite_db_path)
#     cursor = conn.cursor()

#     # 存储数据库结构信息和表格数据
#     db_structure = {}

#     # 遍历 CSV 文件，将每个文件转化为 SQLite 表
#     for file_name in os.listdir(csv_folder_path):
#         if file_name.endswith('.csv'):
#             csv_path = os.path.join(csv_folder_path, file_name)
#             # 读取 CSV 文件
#             df = pd.read_csv(csv_path)
            
#             # 将 NaN 替换为空字符串
#             df = df.fillna('')

#             # 获取表名（去掉 .csv 后缀）
#             table_name = file_name.replace('.csv', '')
            
#             # 将 DataFrame 写入 SQLite 数据库
#             df.to_sql(table_name, conn, if_exists='replace', index=False)
#             print(f"Added table: {table_name} from {csv_path}")
            
#             # 获取表的列名并存储到 db_structure
#             db_structure[table_name] = {
#                 'columns': df.columns.tolist(),
#                 'data': df.to_dict(orient='records')  # 将表格数据转为字典形式
#             }

#     # 提交并关闭数据库连接
#     conn.commit()
#     conn.close()

#     # 返回数据库结构（包括数据）
#     return db_structure

# # 将数据库结构和数据保存为 JSON 文件
# def save_db_structure_and_data_as_json(db_structure, db_folder_path, db_name):
#     # JSON 文件路径，使用数据库名作为文件名，后缀改为 .json
#     json_file_path = os.path.join(db_folder_path, f"{db_name}.json")
    
#     # 将结构和数据写入 JSON 文件
#     with open(json_file_path, 'w', encoding='utf-8') as json_file:
#         json.dump(db_structure, json_file, ensure_ascii=False, indent=4)
#     print(f"Database structure and data saved as JSON: {json_file_path}")

# # 处理 parsed_data 中的所有文件夹，生成 SQLite 数据库并保存 JSON 结构和数据
# def process_parsed_data_to_sqlite(parsed_data_dir, database_dir):
#     # 遍历 parsed_data/ 下的所有文件夹
#     for folder_name in os.listdir(parsed_data_dir):
#         folder_path = os.path.join(parsed_data_dir, folder_name)
#         if os.path.isdir(folder_path):
#             # 为每个文件夹创建一个 SQLite 数据库
#             db_folder_path = os.path.join(database_dir, folder_name)
#             os.makedirs(db_folder_path, exist_ok=True)
            
#             sqlite_db_path = os.path.join(db_folder_path, folder_name + '.db')
            
#             # 转化该文件夹中的所有 CSV 为 SQLite 表，并返回数据库结构和数据
#             db_structure = csv_to_sqlite(folder_path, sqlite_db_path)
            
#             # 保存数据库结构和数据为 JSON 文件，使用数据库名作为 JSON 文件名
#             save_db_structure_and_data_as_json(db_structure, db_folder_path, folder_name)
#             print(f"Database created and structure/data saved for {folder_name}")

# # 定义目录
# parsed_data_dir = '../../data/parsed_data'
# database_dir = '../../data/database'

# # 创建存储 SQLite 数据库的目录
# os.makedirs(database_dir, exist_ok=True)

# # 处理 parsed_data 中的所有文件夹，生成 SQLite 数据库并保存结构和数据
# process_parsed_data_to_sqlite(parsed_data_dir, database_dir)


import os
import sqlite3
import pandas as pd
import json

# 将 CSV 文件转换为 SQLite 表
def csv_to_sqlite(csv_folder_path, sqlite_db_path):
    # 创建 SQLite 数据库连接
    conn = sqlite3.connect(sqlite_db_path)
    cursor = conn.cursor()

    # 存储数据库结构信息和表格数据
    db_structure = {}

    # 遍历 CSV 文件，将每个文件转化为 SQLite 表
    for file_name in os.listdir(csv_folder_path):
        if file_name.endswith('.csv'):
            csv_path = os.path.join(csv_folder_path, file_name)
            try:
                # 读取 CSV 文件
                df = pd.read_csv(csv_path)
                
                # 将 NaN 替换为空字符串
                df = df.fillna('')

                # 获取表名（去掉 .csv 后缀）
                table_name = file_name.replace('.csv', '')
                
                # 将 DataFrame 写入 SQLite 数据库
                df.to_sql(table_name, conn, if_exists='replace', index=False)
                print(f"Added table: {table_name} from {csv_path}")
                
                # 获取表的列名并存储到 db_structure
                db_structure[table_name] = {
                    'columns': df.columns.tolist(),
                    'data': df.to_dict(orient='records')  # 将表格数据转为字典形式
                }
            except Exception as e:
                # 如果出现错误，跳过该文件并打印错误信息
                print(f"Error processing file {csv_path}: {e}")
                continue  # 跳过当前文件，继续处理下一个文件

    # 提交并关闭数据库连接
    conn.commit()
    conn.close()

    # 返回数据库结构（包括数据）
    return db_structure

# 将数据库结构和数据保存为 JSON 文件
def save_db_structure_and_data_as_json(db_structure, db_folder_path, db_name):
    # JSON 文件路径，使用数据库名作为文件名，后缀改为 .json
    json_file_path = os.path.join(db_folder_path, f"{db_name}.json")
    
    try:
        # 将结构和数据写入 JSON 文件
        with open(json_file_path, 'w', encoding='utf-8') as json_file:
            json.dump(db_structure, json_file, ensure_ascii=False, indent=4)
        print(f"Database structure and data saved as JSON: {json_file_path}")
    except Exception as e:
        # 如果保存 JSON 文件时出错，打印错误信息
        print(f"Error saving JSON file {json_file_path}: {e}")

# 处理 parsed_data 中的所有文件夹，生成 SQLite 数据库并保存 JSON 结构和数据
def process_parsed_data_to_sqlite(parsed_data_dir, database_dir):
    # 遍历 parsed_data/ 下的所有文件夹
    for folder_name in os.listdir(parsed_data_dir):
        folder_path = os.path.join(parsed_data_dir, folder_name)
        if os.path.isdir(folder_path):
            try:
                # 为每个文件夹创建一个 SQLite 数据库
                db_folder_path = os.path.join(database_dir, folder_name)
                os.makedirs(db_folder_path, exist_ok=True)
                
                sqlite_db_path = os.path.join(db_folder_path, folder_name + '.db')
                
                # 转化该文件夹中的所有 CSV 为 SQLite 表，并返回数据库结构和数据
                db_structure = csv_to_sqlite(folder_path, sqlite_db_path)
                
                # 保存数据库结构和数据为 JSON 文件，使用数据库名作为 JSON 文件名
                save_db_structure_and_data_as_json(db_structure, db_folder_path, folder_name)
                print(f"Database created and structure/data saved for {folder_name}")
            except Exception as e:
                # 如果处理文件夹时出错，打印错误信息并继续处理下一个文件夹
                print(f"Error processing folder {folder_path}: {e}")
                continue  # 跳过当前文件夹，继续处理下一个文件夹

# 定义目录
parsed_data_dir = '../../data/parsed_data'
database_dir = '../../data/database'

# 创建存储 SQLite 数据库的目录
os.makedirs(database_dir, exist_ok=True)

# 处理 parsed_data 中的所有文件夹，生成 SQLite 数据库并保存结构和数据
process_parsed_data_to_sqlite(parsed_data_dir, database_dir)