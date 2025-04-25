# import json
# import os
# import re
# from tqdm import tqdm
# import random
# import sqlparse
# from openai import OpenAI
# import sqlite3

# key = ""

# # client = OpenAI(api_key="", base_url="")
# client = OpenAI(
#     base_url="",
#     api_key=key
# )

# def generate_text(user_input):
#     try:
#         response = client.chat.completions.create(
#             model="gpt-3.5-turbo",
#             temperature=0.5,
#             messages=[
#                 {"role": "system", "content": "You are a helpful assistant"},
#                 {"role": "user", "content": user_input},
#             ],
#         )
#         assistant_reply = response.choices[0].message.content
#         print(f"Assistant: {assistant_reply}")
#         return assistant_reply.strip()
#     except Exception as e:
#         # print(f"An error occurred: {e}")
#         return None

# # 加载 SQL 框架
# def load_sql_skeleton(sql_skeketon_file):
#     with open(sql_skeketon_file, 'r', encoding='utf-8') as f:
#         sql_skeleton = json.load(f)
#     return sql_skeleton

# # 加载模式信息
# def load_schema(schema_file):
#     with open(schema_file, 'r', encoding='utf-8') as f:
#         schema = json.load(f)
#     return schema

# # 提取模式信息
# def extract_schema_info(schema):
#     schema_info = {
#         'tables': [],
#         'columns': {}
#     }
#     for table in schema['tables']:
#         table_name = table['table_name']
#         schema_info['tables'].append(table_name)
#         columns = []
#         for column in table['columns']:
#             column_name = column['column_name']
#             full_column_name = f"{table_name}.{column_name}"
#             columns.append(full_column_name)
#         schema_info['columns'][table_name] = columns
#     return schema_info

# # 随机选择表
# def select_random_tables(schema_info, num_tables=2):
#     selected_tables = random.sample(schema_info['tables'], min(num_tables, len(schema_info['tables'])))
#     selected_columns = {}
#     for table in selected_tables:
#         selected_columns[table] = schema_info['columns'][table]
#     return selected_tables, selected_columns

# # 构建提示
# def construct_prompt(sql_framework, selected_tables, selected_columns, cross_database=False):
#     # 将模式信息转换为可读格式，并使用双引号引用表名和列名
#     def quote_identifier(identifier):
#         return f'"{identifier}"'

#     if cross_database:
#         # 跨数据库
#         tables = ', '.join([quote_identifier(table) for table in selected_tables])
#         columns = []
#         for table in selected_tables:
#             columns.extend([quote_identifier(col) for col in selected_columns[table]])
#         columns_str = ', '.join(columns)
#         databases = ', '.join(list(set([table.split('.')[0] for table in selected_columns])))
#         prompt = f"""
# 请根据以下 SQL 框架和可用的表名、列名，填充占位符"_"，生成完整且可在 SQLite 上正确执行的 SQL 语句。

# 严格要求：
# - **仅输出最终生成的完整 SQL 语句，不要重复提示内容。**
# - **生成的 SQL 要保证语法正确，可以直接在 SQLite 上运行得到结果。**
# - **不要添加任何额外的解释、注释或输出格式（代码块，空格等）。**
# - **添加的表名、列名、WHERE 条件等内容必须在给定的表和列中。**
# - **在给你的（表名.列名）中用双引号单独包住列名，但一定不要包住表名。结果应例如（表名."列名"）**
# - **可以对给定的SQL框架做调整，最后生成更合理的SQL语句。**

# SQL 框架：
# {sql_framework}

# 可用的数据库：
# {databases}

# 可用的表名：
# {tables}

# 可用的列名（格式：表名.列名）：
# {columns_str}

# 请仅输出生成的完整 SQL 语句：
# """
#     else:
#         # 单一数据库
#         tables = ', '.join([quote_identifier(table) for table in selected_tables])
#         columns = []
#         for table in selected_tables:
#             columns.extend([quote_identifier(col) for col in selected_columns[table]])
#         columns_str = ', '.join(columns)
#         prompt = f"""
# 请根据以下 SQL 框架和可用的表名、列名，填充占位符"_"，生成完整且可在 SQLite 上正确执行的 SQL 语句。

# 严格要求：
# - **仅输出最终生成的完整 SQL 语句，不要重复提示内容。**
# - **生成的 SQL 要保证语法正确，可以直接在 SQLite 上运行得到结果。**
# - **不要添加任何额外的解释、注释或输出格式（代码块，空格等）。**
# - **添加的表名、列名、WHERE 条件等内容必须在给定的表和列中。**
# - **在给你的（表名.列名）中用双引号单独包住列名，但一定不要包住表名。结果应例如（表名."列名"）**
# - **可以对给定的SQL框架做调整，最后生成更合理的SQL语句。**

# SQL 框架：
# {sql_framework}

# 可用的表名：
# {tables}

# 可用的列名（格式：表名.列名）：
# {columns_str}

# 请仅输出生成的完整 SQL 语句：
# """
#     return prompt.strip()

# # 提取模型输出中的 SQL 语句
# def extract_sql_from_response(response):
#     sql_statement = response.strip()
#     # 确保以 "SELECT" 开头
#     if not sql_statement.lower().startswith('select'):
#         match = re.search(r'(SELECT\s.*)', sql_statement, re.IGNORECASE | re.DOTALL)
#         if match:
#             sql_statement = match.group(1).strip()
#     return sql_statement

# # 验证 SQL 语句的语法
# def is_valid_sql(sql_statement):
#     try:
#         parsed = sqlparse.parse(sql_statement)
#         if parsed and len(parsed) > 0:
#             return True
#         else:
#             return False
#     except Exception:
#         return False

# # 执行单一数据库的 SQL 语句
# def execute_single_db_sql(sql, db_path):
#     try:
#         conn = sqlite3.connect(db_path)
#         cursor = conn.cursor()
#         cursor.execute(sql)
#         results = cursor.fetchall()
#         conn.close()
#         if results:
#             return results, True
#         else:
#             return 0, False
#     except sqlite3.Error as e:
#         print(f"SQLite 错误: {e}")
#         return 0, False

# # 执行跨数据库的 SQL 语句
# def execute_cross_db_sql(sql, primary_db_path, secondary_db_path, secondary_alias):
#     try:
#         conn = sqlite3.connect(primary_db_path)
#         cursor = conn.cursor()
#         # 附加第二个数据库
#         attach_statement = f'ATTACH DATABASE "{secondary_db_path}" AS "{secondary_alias}";'
#         cursor.execute(attach_statement)
#         # 执行 SQL 查询
#         cursor.execute(sql)
#         results = cursor.fetchall()
#         # 分离数据库
#         detach_statement = f'DETACH DATABASE "{secondary_alias}";'
#         cursor.execute(detach_statement)
#         conn.close()
#         if results:
#             return results, True
#         else:
#             return 0, False
#     except sqlite3.Error as e:
#         print(f"SQLite 错误: {e}")
#         return 0, False

# # 主函数
# def main():
#     MAX_RETRIES = 3  # 最大重试次数

#     # 获取脚本所在目录
#     script_dir = os.path.dirname(os.path.abspath(__file__))

#     # 定义输入和输出路径
#     new_skeleton_dir = os.path.join(script_dir, '..', '..', 'data', 'new_sql_skeleton')
#     new_structure_dir = os.path.join(script_dir, '..', '..', 'data', 'new_sql_structure')
#     new_schema_dir = os.path.join(script_dir, '..', '..', 'data', 'new_schema')
#     output_dir = os.path.join(script_dir, '..', '..', 'data')
#     single_output_dir = os.path.join(output_dir, 'new_sql_single')
#     cross_output_dir = os.path.join(output_dir, 'new_sql_cross')
#     os.makedirs(single_output_dir, exist_ok=True)
#     os.makedirs(cross_output_dir, exist_ok=True)

#     # 加载所有数据库的 schema 信息
#     schema_files = [f for f in os.listdir(new_schema_dir) if f.endswith('_schema.json')]
#     schemas = {}
#     for schema_file in schema_files:
#         database_name = schema_file.replace('_schema.json', '')
#         schema_file_path = os.path.join(new_schema_dir, schema_file)
#         schema = load_schema(schema_file_path)
#         schema_info = extract_schema_info(schema)
#         schemas[database_name] = schema_info

#     # 加载所有数据库的 SQL skeleton 文件
#     skeleton_files = [f for f in os.listdir(new_skeleton_dir) if f.endswith('_sql_skeleton.json')]

#     print("正在处理新数据库的 SQL skeleton 文件...")

#     for skeleton_file in tqdm(skeleton_files, desc="总体进度"):
#         # 提取数据库名称
#         database_name = skeleton_file.replace('_sql_skeleton.json', '')
#         skeleton_file_path = os.path.join(new_skeleton_dir, skeleton_file)
#         print(f"\n正在处理数据库 '{database_name}' 的 SQL skeleton...")

#         # 加载对应的 SQL skeletons
#         with open(skeleton_file_path, 'r', encoding='utf-8') as f:
#             sql_skeletons = json.load(f)

#         # 获取对应的 schema 信息
#         if database_name not in schemas:
#             print(f"Schema info for database '{database_name}' not found. Skipping.")
#             continue
#         schema_info = schemas[database_name]

#         # 遍历每个 SQL skeleton
#         for idx, sql_skeleton in enumerate(tqdm(sql_skeletons, desc=f"{database_name} 的框架处理进度", leave=False)):
#             # 生成单一数据库的 SQL
#             # 判断 SQL skeleton 需要多少张表（通过 '_' 的数量）
#             num_placeholders = sql_skeleton.count('_')
#             # 假设每个表至少有一个 '_', 根据 '_' 的数量决定
#             if num_placeholders <= 1:
#                 required_tables_single = 1
#             else:
#                 # 例如，如果有多个 SELECT 或 FROM 子句，可以推断需要多张表
#                 # 这里简化为最多需要两张表
#                 required_tables_single = min(2, num_placeholders)

#             # 随机选择需要的表数
#             if required_tables_single == 1:
#                 selected_tables_single, selected_columns_single = select_random_tables(schema_info, num_tables=1)
#             else:
#                 selected_tables_single, selected_columns_single = select_random_tables(schema_info, num_tables=2)

#             # 构建提示
#             prompt_single = construct_prompt(sql_skeleton, selected_tables_single, selected_columns_single, cross_database=False)
#             print(prompt_single)

#             # 检查目标文件是否存在，存在则跳过
#             single_output_path = os.path.join(single_output_dir, database_name)
#             os.makedirs(single_output_path, exist_ok=True)
#             output_file_single = os.path.join(single_output_path, f'generated_sql_statement_{idx}.json')
#             if os.path.exists(output_file_single):
#                 print(f"文件已存在，跳过生成：{output_file_single}")
#             else:
#                 # 尝试生成并验证 SQL，最多重试 MAX_RETRIES 次
#                 for attempt in range(1, MAX_RETRIES + 1):
#                     print(f"生成单一数据库 SQL，第 {attempt} 次尝试...")
#                     sql_statement_single = generate_text(prompt_single)
#                     if not sql_statement_single:
#                         print("生成 SQL 失败，重试...")
#                         continue
#                     sql_statement_single = extract_sql_from_response(sql_statement_single)

#                     if is_valid_sql(sql_statement_single) and sql_statement_single.lower().startswith('select'):
#                         # 获取数据库路径
#                         db_path = os.path.join(script_dir, '..', '..', 'data', 'database', database_name, f"{database_name}.db")
#                         if not os.path.exists(db_path):
#                             print(f"数据库文件不存在：{db_path}，跳过生成。")
#                             break

#                         # 执行 SQL
#                         results, success = execute_single_db_sql(sql_statement_single, db_path)
#                         if success:
#                             # 准备保存的数据
#                             save_data_single = {
#                                 'sql': sql_statement_single,
#                                 'results': results,
#                                 'sql skeleton': sql_skeleton,
#                                 'database': database_name,
#                                 'tables': {},
#                             }
#                             for table in selected_tables_single:
#                                 table_name = table.split('.')[0]
#                                 column_names = selected_columns_single[table]
#                                 save_data_single['tables'][table_name] = column_names

#                             # 保存生成的 SQL 语句到单独的 JSON 文件
#                             with open(output_file_single, 'w', encoding='utf-8') as f:
#                                 json.dump(save_data_single, f, ensure_ascii=False, indent=2)
#                             print(f"成功生成并保存单一数据库 SQL：{output_file_single}")
#                             break  # 成功，跳出重试循环
#                         else:
#                             print(f"执行 SQL 失败或无结果，重试...")
#                     else:
#                         print(f"生成的单一数据库 SQL 语句无效，重试...")

#                     if attempt == MAX_RETRIES:
#                         print(f"已达到最大重试次数，跳过生成：{output_file_single}")

#             # 生成跨数据库的 SQL
#             # 判断是否有足够的数据库进行跨数据库操作
#             available_databases = list(schemas.keys())
#             if len(available_databases) < 2:
#                 print("可用的数据库少于两个，无法生成跨数据库 SQL。")
#                 continue

#             # 随机选择两个不同的数据库
#             database_names_cross = random.sample(available_databases, 2)
#             schema_info_1 = schemas[database_names_cross[0]]
#             schema_info_2 = schemas[database_names_cross[1]]

#             # 判断 SQL skeleton 需要多少张表
#             if num_placeholders <= 1:
#                 required_tables_cross = 1
#             else:
#                 required_tables_cross = min(2, num_placeholders)

#             # 根据需要的表数，从两个数据库中选择表
#             if required_tables_cross == 1:
#                 # 只选择一个表，随机选择其中一个数据库
#                 chosen_db = random.choice([0, 1])
#                 selected_tables_cross, selected_columns_cross = select_random_tables(
#                     schema_info_1 if chosen_db == 0 else schema_info_2,
#                     num_tables=1
#                 )
#                 cross_database = False  # 只用一个数据库
#                 # 设置附加的数据库路径和别名
#                 primary_db = database_names_cross[0] if chosen_db == 0 else database_names_cross[1]
#                 secondary_db = None
#                 secondary_alias = None
#                 secondary_db_path = None
#             else:
#                 # 选择两个表，各自来自不同的数据库
#                 selected_tables_1, selected_columns_1 = select_random_tables(schema_info_1, num_tables=1)
#                 selected_tables_2, selected_columns_2 = select_random_tables(schema_info_2, num_tables=1)
#                 selected_tables_cross = selected_tables_1 + selected_tables_2
#                 selected_columns_cross = {**selected_columns_1, **selected_columns_2}
#                 cross_database = True
#                 # 设置附加的数据库路径和别名
#                 primary_db = database_names_cross[0]
#                 secondary_db = database_names_cross[1]
#                 secondary_alias = secondary_db  # 使用数据库名称作为别名
#                 secondary_db_path = os.path.join(script_dir, '..', '..', 'data', 'database', secondary_db, f"{secondary_db}.db")

#             # 构建提示
#             prompt_cross = construct_prompt(sql_skeleton, selected_tables_cross, selected_columns_cross, cross_database=cross_database)

#             # 检查目标文件是否存在，存在则跳过
#             if cross_database:
#                 cross_output_path = os.path.join(cross_output_dir, f"{database_names_cross[0]}_{database_names_cross[1]}")
#             else:
#                 # 如果不跨数据库，保存到单一数据库目录下
#                 cross_output_path = os.path.join(single_output_dir, primary_db)
#             os.makedirs(cross_output_path, exist_ok=True)
#             output_file_cross = os.path.join(cross_output_path, f'generated_sql_statement_{idx}.json')
#             if os.path.exists(output_file_cross):
#                 print(f"文件已存在，跳过生成：{output_file_cross}")
#             else:
#                 # 尝试生成并验证 SQL，最多重试 MAX_RETRIES 次
#                 for attempt in range(1, MAX_RETRIES + 1):
#                     print(f"生成跨数据库 SQL，第 {attempt} 次尝试...")
#                     sql_statement_cross = generate_text(prompt_cross)
#                     if not sql_statement_cross:
#                         print("生成 SQL 失败，重试...")
#                         continue
#                     sql_statement_cross = extract_sql_from_response(sql_statement_cross)

#                     if is_valid_sql(sql_statement_cross) and sql_statement_cross.lower().startswith('select'):
#                         if cross_database:
#                             if not secondary_db_path or not os.path.exists(secondary_db_path):
#                                 print(f"附加的数据库文件不存在：{secondary_db_path}，跳过生成。")
#                                 break

#                             # 执行跨数据库的 SQL
#                             primary_db_path = os.path.join(script_dir, '..', '..', 'data', 'database', primary_db, f"{primary_db}.db")
#                             results, success = execute_cross_db_sql(sql_statement_cross, primary_db_path, secondary_db_path, secondary_alias)
#                         else:
#                             # 获取数据库路径
#                             db_path = os.path.join(script_dir, '..', '..', 'data', 'database', primary_db, f"{primary_db}.db")
#                             if not os.path.exists(db_path):
#                                 print(f"数据库文件不存在：{db_path}，跳过生成。")
#                                 break
#                             results, success = execute_single_db_sql(sql_statement_cross, db_path)

#                         if success:
#                             # 准备保存的数据
#                             if cross_database:
#                                 save_data_cross = {
#                                     'sql': sql_statement_cross,
#                                     'results': results,
#                                     'sql skeleton': sql_skeleton,
#                                     'databases': database_names_cross,
#                                     'tables': {},
#                                 }
#                                 for table in selected_tables_cross:
#                                     db_name = table.split('.')[0]
#                                     table_name = table
#                                     column_names = selected_columns_cross[table]
#                                     if db_name not in save_data_cross['tables']:
#                                         save_data_cross['tables'][db_name] = {}
#                                     save_data_cross['tables'][db_name][table_name] = column_names
#                             else:
#                                 save_data_cross = {
#                                     'sql': sql_statement_cross,
#                                     'results': results,
#                                     'sql skeleton': sql_skeleton,
#                                     'database': primary_db,
#                                     'tables': {},
#                                 }
#                                 for table in selected_tables_cross:
#                                     table_name = table
#                                     column_names = selected_columns_cross[table]
#                                     save_data_cross['tables'][table_name] = column_names

#                             # 保存生成的 SQL 语句到单独的 JSON 文件
#                             with open(output_file_cross, 'w', encoding='utf-8') as f:
#                                 json.dump(save_data_cross, f, ensure_ascii=False, indent=2)
#                             print(f"成功生成并保存跨数据库 SQL：{output_file_cross}")
#                             break
#                         else:
#                             print(f"执行 SQL 失败或无结果，重试...")
#                     else:
#                         print(f"生成的跨数据库 SQL 语句无效，重试...")

#                     if attempt == MAX_RETRIES:
#                         print(f"已达到最大重试次数，跳过生成：{output_file_cross}")

# if __name__ == '__main__':
#     main()



import json
import os
import re
from tqdm import tqdm
import random
import sqlparse
from openai import OpenAI
import sqlite3
import torch

from transformers import AutoModelForCausalLM, AutoTokenizer

model_name = ""

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype="auto",
    device_map={
        "": "cuda:0",
    }
)
tokenizer = AutoTokenizer.from_pretrained(model_name)



key = ""

# client = OpenAI(api_key="", base_url="")
client = OpenAI(
    base_url="",
    api_key=key
)

def generate_text(user_input):
    prompt = user_input
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": prompt}
    ]
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )

    # 截断输入序列以避免过长
    model_inputs = tokenizer(
        [text],
        return_tensors="pt",
        max_length=4096,  # 根据模型支持的最大长度设置
        truncation=True  # 启用截断
    ).to(model.device)

    # 使用混合精度训练以减少内存占用
    with torch.cuda.amp.autocast():
        generated_ids = model.generate(
            **model_inputs,
            max_new_tokens=512,
            do_sample=True,
            temperature=0.6,
            top_p=0.9
        )

    # 提取生成的文本
    generated_ids = [
        output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
    ]
    response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]

    # 释放 GPU 内存
    torch.cuda.empty_cache()

    return response.strip()

# 加载 SQL 框架
def load_sql_skeleton(sql_skeketon_file):
    with open(sql_skeketon_file, 'r', encoding='utf-8') as f:
        sql_skeleton = json.load(f)
    return sql_skeleton

# 加载模式信息
def load_schema(schema_file):
    with open(schema_file, 'r', encoding='utf-8') as f:
        schema = json.load(f)
    return schema

# 提取模式信息
def extract_schema_info(schema):
    schema_info = {
        'tables': [],
        'columns': {}
    }
    for table in schema['tables']:
        table_name = table['table_name']
        schema_info['tables'].append(table_name)
        columns = []
        for column in table['columns']:
            column_name = column['column_name']
            full_column_name = f"{table_name}.{column_name}"
            columns.append(full_column_name)
        schema_info['columns'][table_name] = columns
    return schema_info

# 随机选择表
def select_random_tables(schema_info, num_tables=2):
    selected_tables = random.sample(schema_info['tables'], min(num_tables, len(schema_info['tables'])))
    selected_columns = {}
    for table in selected_tables:
        selected_columns[table] = schema_info['columns'][table]
    return selected_tables, selected_columns

# 构建提示
def construct_prompt(sql_framework, selected_tables, selected_columns, cross_database=False):
    # 将模式信息转换为可读格式，并使用双引号引用表名和列名
    def quote_identifier(identifier):
        return f'"{identifier}"'

    if cross_database:
        # 跨数据库
        tables = ', '.join([quote_identifier(table) for table in selected_tables])
        columns = []
        for table in selected_tables:
            columns.extend([quote_identifier(col) for col in selected_columns[table]])
        columns_str = ', '.join(columns)
        databases = ', '.join(list(set([table.split('.')[0] for table in selected_columns])))
        prompt = f"""
请根据以下 SQL 框架和可用的表名、列名，填充占位符"_"，生成完整且可在 SQLite 上正确执行的 SQL 语句。

严格要求：
- **仅输出最终生成的完整 SQL 语句，不要重复提示内容。**
- **生成的 SQL 要保证语法正确，可以直接在 SQLite 上运行得到结果。**
- **不要添加任何额外的解释、注释或输出格式（代码块，空格等）。**
- **添加的表名、列名、WHERE 条件等内容必须在给定的表和列中。**
- **在给你的（表名.列名）中用双引号单独包住列名，但一定不要包住表名。结果应例如（表名."列名"）**
- **可以对给定的SQL框架做调整，最后生成更合理的SQL语句。**

SQL 框架：
{sql_framework}

可用的数据库：
{databases}

可用的表名：
{tables}

可用的列名（格式：表名.列名）：
{columns_str}

请仅输出生成的完整 SQL 语句：
"""
    else:
        # 单一数据库
        tables = ', '.join([quote_identifier(table) for table in selected_tables])
        columns = []
        for table in selected_tables:
            columns.extend([quote_identifier(col) for col in selected_columns[table]])
        columns_str = ', '.join(columns)
        prompt = f"""
请根据以下 SQL 框架和可用的表名、列名，填充占位符"_"，生成完整且可在 SQLite 上正确执行的 SQL 语句。

严格要求：
- **仅输出最终生成的完整 SQL 语句，不要重复提示内容。**
- **生成的 SQL 要保证语法正确，可以直接在 SQLite 上运行得到结果。**
- **不要添加任何额外的解释、注释或输出格式（代码块，空格等）。**
- **添加的表名、列名、WHERE 条件等内容必须在给定的表和列中。**
- **在给你的（表名.列名）中用双引号单独包住列名，但一定不要包住表名。结果应例如（表名."列名"）**
- **可以对给定的SQL框架做调整，最后生成更合理的SQL语句。**

SQL 框架：
{sql_framework}

可用的表名：
{tables}

可用的列名（格式：表名.列名）：
{columns_str}

请仅输出生成的完整 SQL 语句：
"""
    return prompt.strip()

# 提取模型输出中的 SQL 语句
def extract_sql_from_response(response):
    sql_statement = response.strip()
    # 确保以 "SELECT" 开头
    if not sql_statement.lower().startswith('select'):
        match = re.search(r'(SELECT\s.*)', sql_statement, re.IGNORECASE | re.DOTALL)
        if match:
            sql_statement = match.group(1).strip()
    return sql_statement

# 验证 SQL 语句的语法
def is_valid_sql(sql_statement):
    try:
        parsed = sqlparse.parse(sql_statement)
        if parsed and len(parsed) > 0:
            return True
        else:
            return False
    except Exception:
        return False

# 执行单一数据库的 SQL 语句
def execute_single_db_sql(sql, db_path):
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        cursor.execute(sql)
        results = cursor.fetchall()
        conn.close()
        if results:
            return results, True
        else:
            return 0, False
    except sqlite3.Error as e:
        print(f"SQLite 错误: {e}")
        return 0, False

# 执行跨数据库的 SQL 语句
def execute_cross_db_sql(sql, primary_db_path, secondary_db_path, secondary_alias):
    try:
        conn = sqlite3.connect(primary_db_path)
        cursor = conn.cursor()
        # 附加第二个数据库
        attach_statement = f'ATTACH DATABASE "{secondary_db_path}" AS "{secondary_alias}";'
        cursor.execute(attach_statement)
        # 执行 SQL 查询
        cursor.execute(sql)
        results = cursor.fetchall()
        # 分离数据库
        detach_statement = f'DETACH DATABASE "{secondary_alias}";'
        cursor.execute(detach_statement)
        conn.close()
        if results:
            return results, True
        else:
            return 0, False
    except sqlite3.Error as e:
        print(f"SQLite 错误: {e}")
        return 0, False

# 主函数
def main():
    MAX_RETRIES = 3  # 最大重试次数

    # 获取脚本所在目录
    script_dir = os.path.dirname(os.path.abspath(__file__))

    # 定义输入和输出路径
    new_skeleton_dir = os.path.join(script_dir, '..', '..', 'data', 'new_sql_skeleton')
    new_structure_dir = os.path.join(script_dir, '..', '..', 'data', 'new_sql_structure')
    new_schema_dir = os.path.join(script_dir, '..', '..', 'data', 'new_schema')
    output_dir = os.path.join(script_dir, '..', '..', 'data')
    single_output_dir = os.path.join(output_dir, 'new_sql_single')
    cross_output_dir = os.path.join(output_dir, 'new_sql_cross')
    os.makedirs(single_output_dir, exist_ok=True)
    os.makedirs(cross_output_dir, exist_ok=True)

    # 加载所有数据库的 schema 信息
    schema_files = [f for f in os.listdir(new_schema_dir) if f.endswith('_schema.json')]
    schemas = {}
    for schema_file in schema_files:
        database_name = schema_file.replace('_schema.json', '')
        schema_file_path = os.path.join(new_schema_dir, schema_file)
        schema = load_schema(schema_file_path)
        schema_info = extract_schema_info(schema)
        schemas[database_name] = schema_info

    # 加载所有数据库的 SQL skeleton 文件
    skeleton_files = [f for f in os.listdir(new_skeleton_dir) if f.endswith('_sql_skeleton.json')]

    # print("正在处理新数据库的 SQL skeleton 文件...")

    for skeleton_file in tqdm(skeleton_files, desc="总体进度"):
        # 提取数据库名称
        database_name = skeleton_file.replace('_sql_skeleton.json', '')
        skeleton_file_path = os.path.join(new_skeleton_dir, skeleton_file)
        # print(f"\n正在处理数据库 '{database_name}' 的 SQL skeleton...")

        # 加载对应的 SQL skeletons
        with open(skeleton_file_path, 'r', encoding='utf-8') as f:
            sql_skeletons = json.load(f)

        # 获取对应的 schema 信息
        if database_name not in schemas:
            # print(f"Schema info for database '{database_name}' not found. Skipping.")
            continue
        schema_info = schemas[database_name]

        # 遍历每个 SQL skeleton
        for idx, sql_skeleton in enumerate(tqdm(sql_skeletons, desc=f"{database_name} 的框架处理进度", leave=False)):
            # 生成单一数据库的 SQL
            # 判断 SQL skeleton 需要多少张表（通过 '_' 的数量）
            num_placeholders = sql_skeleton.count('_')
            # 假设每个表至少有一个 '_', 根据 '_' 的数量决定
            if num_placeholders <= 1:
                required_tables_single = 1
            else:
                # 例如，如果有多个 SELECT 或 FROM 子句，可以推断需要多张表
                # 这里简化为最多需要两张表
                required_tables_single = min(2, num_placeholders)

            # 随机选择需要的表数
            if required_tables_single == 1:
                selected_tables_single, selected_columns_single = select_random_tables(schema_info, num_tables=1)
            else:
                selected_tables_single, selected_columns_single = select_random_tables(schema_info, num_tables=2)

            # 构建提示
            prompt_single = construct_prompt(sql_skeleton, selected_tables_single, selected_columns_single, cross_database=False)
            # print(prompt_single)

            # 检查目标文件是否存在，存在则跳过
            single_output_path = os.path.join(single_output_dir, database_name)
            os.makedirs(single_output_path, exist_ok=True)
            output_file_single = os.path.join(single_output_path, f'generated_sql_statement_{idx}.json')
            if os.path.exists(output_file_single):
                print(f"文件已存在，跳过生成：{output_file_single}")
            else:
                # 尝试生成并验证 SQL，最多重试 MAX_RETRIES 次
                for attempt in range(1, MAX_RETRIES + 1):
                    # print(f"生成单一数据库 SQL，第 {attempt} 次尝试...")
                    sql_statement_single = generate_text(prompt_single)
                    if not sql_statement_single:
                        # print("生成 SQL 失败，重试...")
                        continue
                    sql_statement_single = extract_sql_from_response(sql_statement_single)

                    if is_valid_sql(sql_statement_single) and sql_statement_single.lower().startswith('select'):
                        # 获取数据库路径
                        db_path = os.path.join(script_dir, '..', '..', 'data', 'database', database_name, f"{database_name}.db")
                        if not os.path.exists(db_path):
                            # print(f"数据库文件不存在：{db_path}，跳过生成。")
                            break

                        # 执行 SQL
                        results, success = execute_single_db_sql(sql_statement_single, db_path)

                        # 准备保存的数据
                        save_data_single = {
                            'sql': sql_statement_single,
                            'results': results,
                            'sql skeleton': sql_skeleton,
                            'database': database_name,
                            'tables': {},
                        }
                        for table in selected_tables_single:
                            table_name = table.split('.')[0]
                            column_names = selected_columns_single[table]
                            save_data_single['tables'][table_name] = column_names

                        # 保存生成的 SQL 语句到单独的 JSON 文件
                        with open(output_file_single, 'w', encoding='utf-8') as f:
                            json.dump(save_data_single, f, ensure_ascii=False, indent=2)
                        # print(f"成功生成并保存单一数据库 SQL：{output_file_single}")
                        break  # 成功，跳出重试循环

                    else:
                        # print(f"生成的单一数据库 SQL 语句无效，重试...")
                        continue

                    if attempt == MAX_RETRIES:
                        # print(f"已达到最大重试次数，跳过生成：{output_file_single}")
                        continue

            # 生成跨数据库的 SQL
            # 判断是否有足够的数据库进行跨数据库操作
            available_databases = list(schemas.keys())
            if len(available_databases) < 2:
                # print("可用的数据库少于两个，无法生成跨数据库 SQL。")
                continue

            # 随机选择两个不同的数据库
            database_names_cross = random.sample(available_databases, 2)
            schema_info_1 = schemas[database_names_cross[0]]
            schema_info_2 = schemas[database_names_cross[1]]

            # 判断 SQL skeleton 需要多少张表
            if num_placeholders <= 1:
                required_tables_cross = 1
            else:
                required_tables_cross = min(2, num_placeholders)

            # 根据需要的表数，从两个数据库中选择表
            if required_tables_cross == 1:
                # 只选择一个表，随机选择其中一个数据库
                chosen_db = random.choice([0, 1])
                selected_tables_cross, selected_columns_cross = select_random_tables(
                    schema_info_1 if chosen_db == 0 else schema_info_2,
                    num_tables=1
                )
                cross_database = False  # 只用一个数据库
                # 设置附加的数据库路径和别名
                primary_db = database_names_cross[0] if chosen_db == 0 else database_names_cross[1]
                secondary_db = None
                secondary_alias = None
                secondary_db_path = None
            else:
                # 选择两个表，各自来自不同的数据库
                selected_tables_1, selected_columns_1 = select_random_tables(schema_info_1, num_tables=1)
                selected_tables_2, selected_columns_2 = select_random_tables(schema_info_2, num_tables=1)
                selected_tables_cross = selected_tables_1 + selected_tables_2
                selected_columns_cross = {**selected_columns_1, **selected_columns_2}
                cross_database = True
                # 设置附加的数据库路径和别名
                primary_db = database_names_cross[0]
                secondary_db = database_names_cross[1]
                secondary_alias = secondary_db  # 使用数据库名称作为别名
                secondary_db_path = os.path.join(script_dir, '..', '..', 'data', 'database', secondary_db, f"{secondary_db}.db")

            # 构建提示
            prompt_cross = construct_prompt(sql_skeleton, selected_tables_cross, selected_columns_cross, cross_database=cross_database)

            # 检查目标文件是否存在，存在则跳过
            if cross_database:
                cross_output_path = os.path.join(cross_output_dir, f"{database_names_cross[0]}_{database_names_cross[1]}")
            else:
                # 如果不跨数据库，保存到单一数据库目录下
                cross_output_path = os.path.join(single_output_dir, primary_db)
            os.makedirs(cross_output_path, exist_ok=True)
            output_file_cross = os.path.join(cross_output_path, f'generated_sql_statement_{idx}.json')
            if os.path.exists(output_file_cross):
                print(f"文件已存在，跳过生成：{output_file_cross}")
            else:
                continue
                # 尝试生成并验证 SQL，最多重试 MAX_RETRIES 次
                for attempt in range(1, MAX_RETRIES + 1):
                    print(f"生成跨数据库 SQL，第 {attempt} 次尝试...")
                    sql_statement_cross = generate_text(prompt_cross)
                    if not sql_statement_cross:
                        print("生成 SQL 失败，重试...")
                        continue
                    sql_statement_cross = extract_sql_from_response(sql_statement_cross)

                    if is_valid_sql(sql_statement_cross) and sql_statement_cross.lower().startswith('select'):
                        if cross_database:
                            continue
                            if not secondary_db_path or not os.path.exists(secondary_db_path):
                                print(f"附加的数据库文件不存在：{secondary_db_path}，跳过生成。")
                                break

                            # 执行跨数据库的 SQL
                            primary_db_path = os.path.join(script_dir, '..', '..', 'data', 'database', primary_db, f"{primary_db}.db")
                            results, success = execute_cross_db_sql(sql_statement_cross, primary_db_path, secondary_db_path, secondary_alias)
                        else:
                            # 获取数据库路径
                            db_path = os.path.join(script_dir, '..', '..', 'data', 'database', primary_db, f"{primary_db}.db")
                            if not os.path.exists(db_path):
                                print(f"数据库文件不存在：{db_path}，跳过生成。")
                                break
                            results, success = execute_single_db_sql(sql_statement_cross, db_path)


                        # 准备保存的数据
                        if cross_database:
                            save_data_cross = {
                                'sql': sql_statement_cross,
                                'results': results,
                                'sql skeleton': sql_skeleton,
                                'databases': database_names_cross,
                                'tables': {},
                            }
                            for table in selected_tables_cross:
                                db_name = table.split('.')[0]
                                table_name = table
                                column_names = selected_columns_cross[table]
                                if db_name not in save_data_cross['tables']:
                                    save_data_cross['tables'][db_name] = {}
                                save_data_cross['tables'][db_name][table_name] = column_names
                        else:
                            save_data_cross = {
                                'sql': sql_statement_cross,
                                'results': results,
                                'sql skeleton': sql_skeleton,
                                'database': primary_db,
                                'tables': {},
                            }
                            for table in selected_tables_cross:
                                table_name = table
                                column_names = selected_columns_cross[table]
                                save_data_cross['tables'][table_name] = column_names

                        # 保存生成的 SQL 语句到单独的 JSON 文件
                        with open(output_file_cross, 'w', encoding='utf-8') as f:
                            json.dump(save_data_cross, f, ensure_ascii=False, indent=2)
                        print(f"成功生成并保存跨数据库 SQL：{output_file_cross}")
                        break

                    else:
                        print(f"生成的跨数据库 SQL 语句无效，重试...")

                    if attempt == MAX_RETRIES:
                        print(f"已达到最大重试次数，跳过生成：{output_file_cross}")

if __name__ == '__main__':
    main()
