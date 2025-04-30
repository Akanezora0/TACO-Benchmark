# fill the api key and model name

import json
import os
import re
from tqdm import tqdm
import random
import sqlparse
from openai import OpenAI
import sqlite3

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
    model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

    generated_ids = model.generate(
        **model_inputs,
        max_new_tokens=512
    )
    generated_ids = [
        output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
    ]

    response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
    return response.strip()
    # try:
    #     response = client.chat.completions.create(
    #         model="gpt-3.5-turbo",
    #         # model="deepseek-chat",
    #         temperature=0.5,
    #         messages=[
    #             {"role": "system", "content": "You are a helpful assistant"},
    #             {"role": "user", "content": user_input},
    #         ],
    #     )
    #     assistant_reply = response.choices[0].message.content
    #     # print(f"Assistant: {assistant_reply}")
    #     return assistant_reply.strip()
    # except Exception as e:
    #     # print(f"An error occurred: {e}")
    #     return None

# load the SQL framework
def load_sql_skeleton(sql_skeketon_file):
    with open(sql_skeketon_file, 'r', encoding='utf-8') as f:
        sql_skeleton = json.load(f)
    return sql_skeleton

# load the schema information
def load_schema(schema_file):
    with open(schema_file, 'r', encoding='utf-8') as f:
        schema = json.load(f)
    return schema

# extract the schema information
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

# select the random tables
def select_random_tables(schema_info, num_tables=2):
    selected_tables = random.sample(schema_info['tables'], min(num_tables, len(schema_info['tables'])))
    selected_columns = {}
    for table in selected_tables:
        selected_columns[table] = schema_info['columns'][table]
    return selected_tables, selected_columns

# construct the prompt
def construct_prompt(sql_framework, selected_tables, selected_columns, cross_database=False):
    # quote the identifier
    def quote_identifier(identifier):
        return f'"{identifier}"'

    if cross_database:
        # cross-database
        tables = ', '.join([quote_identifier(table) for table in selected_tables])
        columns = []
        for table in selected_tables:
            columns.extend([quote_identifier(col) for col in selected_columns[table]])
        columns_str = ', '.join(columns)
        databases = ', '.join(list(set([table.split('.')[0] for table in selected_columns])))
        prompt = f"""
Please fill the placeholder "_" based on the following SQL framework and the available table names and column names, generate a complete and correct SQL statement that can be executed on SQLite.

Strict requirements:
- **Only output the final generated complete SQL statement, do not repeat the prompt content.**
- **The generated SQL must be grammatically correct and can be executed directly on SQLite to get results.**
- **Do not add any additional explanations, comments, or output formats (code blocks, spaces, etc.).**
- **Add the table names, column names, WHERE conditions, etc. must be in the given tables and columns.**
- **In the (table_name.column_name) given to you, quote the column name with double quotes, but do not quote the table name. The result should be like (table_name."column_name").**
- **You can adjust the given SQL framework to generate a more reasonable SQL statement.**

SQL framework:
{sql_framework}

Available databases:
{databases}

Available table names:
{tables}

Available column names (format: table_name.column_name):
{columns_str}

Please only output the final generated complete SQL statement:
"""
    else:
        # 单一数据库
        tables = ', '.join([quote_identifier(table) for table in selected_tables])
        columns = []
        for table in selected_tables:
            columns.extend([quote_identifier(col) for col in selected_columns[table]])
        columns_str = ', '.join(columns)
        prompt = f"""
Please fill the placeholder "_" based on the following SQL framework and the available table names and column names, generate a complete and correct SQL statement that can be executed on SQLite.

Strict requirements:
- **Only output the final generated complete SQL statement, do not repeat the prompt content.**
- **The generated SQL must be grammatically correct and can be executed directly on SQLite to get results.**
- **Do not add any additional explanations, comments, or output formats (code blocks, spaces, etc.).**
- **Add the table names, column names, WHERE conditions, etc. must be in the given tables and columns.**
- **In the (table_name.column_name) given to you, quote the column name with double quotes, but do not quote the table name. The result should be like (table_name."column_name").**
- **You can adjust the given SQL framework to generate a more reasonable SQL statement.**

SQL framework:
{sql_framework}

Available table names:
{tables}

Available column names (format: table_name.column_name):
{columns_str}

Please only output the final generated complete SQL statement:
"""
    return prompt.strip()

# extract the SQL statement from the model output
def extract_sql_from_response(response):
    sql_statement = response.strip()
    # ensure the SQL statement starts with "SELECT"
    if not sql_statement.lower().startswith('select'):
        match = re.search(r'(SELECT\s.*)', sql_statement, re.IGNORECASE | re.DOTALL)
        if match:
            sql_statement = match.group(1).strip()
    return sql_statement

# validate the syntax of the SQL statement
def is_valid_sql(sql_statement):
    try:
        parsed = sqlparse.parse(sql_statement)
        if parsed and len(parsed) > 0:
            return True
        else:
            return False
    except Exception:
        return False

# execute the SQL statement on a single database
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
        print(f"SQLite error: {e}")
        return 0, False

# execute the SQL statement on a cross-database
def execute_cross_db_sql(sql, primary_db_path, secondary_db_path, secondary_alias):
    try:
        conn = sqlite3.connect(primary_db_path)
        cursor = conn.cursor()
        # attach the secondary database
        attach_statement = f'ATTACH DATABASE "{secondary_db_path}" AS "{secondary_alias}";'
        cursor.execute(attach_statement)
        # execute the SQL query
        cursor.execute(sql)
        results = cursor.fetchall()
        # detach the database
        detach_statement = f'DETACH DATABASE "{secondary_alias}";'
        cursor.execute(detach_statement)
        conn.close()
        if results:
            return results, True
        else:
            return 0, False
    except sqlite3.Error as e:
        print(f"SQLite error: {e}")
        return 0, False


def main():
    MAX_RETRIES = 3  # maximum number of retries

    script_dir = os.path.dirname(os.path.abspath(__file__))

    new_skeleton_dir = os.path.join(script_dir, '..', '..', 'data', 'new_sql_skeleton')
    new_structure_dir = os.path.join(script_dir, '..', '..', 'data', 'new_sql_structure')
    new_schema_dir = os.path.join(script_dir, '..', '..', 'data', 'new_schema')
    output_dir = os.path.join(script_dir, '..', '..', 'data')
    single_output_dir = os.path.join(output_dir, 'new_sql_single')
    cross_output_dir = os.path.join(output_dir, 'new_sql_cross')
    os.makedirs(single_output_dir, exist_ok=True)
    os.makedirs(cross_output_dir, exist_ok=True)


    schema_files = [f for f in os.listdir(new_schema_dir) if f.endswith('_schema.json')]
    schemas = {}
    for schema_file in schema_files:
        database_name = schema_file.replace('_schema.json', '')
        schema_file_path = os.path.join(new_schema_dir, schema_file)
        schema = load_schema(schema_file_path)
        schema_info = extract_schema_info(schema)
        schemas[database_name] = schema_info


    skeleton_files = [f for f in os.listdir(new_skeleton_dir) if f.endswith('_sql_skeleton.json')]

    print("Processing the SQL skeleton files of the new database...")

    for skeleton_file in tqdm(skeleton_files, desc="Overall progress"):
        database_name = skeleton_file.replace('_sql_skeleton.json', '')
        skeleton_file_path = os.path.join(new_skeleton_dir, skeleton_file)
        print(f"\nProcessing the SQL skeleton of the database '{database_name}'...")

        with open(skeleton_file_path, 'r', encoding='utf-8') as f:
            sql_skeletons = json.load(f)

        if database_name not in schemas:
            print(f"Schema info for database '{database_name}' not found. Skipping.")
            continue
        schema_info = schemas[database_name]

        for idx, sql_skeleton in enumerate(tqdm(sql_skeletons, desc=f"{database_name} framework processing progress", leave=False)):
            # generate the SQL statement on a single database
            # determine the number of tables needed (based on the number of '_')
            num_placeholders = sql_skeleton.count('_')
            # assume each table has at least one '_', determine the number of tables needed
            if num_placeholders <= 1:
                required_tables_single = 1
            else:
                # for example, if there are multiple SELECT or FROM clauses, it can be inferred that more than two tables are needed
                # here simplified to at most two tables
                required_tables_single = min(2, num_placeholders)

            # randomly select the number of tables needed
            if required_tables_single == 1:
                selected_tables_single, selected_columns_single = select_random_tables(schema_info, num_tables=1)
            else:
                selected_tables_single, selected_columns_single = select_random_tables(schema_info, num_tables=2)

            # construct the prompt
            prompt_single = construct_prompt(sql_skeleton, selected_tables_single, selected_columns_single, cross_database=False)
            # print(prompt_single)

            # check if the target file exists, skip if it exists
            single_output_path = os.path.join(single_output_dir, database_name)
            os.makedirs(single_output_path, exist_ok=True)
            output_file_single = os.path.join(single_output_path, f'generated_sql_statement_{idx}.json')
            if os.path.exists(output_file_single):
                print(f"File already exists, skip generation: {output_file_single}")
            else:
                # try to generate and validate the SQL, up to MAX_RETRIES times
                for attempt in range(1, MAX_RETRIES + 1):
                    print(f"Generating single database SQL, attempt {attempt}...")
                    sql_statement_single = generate_text(prompt_single)
                    if not sql_statement_single:
                        print("Failed to generate SQL, retry...")
                        continue
                    sql_statement_single = extract_sql_from_response(sql_statement_single)

                    if is_valid_sql(sql_statement_single) and sql_statement_single.lower().startswith('select'):
                        # get the database path
                        db_path = os.path.join(script_dir, '..', '..', 'data', 'database', database_name, f"{database_name}.db")
                        if not os.path.exists(db_path):
                            print(f"Database file does not exist: {db_path}, skip generation.")
                            break

                        results, success = execute_single_db_sql(sql_statement_single, db_path)

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

                        with open(output_file_single, 'w', encoding='utf-8') as f:
                            json.dump(save_data_single, f, ensure_ascii=False, indent=2)
                        print(f"Successfully generated and saved single database SQL: {output_file_single}")
                        break

                    else:
                        print(f"The generated single database SQL is invalid, retry...")

                    if attempt == MAX_RETRIES:
                        print(f"Reach the maximum number of retries, skip generation: {output_file_single}")

            # generate the SQL statement on a cross-database
            # determine if there are enough databases for cross-database operations
            available_databases = list(schemas.keys())
            if len(available_databases) < 2:
                print("Less than two available databases, cannot generate cross-database SQL.")
                continue

            # randomly select two different databases
            database_names_cross = random.sample(available_databases, 2)
            schema_info_1 = schemas[database_names_cross[0]]
            schema_info_2 = schemas[database_names_cross[1]]

            # determine the number of tables needed (based on the number of '_')
            if num_placeholders <= 1:
                required_tables_cross = 1
            else:
                required_tables_cross = min(2, num_placeholders)

            # according to the number of tables needed, select the tables from the two databases
            if required_tables_cross == 1:
                # select one table, randomly choose one of the two databases
                chosen_db = random.choice([0, 1])
                selected_tables_cross, selected_columns_cross = select_random_tables(
                    schema_info_1 if chosen_db == 0 else schema_info_2,
                    num_tables=1
                )
                cross_database = False  # only use one database
                # set the additional database path and alias
                primary_db = database_names_cross[0] if chosen_db == 0 else database_names_cross[1]
                secondary_db = None
                secondary_alias = None
                secondary_db_path = None
            else:
                # select two tables, each from different databases
                selected_tables_1, selected_columns_1 = select_random_tables(schema_info_1, num_tables=1)
                selected_tables_2, selected_columns_2 = select_random_tables(schema_info_2, num_tables=1)
                selected_tables_cross = selected_tables_1 + selected_tables_2
                selected_columns_cross = {**selected_columns_1, **selected_columns_2}
                cross_database = True
                # set the additional database path and alias
                primary_db = database_names_cross[0]
                secondary_db = database_names_cross[1]
                secondary_alias = secondary_db  # use the database name as the alias
                secondary_db_path = os.path.join(script_dir, '..', '..', 'data', 'database', secondary_db, f"{secondary_db}.db")

            # construct the prompt
            prompt_cross = construct_prompt(sql_skeleton, selected_tables_cross, selected_columns_cross, cross_database=cross_database)

            # check if the target file exists, skip if it exists
            if cross_database:
                cross_output_path = os.path.join(cross_output_dir, f"{database_names_cross[0]}_{database_names_cross[1]}")
            else:
                # if not cross-database, save to the single database directory
                cross_output_path = os.path.join(single_output_dir, primary_db)
            os.makedirs(cross_output_path, exist_ok=True)
            output_file_cross = os.path.join(cross_output_path, f'generated_sql_statement_{idx}.json')
            if os.path.exists(output_file_cross):
                print(f"File already exists, skip generation: {output_file_cross}")
            else:
                # try to generate and validate the SQL, up to MAX_RETRIES times
                for attempt in range(1, MAX_RETRIES + 1):
                    print(f"Generating cross-database SQL, attempt {attempt}...")
                    sql_statement_cross = generate_text(prompt_cross)
                    if not sql_statement_cross:
                        print("Failed to generate SQL, retry...")
                        continue
                    sql_statement_cross = extract_sql_from_response(sql_statement_cross)

                    if is_valid_sql(sql_statement_cross) and sql_statement_cross.lower().startswith('select'):
                        if cross_database:
                            if not secondary_db_path or not os.path.exists(secondary_db_path):
                                print(f"The additional database file does not exist: {secondary_db_path}, skip generation.")
                                break

                            # execute the cross-database SQL
                            primary_db_path = os.path.join(script_dir, '..', '..', 'data', 'database', primary_db, f"{primary_db}.db")
                            results, success = execute_cross_db_sql(sql_statement_cross, primary_db_path, secondary_db_path, secondary_alias)
                        else:
                            # get the database path
                            db_path = os.path.join(script_dir, '..', '..', 'data', 'database', primary_db, f"{primary_db}.db")
                            if not os.path.exists(db_path):
                                print(f"The database file does not exist: {db_path}, skip generation.")
                                break
                            results, success = execute_single_db_sql(sql_statement_cross, db_path)


                        # prepare the data to save
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

                        # save the generated SQL statement to a separate JSON file
                        with open(output_file_cross, 'w', encoding='utf-8') as f:
                            json.dump(save_data_cross, f, ensure_ascii=False, indent=2)
                        print(f"Successfully generated and saved cross-database SQL: {output_file_cross}")
                        break

                    else:
                        print(f"The generated cross-database SQL is invalid, retry...")

                    if attempt == MAX_RETRIES:
                        print(f"Reach the maximum number of retries, skip generation: {output_file_cross}")

if __name__ == '__main__':
    main()
