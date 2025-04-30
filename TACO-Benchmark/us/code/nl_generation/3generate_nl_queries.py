# fill the api key and model name


import json
import os
import re
from tqdm import tqdm
from openai import OpenAI
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

client = OpenAI(api_key="", base_url="")

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


def main():
    # manually specify the start and end index of the SQL statements to process (starting from 0)
    # here remove the start and end index, process all files
    # start_idx = 0   # start index, include
    # end_idx = 0   # end index, include

    script_dir = os.path.dirname(os.path.abspath(__file__))

    sql_statements_dir = os.path.join(script_dir, '..', '..', 'data')
    output_dir = os.path.join(script_dir, '..', '..', 'data')
    # prompts_dir = os.path.join(output_dir, 'prompts')
    os.makedirs(output_dir, exist_ok=True)
    # os.makedirs(prompts_dir, exist_ok=True)

    # example list (can be adjusted or loaded from external examples)
    examples = [
        {
            'problem_description': 'Query the potential hazards in Shidi Village, Han Shiqiao, Yang Town, Shunyi District, including possible health and safety issues.',
            'natural_language_query': 'A citizen reported that at the bridgehead of Shidi Village, Han Shiqiao, Yang Town, Shunyi District, there are vendors selling fireworks and firecrackers for children, which poses safety hazards. They hope for prompt verification and handling, calling to report the illegal sale of fireworks and firecrackers.'
        },
        {
            'problem_description': 'Search for corresponding records in marriage registration information based on the names of both parents.',
            'natural_language_query': 'A citizen reported that when handling birth registration for their newborn, the system showed that the marriage certificate information of parents Xiao Ming and Xiao Hong did not match. Their marriage certificate was registered on March 25, 2022, at the Haidian District Civil Affairs Bureau. They hope to verify and resolve this situation, calling to report the mismatch in marriage certificate information.'
        },
        {
            'problem_description': 'Check whether this enterprise has any issues regarding tax payment.',
            'natural_language_query': 'A citizen reported purchasing Coca-Cola from Beijing Jinhui Fa Trading Co., Ltd. (address: Shunyi District Shunxin Shimen Agricultural Wholesale Market) through Pinduoduo, but the company has been delaying issuing an invoice with various excuses. Order number: 240123149364469331471. Calling to report Jinhui Fa\'s refusal to issue an invoice.'
        },
    ]

    single_sql_dir = os.path.join(sql_statements_dir, 'new_sql_single')
    for database_name in os.listdir(single_sql_dir):
        db_sql_dir = os.path.join(single_sql_dir, database_name)
        if not os.path.isdir(db_sql_dir):
            continue
        for file_name in tqdm(os.listdir(db_sql_dir), desc=f"Processing SQL files for single database '{database_name}'"):
            if not file_name.endswith(".json"):
                continue
            file_idx = int(re.findall(r'\d+', file_name)[0]) if re.findall(r'\d+', file_name) else "unknown"
            output_file = os.path.join(output_dir, 'new_sql_nl_single', database_name, f'generated_nl_query_{file_idx}.json')
            os.makedirs(os.path.dirname(output_file), exist_ok=True)

            # check if the output file already exists
            if os.path.exists(output_file):
                print(f"Already exists: {output_file}, skip")
                continue  # skip the current file, continue to the next

            sql_file_path = os.path.join(db_sql_dir, file_name)
            # print(f"Processing SQL file: {sql_file_path}")

            # load the SQL statement
            with open(sql_file_path, 'r', encoding='utf-8') as f:
                try:
                    sql_data = json.load(f)
                    sql_query = sql_data['sql']
                    results = sql_data['results']
                    sql_skeleton = sql_data['sql skeleton']
                    database = sql_data['database']
                    tables = sql_data['tables']
                    
                except json.JSONDecodeError as e:
                    # print(f"Failed to parse JSON file {sql_file_path}: {e}")
                    continue
                except KeyError:
                    # print(f"JSON file {sql_file_path} is missing fields, skip")
                    continue

            # define the list to save prompts
            prompts = []

            # stage 1: generate problem description
            problem_description_prompt = f"""
Please read the following SQL statement, analyze it step by step, and use detailed language to describe the query intent, including the involved tables, columns, conditions, and possible business scenarios.

SQL statement:
{sql_query}

Please output in the following format:
- **Query intent**: briefly explain the purpose of the SQL statement.
- **Detailed description**: explain in detail the tables, columns, conditions, and other information involved in the query.
- **Business scenario**: what business scenarios the queried tables, columns, conditions, etc. can be used to solve.
- **User description**: actual problems or needs that users may encounter in the above business scenarios.
"""

            prompts.append({
                'idx': file_idx,
                'sql_query': sql_query,
                'stage': 'problem_description',
                'prompt': problem_description_prompt
            })

            problem_description = generate_text(problem_description_prompt)

            if not problem_description:
                print(f"Failed to generate problem description, skip file: {sql_file_path}")
                continue

            # stage 2: generate natural language query
            # build the example text
            example_text = ""
            if examples:
                for example in examples:
                    example_text += f"### Example\n"
                    example_text += f"- Problem description:\n{example['problem_description']}\n"
                    example_text += f"- Natural language query:\n{example['natural_language_query']}\n\n"

            natural_language_query_prompt = f"""
Based on the following problem description, simulate a real user's question, and generate a natural language query. Requirements:
- The language style should be like a real user's question, which may include unclear intent and redundant information.
- The user's intent may not be clear, the user may not specify the specific table or column to query, only describe the scenario they encountered, and include vague or uncertain expressions.
- The user tends to describe their scenario in detail, which may include their personal information or related event information.
- The expression may not be professional, which may include vague or uncertain expressions.
- It may include emotional tendency, such as complaints, dissatisfaction, or confusion.

Strict requirements:
- **The generated text is in English.**
- **The natural language query should not include specific table names or column names, as the user does not understand the underlying database structure.**

{example_text}
### Problem description:
{problem_description}

Please generate the corresponding natural language query based on the above problem description.
"""

            prompts.append({
                'idx': file_idx,
                'problem_description': problem_description,
                'stage': 'natural_language_query',
                'prompt': natural_language_query_prompt
            })

            natural_language_query = generate_text(natural_language_query_prompt)

            if not natural_language_query:
                print(f"Failed to generate natural language query, skip file: {sql_file_path}")
                continue

            # stage 3: generate table names
            table_name_prompt = f"""
Please analyze the following SQL statement, extract all the table names involved, and list them separated by commas:

SQL statement:
{sql_query}

Please return the list of table names, for example:
table1, table2, table3
- Only return the list of table names, do not include other information
- The table names may contain other symbols, please correctly identify the table names
"""
            # generate table names
            table_name = generate_text(table_name_prompt)

            if not table_name:
                print(f"Failed to generate table names, skip file: {sql_file_path}")
                continue

            result = {
                'sql': sql_query,
                'problem_description': problem_description,
                'natural_language_query': natural_language_query,
                'results': results,
                'sql skeleton': sql_skeleton,
                'database': database,
                'tables': tables
            }
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(result, f, ensure_ascii=False, indent=2)

            prompts_dir = os.path.join(output_dir, 'new_sql_nl_single', database_name, 'prompts')
            os.makedirs(prompts_dir, exist_ok=True)
            prompts_file = os.path.join(prompts_dir, f'prompts_single_{file_idx}.json')
            if os.path.exists(prompts_file):
                print(f"Already exists: {prompts_file}, skip")
            else:
                with open(prompts_file, 'w', encoding='utf-8') as f:
                    json.dump(prompts, f, ensure_ascii=False, indent=2)

            print(f"Generated and saved natural language query: {output_file}")

if __name__ == '__main__':
    main()
