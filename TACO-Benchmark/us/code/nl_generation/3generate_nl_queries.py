# import json
# import os
# import re
# from tqdm import tqdm
# from openai import OpenAI

# client = OpenAI(api_key="", base_url="")

# def generate_text(user_input):
#     try:
#         response = client.chat.completions.create(
#             model="",
#             temperature=0.5,
#             messages=[
#                 {"role": "system", "content": "You are a helpful assistant"},
#                 {"role": "user", "content": user_input},
#             ],
#         )
#         assistant_reply = response.choices[0].message.content
#         return assistant_reply.strip()
#     except openai.error.OpenAIError as e:
#         print(f"API 调用出错：{e}")
#         return ""

# # 主函数
# def main():
#     # 手动指定要处理的 SQL 语句的起始和结束序号（从 0 开始计数）
#     # 这里移除起始和结束序号，处理所有文件
#     # start_idx = 0   # 起始序号，包含
#     # end_idx = 0   # 结束序号，包含

#     # 获取脚本所在目录
#     script_dir = os.path.dirname(os.path.abspath(__file__))

#     # 定义输入和输出路径
#     sql_statements_dir = os.path.join(script_dir, '..', '..', 'data')
#     output_dir = os.path.join(script_dir, '..', '..', 'data')
#     # prompts_dir = os.path.join(output_dir, 'prompts')
#     os.makedirs(output_dir, exist_ok=True)
#     # os.makedirs(prompts_dir, exist_ok=True)

#     # 示例列表（可根据需要调整或加载外部示例）
#     examples = [
#         {
#             'problem_description': '查询顺义区杨镇汉石桥事地村的隐患名称，包括可能存在的卫生、安全等问题。',
#             'natural_language_query': '市民反映，顺义区杨镇汉石桥事地村，在桥头有售卖小孩玩的烟花爆竹，存在安全隐患，希望尽快进行核实处理，来电反映违规售卖烟花爆竹问题。'
#         },
#         {
#             'problem_description': '根据父母双方的姓名，在婚姻登记信息中查找有无存在对应记录。',
#             'natural_language_query': '市民反映，新生儿办理出生一件事的时候，显示父母双方小明和小红结婚证信息不匹配，自己的结婚证是22年3月25号在海淀区民政局登记的，希望能帮助核实处理一下是什么情况，来电反映结婚证信息不匹配问题。'
#         },
#         {
#             'problem_description': '查询这家企业在纳税方面有没有问题。',
#             'natural_language_query': '市民反映，通过拼多多购买北京金汇发商贸有限公司的可乐，地址：顺义区顺鑫石门农产品批发市场，但是购买后一直不给开发票，找各种理由推拖，订单号：240123149364469331471，来电反映金汇发不给开发票问题。'
#         },
#     ]

#     # 遍历单一数据库的 SQL JSON 文件
#     single_sql_dir = os.path.join(sql_statements_dir, 'new_sql_single')
#     for database_name in os.listdir(single_sql_dir):
#         db_sql_dir = os.path.join(single_sql_dir, database_name)
#         if not os.path.isdir(db_sql_dir):
#             continue
#         for file_name in tqdm(os.listdir(db_sql_dir), desc=f"处理单一数据库 '{database_name}' 的 SQL 文件"):
#             if not file_name.endswith(".json"):
#                 continue
#             file_idx = int(re.findall(r'\d+', file_name)[0]) if re.findall(r'\d+', file_name) else "unknown"
#             output_file = os.path.join(output_dir, 'new_sql_nl_single', database_name, f'generated_nl_query_{file_idx}.json')
#             os.makedirs(os.path.dirname(output_file), exist_ok=True)

#             # 检查输出文件是否已经存在
#             if os.path.exists(output_file):
#                 print(f"已存在：{output_file}，跳过")
#                 continue  # 跳过当前文件，继续下一个

#             sql_file_path = os.path.join(db_sql_dir, file_name)
#             print(f"处理 SQL 文件：{sql_file_path}")

#             # 加载 SQL 语句
#             with open(sql_file_path, 'r', encoding='utf-8') as f:
#                 try:
#                     sql_data = json.load(f)
#                     sql_query = sql_data['sql']
#                     results = sql_data['results']
#                     sql_skeleton = sql_data['sql skeleton']
#                     database = sql_data['database']
#                     tables = sql_data['tables']
                    
#                 except json.JSONDecodeError as e:
#                     print(f"无法解析 JSON 文件 {sql_file_path}：{e}")
#                     continue
#                 except KeyError:
#                     print(f"JSON 文件 {sql_file_path} 缺少字段，跳过")
#                     continue

#             # 定义保存提示的列表
#             prompts = []

#             # 阶段 1：生成问题描述
#             problem_description_prompt = f"""
# 请阅读以下 SQL 语句，逐步分析并用详细的语言描述该 SQL 语句的查询意图，包括涉及的表、列、条件，以及可能的业务场景。

# SQL 语句：
# {sql_query}

# 请按照以下格式输出：
# - **查询意图**：简要说明 SQL 语句的目的。
# - **详细描述**：详细解释查询涉及的表、列、条件等信息。
# - **业务场景**：查询到的表、列、条件等可以用来解决什么业务场景。
# - **用户描述**：上面的业务场景下用户可能遇到的实际问题或需求。
# """
#             # 保存提示
#             prompts.append({
#                 'idx': file_idx,
#                 'sql_query': sql_query,
#                 'stage': 'problem_description',
#                 'prompt': problem_description_prompt
#             })

#             problem_description = generate_text(problem_description_prompt)

#             if not problem_description:
#                 print(f"未能生成问题描述，跳过文件：{sql_file_path}")
#                 continue

#             # 阶段 2：生成自然语言查询
#             # 构建示例文本
#             example_text = ""
#             if examples:
#                 for example in examples:
#                     example_text += f"### 示例\n"
#                     example_text += f"- 问题描述：\n{example['problem_description']}\n"
#                     example_text += f"- 自然语言查询：\n{example['natural_language_query']}\n\n"

# # - 尽量以“市民反映”开头的方式来转述。
#             natural_language_query_prompt = f"""
# 根据以下问题描述，模拟真实用户的提问，生成一段自然语言查询。要求如下：
# - 语言风格要符合真实场景中的用户提问，可能包含意图不明、信息冗余等特点。
# - 用户意图可能不够明确，用户无法指定具体要查询什么表或列，只描述自己遇到的场景，包含模糊或不确定的表述。
# - 用户偏向于详细描述自己遇到的场景，可能包含自己的个人信息或者相关事件的信息。
# - 表达方式可以不够专业，可能包含模糊或不确定的表述。
# - 有可能融入情感倾向，如抱怨、不满、疑惑等。

# 严格要求：
# - **用户自然语言中不要出现具体的表名或列名，因为用户不了解底层数据库结构。**

# {example_text}
# ### 问题描述：
# {problem_description}

# 请根据上述问题描述，生成对应的自然语言查询。
# """
#             # 保存提示
#             prompts.append({
#                 'idx': file_idx,
#                 'problem_description': problem_description,
#                 'stage': 'natural_language_query',
#                 'prompt': natural_language_query_prompt
#             })

#             natural_language_query = generate_text(natural_language_query_prompt)

#             if not natural_language_query:
#                 print(f"未能生成自然语言查询，跳过文件：{sql_file_path}")
#                 continue

# #             # 阶段 3：生成表格名
# #             table_name_prompt = f"""
# # 请分析以下 SQL 语句，提取其中涉及的所有表名，并将它们以逗号分隔列出：

# # SQL 语句：
# # {sql_query}

# # 请返回表名列表，例如：
# # 表名：table1, table2, table3
# # - 只需要返回表名的list，不要包含其他信息
# # - 表名内部可能有其他符号，请正确识别表名
# # """
#             # # 生成表格名
#             # table_name = generate_text(table_name_prompt)

#             # if not table_name:
#             #     print(f"未能生成表名，跳过文件：{sql_file_path}")
#             #     continue

#             # 保存结果到单独的文件
#             result = {
#                 'sql': sql_query,
#                 'problem_description': problem_description,
#                 'natural_language_query': natural_language_query,
#                 'results': results,
#                 'sql skeleton': sql_skeleton,
#                 'database': database,
#                 'tables': tables
#             }
#             with open(output_file, 'w', encoding='utf-8') as f:
#                 json.dump(result, f, ensure_ascii=False, indent=2)

#             # 保存提示到单独的文件
#             prompts_dir = os.path.join(output_dir, 'new_sql_nl_single', database_name, 'prompts')
#             os.makedirs(prompts_dir, exist_ok=True)
#             prompts_file = os.path.join(prompts_dir, f'prompts_single_{file_idx}.json')
#             if os.path.exists(prompts_file):
#                 print(f"已存在：{prompts_file}，跳过")
#             else:
#                 with open(prompts_file, 'w', encoding='utf-8') as f:
#                     json.dump(prompts, f, ensure_ascii=False, indent=2)

#             print(f"已生成并保存自然语言查询：{output_file}")

# if __name__ == '__main__':
#     main()




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

# 主函数
def main():
    # 手动指定要处理的 SQL 语句的起始和结束序号（从 0 开始计数）
    # 这里移除起始和结束序号，处理所有文件
    # start_idx = 0   # 起始序号，包含
    # end_idx = 0   # 结束序号，包含

    # 获取脚本所在目录
    script_dir = os.path.dirname(os.path.abspath(__file__))

    # 定义输入和输出路径
    sql_statements_dir = os.path.join(script_dir, '..', '..', 'data')
    output_dir = os.path.join(script_dir, '..', '..', 'data')
    # prompts_dir = os.path.join(output_dir, 'prompts')
    os.makedirs(output_dir, exist_ok=True)
    # os.makedirs(prompts_dir, exist_ok=True)

    # 示例列表（可根据需要调整或加载外部示例）
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

    # 遍历单一数据库的 SQL JSON 文件
    single_sql_dir = os.path.join(sql_statements_dir, 'new_sql_single')
    for database_name in os.listdir(single_sql_dir):
        db_sql_dir = os.path.join(single_sql_dir, database_name)
        if not os.path.isdir(db_sql_dir):
            continue
        for file_name in tqdm(os.listdir(db_sql_dir), desc=f"处理单一数据库 '{database_name}' 的 SQL 文件"):
            if not file_name.endswith(".json"):
                continue
            file_idx = int(re.findall(r'\d+', file_name)[0]) if re.findall(r'\d+', file_name) else "unknown"
            output_file = os.path.join(output_dir, 'new_sql_nl_single', database_name, f'generated_nl_query_{file_idx}.json')
            os.makedirs(os.path.dirname(output_file), exist_ok=True)

            # 检查输出文件是否已经存在
            if os.path.exists(output_file):
                print(f"已存在：{output_file}，跳过")
                continue  # 跳过当前文件，继续下一个

            sql_file_path = os.path.join(db_sql_dir, file_name)
            # print(f"处理 SQL 文件：{sql_file_path}")

            # 加载 SQL 语句
            with open(sql_file_path, 'r', encoding='utf-8') as f:
                try:
                    sql_data = json.load(f)
                    sql_query = sql_data['sql']
                    results = sql_data['results']
                    sql_skeleton = sql_data['sql skeleton']
                    database = sql_data['database']
                    tables = sql_data['tables']
                    
                except json.JSONDecodeError as e:
                    # print(f"无法解析 JSON 文件 {sql_file_path}：{e}")
                    continue
                except KeyError:
                    # print(f"JSON 文件 {sql_file_path} 缺少字段，跳过")
                    continue

            # 定义保存提示的列表
            prompts = []

            # 阶段 1：生成问题描述
            problem_description_prompt = f"""
请阅读以下 SQL 语句，逐步分析并用详细的语言描述该 SQL 语句的查询意图，包括涉及的表、列、条件，以及可能的业务场景。

SQL 语句：
{sql_query}

请按照以下格式输出：
- **Query intent**: briefly explain the purpose of the SQL statement.
- **Detailed description**: explain in detail the tables, columns, conditions, and other information involved in the query.
- **Business scenario**: what business scenarios the queried tables, columns, conditions, etc. can be used to solve.
- **User description**: actual problems or needs that users may encounter in the above business scenarios.
"""
            # 保存提示
            prompts.append({
                'idx': file_idx,
                'sql_query': sql_query,
                'stage': 'problem_description',
                'prompt': problem_description_prompt
            })

            problem_description = generate_text(problem_description_prompt)

            if not problem_description:
                print(f"未能生成问题描述，跳过文件：{sql_file_path}")
                continue

            # 阶段 2：生成自然语言查询
            # 构建示例文本
            example_text = ""
            if examples:
                for example in examples:
                    example_text += f"### 示例\n"
                    example_text += f"- 问题描述：\n{example['problem_description']}\n"
                    example_text += f"- 自然语言查询：\n{example['natural_language_query']}\n\n"

# - 尽量以“市民反映”开头的方式来转述。
            natural_language_query_prompt = f"""
根据以下问题描述，模拟真实用户的提问，生成一段自然语言查询。要求如下：
- 语言风格要符合真实场景中的用户提问，可能包含意图不明、信息冗余等特点。
- 用户意图可能不够明确，用户无法指定具体要查询什么表或列，只描述自己遇到的场景，包含模糊或不确定的表述。
- 用户偏向于详细描述自己遇到的场景，可能包含自己的个人信息或者相关事件的信息。
- 表达方式可以不够专业，可能包含模糊或不确定的表述。
- 有可能融入情感倾向，如抱怨、不满、疑惑等。

严格要求：
- **生成的文本为英文。**
- **用户自然语言中不要出现具体的表名或列名，因为用户不了解底层数据库结构。**

{example_text}
### 问题描述：
{problem_description}

请根据上述问题描述，生成对应的自然语言查询。
"""
            # 保存提示
            prompts.append({
                'idx': file_idx,
                'problem_description': problem_description,
                'stage': 'natural_language_query',
                'prompt': natural_language_query_prompt
            })

            natural_language_query = generate_text(natural_language_query_prompt)

            if not natural_language_query:
                print(f"未能生成自然语言查询，跳过文件：{sql_file_path}")
                continue

#             # 阶段 3：生成表格名
#             table_name_prompt = f"""
# 请分析以下 SQL 语句，提取其中涉及的所有表名，并将它们以逗号分隔列出：

# SQL 语句：
# {sql_query}

# 请返回表名列表，例如：
# 表名：table1, table2, table3
# - 只需要返回表名的list，不要包含其他信息
# - 表名内部可能有其他符号，请正确识别表名
# """
            # # 生成表格名
            # table_name = generate_text(table_name_prompt)

            # if not table_name:
            #     print(f"未能生成表名，跳过文件：{sql_file_path}")
            #     continue

            # 保存结果到单独的文件
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

            # 保存提示到单独的文件
            prompts_dir = os.path.join(output_dir, 'new_sql_nl_single', database_name, 'prompts')
            os.makedirs(prompts_dir, exist_ok=True)
            prompts_file = os.path.join(prompts_dir, f'prompts_single_{file_idx}.json')
            if os.path.exists(prompts_file):
                print(f"已存在：{prompts_file}，跳过")
            else:
                with open(prompts_file, 'w', encoding='utf-8') as f:
                    json.dump(prompts, f, ensure_ascii=False, indent=2)

            print(f"已生成并保存自然语言查询：{output_file}")

if __name__ == '__main__':
    main()
