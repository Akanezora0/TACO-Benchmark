import json
import os
import re
import random
import sqlite3
import sqlparse
import networkx as nx
from tqdm import tqdm
import torch
from openai import OpenAI
from transformers import AutoModelForCausalLM, AutoTokenizer
import time
import pandas as pd
from typing import List, Dict, Any, Optional, Tuple

class SQLNLGenerator:
    def __init__(self, config):
        """
        initialize the SQLNLGenerator class
        :param config: configuration dictionary, containing API key and model name
        """
        self.config = config
        self.script_dir = os.path.dirname(os.path.abspath(__file__))
        

        self.model = AutoModelForCausalLM.from_pretrained(
            config['model_name'],
            torch_dtype="auto",
            device_map={"": "cuda:0"}
        )
        self.tokenizer = AutoTokenizer.from_pretrained(config['model_name'])
        

        self.client = OpenAI(
            base_url=config['base_url'],
            api_key=config['api_key']
        )
        
        # example data
        self.examples = [
            {
                'problem_description': '查询顺义区杨镇汉石桥事地村的隐患名称，包括可能存在的卫生、安全等问题。',
                'natural_language_query': '市民反映，顺义区杨镇汉石桥事地村，在桥头有售卖小孩玩的烟花爆竹，存在安全隐患，希望尽快进行核实处理，来电反映违规售卖烟花爆竹问题。'
            },
            {
                'problem_description': '根据父母双方的姓名，在婚姻登记信息中查找有无存在对应记录。',
                'natural_language_query': '市民反映，新生儿办理出生一件事的时候，显示父母双方小明和小红结婚证信息不匹配，自己的结婚证是22年3月25号在海淀区民政局登记的，希望能帮助核实处理一下是什么情况，来电反映结婚证信息不匹配问题。'
            },
            {
                'problem_description': '查询这家企业在纳税方面有没有问题。',
                'natural_language_query': '市民反映，通过拼多多购买北京金汇发商贸有限公司的可乐，地址：顺义区顺鑫石门农产品批发市场，但是购买后一直不给开发票，找各种理由推拖，订单号：240123149364469331471，来电反映金汇发不给开发票问题。'
            }
        ]

    def load_schema(self, schema_file):
        """load the database schema"""
        with open(schema_file, 'r', encoding='utf-8') as f:
            return json.load(f)

    def extract_schema_info(self, schema):
        """extract the schema information"""
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

    def build_sql_schema_graph(self, schema_info, sql_framework):
        """build the SQL schema graph"""
        G = nx.Graph()

        # add the schema nodes and edges
        for table in schema_info['tables']:
            table_name = table['table_name']
            G.add_node(table_name, node_type='table', label=table_name)

            for column in table['columns']:
                column_name = column['column_name']
                full_column_name = f"{table_name}.{column_name}"
                G.add_node(full_column_name, node_type='column', label=full_column_name)
                G.add_edge(table_name, full_column_name, edge_type='table-column')

        # foreign key edges
        for fk in schema_info.get('foreign_keys', []):
            source_node = f"{fk['table']}.{fk['column']}"
            target_node = f"{fk['references']['table']}.{fk['references']['column']}"
            if source_node in G.nodes and target_node in G.nodes:
                G.add_edge(source_node, target_node, edge_type='foreign-key')

        # parse the SQL framework, add the SQL placeholder nodes
        sql_placeholders = self.parse_sql_framework(sql_framework)
        for idx, placeholder in enumerate(sql_placeholders):
            sql_node = f"sql_{idx}"
            G.add_node(sql_node, 
                      node_type='sql_placeholder',
                      placeholder_type=placeholder.get('clause', 'UNKNOWN'),
                      label=placeholder.get('text', '_'))
            
            possible_nodes = self.get_possible_schema_nodes(G, placeholder.get('clause', 'UNKNOWN'))
            for node in possible_nodes:
                G.add_edge(sql_node, node, edge_type='sql-schema')

        return G

    def parse_sql_framework(self, sql_framework):
        """parse the SQL framework, add the SQL placeholder nodes"""
        placeholders = []
        clauses = ['SELECT', 'FROM', 'WHERE', 'GROUP BY', 'HAVING', 'ORDER BY', 'LIMIT', 'JOIN', 'ON']
        pattern = re.compile(r'\b(' + '|'.join(clauses) + r')\b', re.IGNORECASE)
        tokens = pattern.split(sql_framework)
        current_clause = 'UNKNOWN'
        
        for token in tokens:
            token = token.strip()
            if not token:
                continue
            if token.upper() in clauses:
                current_clause = token.upper()
            else:
                placeholder_matches = re.findall(r'(_+)', token)
                for placeholder in placeholder_matches:
                    placeholders.append({'clause': current_clause, 'text': placeholder})
        return placeholders

    def get_possible_schema_nodes(self, G, placeholder_type):
        """get the possible schema nodes"""
        if placeholder_type == 'SELECT':
            return [n for n, attr in G.nodes(data=True) if attr['node_type'] == 'column']
        elif placeholder_type == 'FROM':
            return [n for n, attr in G.nodes(data=True) if attr['node_type'] == 'table']
        elif placeholder_type == 'WHERE':
            return [n for n, attr in G.nodes(data=True) if attr['node_type'] == 'column']
        else:
            return [n for n, attr in G.nodes(data=True) if attr['node_type'] in ('table', 'column')]

    def select_random_tables(self, schema_info, num_tables=2):
        """select the random tables"""
        selected_tables = random.sample(schema_info['tables'], min(num_tables, len(schema_info['tables'])))
        selected_columns = {}
        for table in selected_tables:
            selected_columns[table] = schema_info['columns'][table]
        return selected_tables, selected_columns

    def construct_prompt(self, sql_framework, selected_tables, selected_columns, cross_database=False):
        """construct the prompt"""
        def quote_identifier(identifier):
            return f'"{identifier}"'

        tables = ', '.join([quote_identifier(table) for table in selected_tables])
        columns = []
        for table in selected_tables:
            columns.extend([quote_identifier(col) for col in selected_columns[table]])
        columns_str = ', '.join(columns)

        if cross_database:
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

    def generate_text(self, user_input):
        """generate the text"""
        try:
            messages = [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": user_input}
            ]
            text = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )

            model_inputs = self.tokenizer(
                [text],
                return_tensors="pt",
                max_length=4096,
                truncation=True
            ).to(self.model.device)

            with torch.cuda.amp.autocast():
                generated_ids = self.model.generate(
                    **model_inputs,
                    max_new_tokens=512,
                    do_sample=True,
                    temperature=0.6,
                    top_p=0.9
                )

            generated_ids = [
                output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
            ]
            response = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]

            torch.cuda.empty_cache()
            return response.strip()

        except Exception as e:
            print(f"API call failed: {e}")
            return ""

    def is_valid_sql(self, sql_statement):
        """validate the validity of the SQL statement"""
        try:
            parsed = sqlparse.parse(sql_statement)
            return parsed and len(parsed) > 0
        except Exception:
            return False

    def execute_single_db_sql(self, sql, db_path):
        """execute the single database SQL statement"""
        try:
            conn = sqlite3.connect(db_path)
            cursor = conn.cursor()
            cursor.execute(sql)
            results = cursor.fetchall()
            conn.close()
            return (results, True) if results else (0, False)
        except sqlite3.Error as e:
            print(f"SQLite error: {e}")
            return 0, False

    def execute_cross_db_sql(self, sql, primary_db_path, secondary_db_path, secondary_alias):
        """execute the cross-database SQL statement"""
        try:
            conn = sqlite3.connect(primary_db_path)
            cursor = conn.cursor()
            cursor.execute(f'ATTACH DATABASE "{secondary_db_path}" AS "{secondary_alias}";')
            cursor.execute(sql)
            results = cursor.fetchall()
            cursor.execute(f'DETACH DATABASE "{secondary_alias}";')
            conn.close()
            return (results, True) if results else (0, False)
        except sqlite3.Error as e:
            print(f"SQLite error: {e}")
            return 0, False

    def generate_nl_query(self, sql_query, results, sql_skeleton, database, tables):
        """generate the natural language query"""
        # generate the problem description
        problem_description_prompt = f"""
请阅读以下 SQL 语句，逐步分析并用详细的语言描述该 SQL 语句的查询意图，包括涉及的表、列、条件，以及可能的业务场景。

SQL 语句：
{sql_query}

请按照以下格式输出：
- **查询意图**：简要说明 SQL 语句的目的。
- **详细描述**：详细解释查询涉及的表、列、条件等信息。
- **业务场景**：查询到的表、列、条件等可以用来解决什么业务场景。
- **用户描述**：上面的业务场景下用户可能遇到的实际问题或需求。
"""
        problem_description = self.generate_text(problem_description_prompt)
        if not problem_description:
            return None

        # 生成自然语言查询
        example_text = ""
        for example in self.examples:
            example_text += f"### 示例\n"
            example_text += f"- 问题描述：\n{example['problem_description']}\n"
            example_text += f"- 自然语言查询：\n{example['natural_language_query']}\n\n"

        natural_language_query_prompt = f"""
根据以下问题描述，模拟真实用户的提问，生成一段自然语言查询。要求如下：
- 语言风格要符合真实场景中的用户提问，可能包含意图不明、信息冗余等特点。
- 用户意图可能不够明确，用户无法指定具体要查询什么表或列，只描述自己遇到的场景，包含模糊或不确定的表述。
- 用户偏向于详细描述自己遇到的场景，可能包含自己的个人信息或者相关事件的信息。
- 表达方式可以不够专业，可能包含模糊或不确定的表述。
- 有可能融入情感倾向，如抱怨、不满、疑惑等。

严格要求：
- **用户自然语言中不要出现具体的表名或列名，因为用户不了解底层数据库结构。**

{example_text}
### 问题描述：
{problem_description}

请根据上述问题描述，生成对应的自然语言查询。
"""
        natural_language_query = self.generate_text(natural_language_query_prompt)
        if not natural_language_query:
            return None

        return {
            'sql': sql_query,
            'problem_description': problem_description,
            'natural_language_query': natural_language_query,
            'results': results,
            'sql skeleton': sql_skeleton,
            'database': database,
            'tables': tables
        }

    def process_database(self, database_name, sql_skeletons, schema_info, output_dir, is_cross_database=False):
        """process the database"""
        MAX_RETRIES = 3
        for idx, sql_skeleton in enumerate(tqdm(sql_skeletons, desc=f"processing {database_name}")):
            num_placeholders = sql_skeleton.count('_')
            required_tables = min(2, num_placeholders) if num_placeholders > 1 else 1

            if required_tables == 1:
                selected_tables, selected_columns = self.select_random_tables(schema_info, num_tables=1)
            else:
                selected_tables, selected_columns = self.select_random_tables(schema_info, num_tables=2)

            prompt = self.construct_prompt(sql_skeleton, selected_tables, selected_columns, is_cross_database)
            
            output_file = os.path.join(output_dir, database_name, f'generated_sql_statement_{idx}.json')
            if os.path.exists(output_file):
                continue

            for attempt in range(MAX_RETRIES):
                sql_statement = self.generate_text(prompt)
                if not sql_statement:
                    continue

                if self.is_valid_sql(sql_statement) and sql_statement.lower().startswith('select'):
                    db_path = os.path.join(self.script_dir, '..', '..', 'data', 'database', database_name, f"{database_name}.db")
                    if not os.path.exists(db_path):
                        break

                    results, success = self.execute_single_db_sql(sql_statement, db_path)
                    if success:
                        result = self.generate_nl_query(sql_statement, results, sql_skeleton, database_name, selected_columns)
                        if result:
                            os.makedirs(os.path.dirname(output_file), exist_ok=True)
                            with open(output_file, 'w', encoding='utf-8') as f:
                                json.dump(result, f, ensure_ascii=False, indent=2)
                        break

    def run(self):
        """run the main program"""
        # process the single database
        single_sql_dir = os.path.join(self.script_dir, '..', '..', 'data', 'new_sql_single')
        for database_name in os.listdir(single_sql_dir):
            db_sql_dir = os.path.join(single_sql_dir, database_name)
            if not os.path.isdir(db_sql_dir):
                continue

            schema_file = os.path.join(self.script_dir, '..', '..', 'data', 'new_schema', f"{database_name}_schema.json")
            if not os.path.exists(schema_file):
                continue

            schema = self.load_schema(schema_file)
            schema_info = self.extract_schema_info(schema)

            with open(os.path.join(db_sql_dir, f"{database_name}_sql_skeleton.json"), 'r', encoding='utf-8') as f:
                sql_skeletons = json.load(f)

            self.process_database(database_name, sql_skeletons, schema_info, single_sql_dir)

if __name__ == '__main__':
    config = {
        'model_name': "your_model_name",
        'base_url': "your_base_url",
        'api_key': "your_api_key"
    }
    
    generator = SQLNLGenerator(config)
    generator.run() 