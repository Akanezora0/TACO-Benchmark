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
        Initialize the SQLNLGenerator class
        :param config: configuration dictionary containing API key and model name
        """
        self.config = config
        self.script_dir = os.path.dirname(os.path.abspath(__file__))
        
        # Initialize the language model
        self.model = AutoModelForCausalLM.from_pretrained(
            config['model_name'],
            torch_dtype="auto",
            device_map={"": "cuda:0"}
        )
        self.tokenizer = AutoTokenizer.from_pretrained(config['model_name'])
        
        # Initialize OpenAI client
        self.client = OpenAI(
            base_url=config['base_url'],
            api_key=config['api_key']
        )
        
        # Example data for natural language generation
        self.examples = [
            {
                'problem_description': 'Query potential hazards in Shidi Village, Han Shiqiao, Yang Town, Shunyi District, including possible health and safety issues.',
                'natural_language_query': 'A citizen reported that at the bridgehead of Shidi Village, Han Shiqiao, Yang Town, Shunyi District, there are vendors selling fireworks and firecrackers for children, which poses safety hazards. They hope for prompt verification and handling, calling to report the illegal sale of fireworks and firecrackers.'
            },
            {
                'problem_description': 'Search for corresponding records in marriage registration information based on the names of both parents.',
                'natural_language_query': 'A citizen reported that when handling birth registration for their newborn, the system showed that the marriage certificate information of parents Xiao Ming and Xiao Hong did not match. Their marriage certificate was registered on March 25, 2022, at the Haidian District Civil Affairs Bureau. They hope to verify and resolve this situation, calling to report the mismatch in marriage certificate information.'
            },
            {
                'problem_description': 'Check whether this enterprise has any issues regarding tax payment.',
                'natural_language_query': 'A citizen reported purchasing Coca-Cola from Beijing Jinhui Fa Trading Co., Ltd. (address: Shunyi District Shunxin Shimen Agricultural Wholesale Market) through Pinduoduo, but the company has been delaying issuing an invoice with various excuses. Order number: 240123149364469331471. Calling to report Jinhui Fa\'s refusal to issue an invoice.'
            }
        ]

    def load_schema(self, schema_file):
        """Load the database schema"""
        with open(schema_file, 'r', encoding='utf-8') as f:
            return json.load(f)

    def extract_schema_info(self, schema):
        """Extract schema information"""
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
        """Build the SQL schema graph"""
        G = nx.Graph()

        # Add schema nodes and edges
        for table in schema_info['tables']:
            table_name = table['table_name']
            G.add_node(table_name, node_type='table', label=table_name)

            for column in table['columns']:
                column_name = column['column_name']
                full_column_name = f"{table_name}.{column_name}"
                G.add_node(full_column_name, node_type='column', label=full_column_name)
                G.add_edge(table_name, full_column_name, edge_type='table-column')

        # Add foreign key edges
        for fk in schema_info.get('foreign_keys', []):
            source_node = f"{fk['table']}.{fk['column']}"
            target_node = f"{fk['references']['table']}.{fk['references']['column']}"
            if source_node in G.nodes and target_node in G.nodes:
                G.add_edge(source_node, target_node, edge_type='foreign-key')

        # Parse SQL framework and add placeholder nodes
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
        """Parse SQL framework and identify placeholders"""
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
        """Get possible schema nodes based on placeholder type"""
        if placeholder_type == 'SELECT':
            return [n for n, attr in G.nodes(data=True) if attr['node_type'] == 'column']
        elif placeholder_type == 'FROM':
            return [n for n, attr in G.nodes(data=True) if attr['node_type'] == 'table']
        elif placeholder_type == 'WHERE':
            return [n for n, attr in G.nodes(data=True) if attr['node_type'] == 'column']
        else:
            return [n for n, attr in G.nodes(data=True) if attr['node_type'] in ('table', 'column')]

    def select_random_tables(self, schema_info, num_tables=2):
        """Select random tables from schema"""
        selected_tables = random.sample(schema_info['tables'], min(num_tables, len(schema_info['tables'])))
        selected_columns = {}
        for table in selected_tables:
            selected_columns[table] = schema_info['columns'][table]
        return selected_tables, selected_columns

    def construct_prompt(self, sql_framework, selected_tables, selected_columns, cross_database=False):
        """Construct prompt for SQL generation"""
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
Please fill in the placeholder "_" based on the following SQL framework and available table names and column names, to generate a complete and correct SQL statement that can be executed on SQLite.

Strict requirements:
- **Only output the final generated complete SQL statement, do not repeat the prompt content.**
- **The generated SQL must be grammatically correct and can be executed directly on SQLite to get results.**
- **Do not add any additional explanations, comments, or output formats (code blocks, spaces, etc.).**
- **The table names, column names, WHERE conditions, etc. must be in the given tables and columns.**
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
            prompt = f"""
Please fill in the placeholder "_" based on the following SQL framework and available table names and column names, to generate a complete and correct SQL statement that can be executed on SQLite.

Strict requirements:
- **Only output the final generated complete SQL statement, do not repeat the prompt content.**
- **The generated SQL must be grammatically correct and can be executed directly on SQLite to get results.**
- **Do not add any additional explanations, comments, or output formats (code blocks, spaces, etc.).**
- **The table names, column names, WHERE conditions, etc. must be in the given tables and columns.**
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

    def generate_text(self, user_input):
        """Generate text using the language model"""
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
        """Validate SQL statement syntax"""
        try:
            parsed = sqlparse.parse(sql_statement)
            return parsed and len(parsed) > 0
        except Exception:
            return False

    def execute_single_db_sql(self, sql, db_path):
        """Execute SQL statement on a single database"""
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
        """Execute SQL statement across multiple databases"""
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
        """Generate natural language query from SQL"""
        # Generate problem description
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
        problem_description = self.generate_text(problem_description_prompt)
        if not problem_description:
            return None

        # Generate natural language query
        example_text = ""
        for example in self.examples:
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
        """Process a single database"""
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
        """Run the main program"""
        # Process single database
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