import json
import os
import re
import random
import sqlparse
import sqlglot
from sqlparse.tokens import Keyword, Name, Literal, DML, Whitespace
from sqlparse.sql import TokenList, Token
from sqlglot import parse_one
from tqdm import tqdm
from typing import List, Dict, Any, Optional, Tuple

class SQLGenerator:
    def __init__(self, config):
        """
        Initialize the SQLGenerator class
        :param config: configuration dictionary containing paths and parameters
        """
        self.config = config
        self.script_dir = os.path.dirname(os.path.abspath(__file__))
        
        # Initialize paths
        self.parsed_data_dir = os.path.join(self.script_dir, '..', '..', 'data', 'parsed_data')
        self.new_structure_dir = os.path.join(self.script_dir, '..', '..', 'data', 'new_sql_structure')
        self.new_skeleton_dir = os.path.join(self.script_dir, '..', '..', 'data', 'new_sql_skeleton')
        self.new_logs_file = os.path.join(self.script_dir, '..', '..', 'data', 'new_sql_skeletons.json')
        self.old_data_file = os.path.join(self.script_dir, '..', '..', 'data', '12345_sql_skeletons.json')

    def is_keyword(self, token):
        """Check if token is a SQL keyword"""
        return token.ttype in Keyword

    def is_identifier(self, token):
        """Check if token is an identifier"""
        return token.ttype in Name

    def is_literal(self, token):
        """Check if token is a literal"""
        return token.ttype in Literal

    def is_whitespace(self, token):
        """Check if token is whitespace"""
        return token.ttype in Whitespace

    def replace_tokens(self, tokens):
        """Replace tokens with placeholders"""
        new_tokens = []
        for token in tokens:
            if token.is_group:
                replaced = self.replace_tokens(token.tokens)
                new_group = token.__class__(replaced)
                new_tokens.append(new_group)
            elif self.is_keyword(token) or token.ttype == DML:
                new_tokens.append(Token(token.ttype, token.value.upper()))
            elif self.is_identifier(token) or self.is_literal(token):
                new_tokens.append(Token(token.ttype, '_'))
            elif self.is_whitespace(token):
                new_tokens.append(Token(token.ttype, ' '))
            else:
                new_tokens.append(Token(token.ttype, token.value))
        return new_tokens

    def extract_sql_framework(self, sql_text):
        """Extract SQL framework from SQL text"""
        parsed = sqlparse.parse(sql_text)
        if not parsed:
            return ''
        statement = parsed[0]
        new_tokens = self.replace_tokens(statement.tokens)
        sql_framework = ''.join(str(token) for token in new_tokens)
        sql_framework = ' '.join(sql_framework.split())
        return sql_framework

    def get_cfg_rules(self, node, rules=None):
        """Generate CFG rules from AST"""
        if rules is None:
            rules = []

        if not node:
            return rules

        rule = node.key
        children = []
        for arg_value in node.args.values():
            if isinstance(arg_value, sqlglot.Expression):
                children.append(arg_value.key)
            elif isinstance(arg_value, list):
                for item in arg_value:
                    if isinstance(item, sqlglot.Expression):
                        children.append(item.key)
        if children:
            rule_str = f"{rule} -> {' '.join(children)}"
            rules.append(rule_str)
            for arg_value in node.args.values():
                if isinstance(arg_value, sqlglot.Expression):
                    self.get_cfg_rules(arg_value, rules)
                elif isinstance(arg_value, list):
                    for item in arg_value:
                        if isinstance(item, sqlglot.Expression):
                            self.get_cfg_rules(item, rules)
        else:
            rule_str = f"{rule} -> terminal"
            rules.append(rule_str)

        return rules

    def ast_to_dict(self, node):
        """Convert AST node to dictionary"""
        if not node:
            return None
        node_dict = {
            'type': node.key,
            'args': {}
        }
        for key, value in node.args.items():
            if isinstance(value, sqlglot.Expression):
                node_dict['args'][key] = self.ast_to_dict(value)
            elif isinstance(value, list):
                node_dict['args'][key] = [self.ast_to_dict(item) if isinstance(item, sqlglot.Expression) else str(item) for item in value]
            else:
                node_dict['args'][key] = str(value)
        return node_dict

    def build_parse_tree_from_cfg_rules(self, cfg_rules):
        """Build parse tree from CFG rules"""
        class Node:
            def __init__(self, symbol):
                self.symbol = symbol
                self.children = []

        index = 0

        def build_node():
            nonlocal index
            if index >= len(cfg_rules):
                return None

            rule = cfg_rules[index]
            index += 1

            parts = rule.split('->')
            lhs = parts[0].strip()
            rhs_symbols = parts[1].strip().split()

            node = Node(lhs)
            for symbol in rhs_symbols:
                if symbol == 'terminal':
                    child = Node(symbol)
                    node.children.append(child)
                else:
                    if index < len(cfg_rules) and cfg_rules[index].startswith(symbol + ' ->'):
                        child = build_node()
                        if child:
                            node.children.append(child)
                        else:
                            return None
                    else:
                        child = Node(symbol)
                        node.children.append(child)
            return node

        root = build_node()
        return root

    def generate_sql_skeleton(self, node):
        """Generate SQL skeleton from parse tree"""
        if not node:
            return ''

        symbol = node.symbol.lower()

        if symbol == 'select_statement':
            select_part = self.generate_sql_skeleton(node.children[0]) if node.children else ''
            return select_part + ';'

        elif symbol == 'select':
            select_clause = 'SELECT ' + (self.generate_sql_skeleton(node.children[0]) if node.children else '_')
            from_clause = ''
            where_clause = ''
            group_by_clause = ''
            having_clause = ''
            order_by_clause = ''
            limit_clause = ''

            for child in node.children[1:]:
                child_sql = self.generate_sql_skeleton(child)
                child_symbol = child.symbol.lower()
                if child_symbol == 'from':
                    from_clause = ' FROM ' + child_sql
                elif child_symbol == 'where':
                    where_clause = ' WHERE ' + child_sql
                elif child_symbol == 'group_by':
                    group_by_clause = ' GROUP BY ' + child_sql
                elif child_symbol == 'having':
                    having_clause = ' HAVING ' + child_sql
                elif child_symbol == 'order_by':
                    order_by_clause = ' ORDER BY ' + child_sql
                elif child_symbol == 'limit':
                    limit_clause = ' LIMIT ' + child_sql
                else:
                    if 'SELECT' in child_sql.upper():
                        child_sql = f'({child_sql})'
                    select_clause += ' ' + child_sql

            return select_clause + from_clause + where_clause + group_by_clause + having_clause + order_by_clause + limit_clause

        elif symbol == 'select_elements':
            elements = [self.generate_sql_skeleton(child) for child in node.children]
            return ', '.join(elements)

        elif symbol == 'column':
            return '_'

        elif symbol == 'from':
            return self.generate_sql_skeleton(node.children[0]) if node.children else '_'

        elif symbol == 'table_reference':
            return '_'

        elif symbol == 'join_clause':
            left_table = self.generate_sql_skeleton(node.children[0]) if len(node.children) > 0 else '_'
            join_type = node.children[1].symbol.upper() if len(node.children) > 1 else 'JOIN'
            right_table = self.generate_sql_skeleton(node.children[2]) if len(node.children) > 2 else '_'
            on_condition = self.generate_sql_skeleton(node.children[3]) if len(node.children) > 3 else '_'
            return f"{left_table} {join_type} {right_table} ON {on_condition}"

        elif symbol == 'where':
            return self.generate_sql_skeleton(node.children[0]) if node.children else '_'

        elif symbol == 'condition':
            return '_'

        elif symbol == 'group_by':
            return self.generate_sql_skeleton(node.children[0]) if node.children else '_'

        elif symbol == 'having':
            return self.generate_sql_skeleton(node.children[0]) if node.children else '_'

        elif symbol == 'order_by':
            return self.generate_sql_skeleton(node.children[0]) if node.children else '_'

        elif symbol == 'limit':
            return '_'

        elif symbol == 'aggregate_function':
            func_name = node.children[0].symbol.upper() if node.children else '_'
            column = self.generate_sql_skeleton(node.children[1]) if len(node.children) > 1 else '_'
            return f"{func_name}({column})"

        elif symbol == 'terminal':
            return '_'

        else:
            result = ' '.join(self.generate_sql_skeleton(child) for child in node.children)
            return result.strip()

    def is_valid_sql_skeleton(self, sql_skeleton):
        """Validate SQL skeleton"""
        select_positions = [m.start() for m in re.finditer(r'\bSELECT\b', sql_skeleton, re.IGNORECASE)]
        if len(select_positions) <= 1:
            return True

        for pos in select_positions[1:]:
            before_select = sql_skeleton[:pos]
            open_parens = before_select.count('(')
            close_parens = before_select.count(')')
            if open_parens <= close_parens:
                return False
        return True

    def sql_query_to_sql_skeleton(self, sql_query):
        """Convert SQL query to SQL skeleton"""
        sql_skeleton = re.sub(r"'[^']*'", '_', sql_query)
        sql_skeleton = re.sub(r'"[^"]*"', '_', sql_skeleton)
        sql_skeleton = re.sub(r'\b\d+\b', '_', sql_skeleton)

        sql_keywords = set(['SELECT', 'FROM', 'WHERE', 'GROUP', 'BY', 'HAVING', 'ORDER', 'LIMIT', 'JOIN', 'ON', 'AS', 'AND', 'OR', 'IN', 'NOT', 'NULL', 'IS', 'DISTINCT', 'COUNT', 'SUM', 'AVG', 'MIN', 'MAX', 'INNER', 'LEFT', 'RIGHT', 'FULL', 'OUTER', 'UNION', 'ALL'])

        def replace_identifier(match):
            word = match.group(0)
            if word.upper() in sql_keywords or word == '*':
                return word
            else:
                return '_'
        sql_skeleton = re.sub(r'\b\w+\b', replace_identifier, sql_skeleton)
        sql_skeleton = re.sub(r'(_\s*)+', '_ ', sql_skeleton)
        sql_skeleton = ' '.join(sql_skeleton.strip().split())
        return sql_skeleton

    def process_sql_file(self, input_file, output_file):
        """Process SQL file and extract frameworks"""
        with open(input_file, 'r', encoding='utf-8') as infile:
            try:
                data_list = json.load(infile)
                print(f"Processing file {input_file}, containing {len(data_list)} records.")
            except json.JSONDecodeError as e:
                print(f"Error reading file {input_file}: {e}")
                return

        processed_data = []
        for data in data_list:
            sql_text = data.get('sql', '')
            if not sql_text:
                continue
            try:
                ast = parse_one(sql_text)
                cfg_rules = self.get_cfg_rules(ast)
                ast_dict = self.ast_to_dict(ast)
                data['ast'] = ast_dict
                data['cfg_rules'] = cfg_rules
                processed_data.append(data)
            except Exception as e:
                print(f"Error parsing SQL: {e}")
                continue

        with open(output_file, 'w', encoding='utf-8') as outfile:
            json.dump(processed_data, outfile, ensure_ascii=False, indent=2)
        print(f"AST and CFG rules generation completed, results saved in {output_file}")

    def generate_sql_structures(self, cfg_rules_list, num_samples):
        """Generate new SQL structures based on CFG rules"""
        unique_rule_sequences = set(tuple(seq) for seq in cfg_rules_list)
        all_rule_sequences = list(unique_rule_sequences)
        generated_structures = []
        generated_structures.extend(all_rule_sequences)

        with tqdm(total=num_samples, desc="Generating SQL structures") as pbar:
            pbar.update(len(generated_structures))

            while len(generated_structures) < num_samples:
                seq = list(random.choice(all_rule_sequences))
                transformed_seq = self.apply_valid_transformation(seq)
                seq_tuple = tuple(transformed_seq)
                if seq_tuple not in unique_rule_sequences:
                    generated_structures.append(transformed_seq)
                    unique_rule_sequences.add(seq_tuple)
                    pbar.update(1)

            if len(generated_structures) < num_samples:
                additional_structures = list(generated_structures)
                while len(generated_structures) < num_samples:
                    generated_structures.extend(additional_structures)
                    pbar.update(len(additional_structures))
                generated_structures = generated_structures[:num_samples]

        return generated_structures

    def apply_valid_transformation(self, seq):
        """Apply valid transformations to CFG rules"""
        new_seq = seq.copy()
        indices = [i for i in range(len(new_seq)) if '->' in new_seq[i]]
        if len(indices) >= 2:
            i1, i2 = random.sample(indices, 2)
            new_seq[i1], new_seq[i2] = new_seq[i2], new_seq[i1]
        return new_seq

    def process_all_databases(self, total_skeletons_per_db=200, log_ratio=0.1):
        """Process all databases and generate SQL skeletons"""
        os.makedirs(self.new_skeleton_dir, exist_ok=True)

        with open(self.new_logs_file, 'r', encoding='utf-8') as f:
            new_sql_logs = json.load(f)

        with open(self.old_data_file, 'r', encoding='utf-8') as f:
            old_data = json.load(f)

        old_sql_queries = [entry.get('sql', '') for entry in old_data if entry.get('sql', '')]
        old_sql_skeletons = [self.sql_query_to_sql_skeleton(sql_query) for sql_query in old_sql_queries]
        old_sql_skeletons = [s for s in old_sql_skeletons if s.lower().startswith('select')]

        log_sql_skeletons = [self.sql_query_to_sql_skeleton(log_entry['sql']) for log_entry in new_sql_logs]
        log_sql_skeletons = [s for s in log_sql_skeletons if s.lower().startswith('select')]

        for file_name in os.listdir(self.new_structure_dir):
            if file_name.endswith('_structure.json'):
                database_name = file_name.replace('_structure.json', '')
                structure_file = os.path.join(self.new_structure_dir, file_name)
                output_file = os.path.join(self.new_skeleton_dir, f"{database_name}_sql_skeleton.json")

                with open(structure_file, 'r', encoding='utf-8') as f:
                    cfg_rules = json.load(f)

                generated_sql_skeletons = []
                for cfg_rule in cfg_rules:
                    parse_tree = self.build_parse_tree_from_cfg_rules(cfg_rule)
                    if parse_tree:
                        sql_skeleton = self.generate_sql_skeleton(parse_tree)
                        if sql_skeleton and sql_skeleton.lower().startswith('select') and self.is_valid_sql_skeleton(sql_skeleton):
                            generated_sql_skeletons.append(sql_skeleton)

                generated_sql_skeletons = list(set(generated_sql_skeletons))
                combined_sql_skeletons = generated_sql_skeletons + [s for s in old_sql_skeletons]
                combined_sql_skeletons = list(set(combined_sql_skeletons))
                combined_sql_skeletons = [s for s in combined_sql_skeletons if self.is_valid_sql_skeleton(s) and s.lower().startswith('select')]

                num_combined = len(combined_sql_skeletons)
                num_logs_to_include = total_skeletons_per_db - num_combined
                if num_logs_to_include > 0:
                    log_sql_skeletons_unique = list(set(log_sql_skeletons))
                    random.seed(42)
                    if num_logs_to_include > len(log_sql_skeletons_unique):
                        selected_log_skeletons = random.choices(log_sql_skeletons_unique, k=num_logs_to_include)
                    else:
                        selected_log_skeletons = random.sample(log_sql_skeletons_unique, k=num_logs_to_include)
                    combined_sql_skeletons.extend(selected_log_skeletons)

                while len(combined_sql_skeletons) < total_skeletons_per_db:
                    combined_sql_skeletons.extend(combined_sql_skeletons)
                combined_sql_skeletons = combined_sql_skeletons[:total_skeletons_per_db]

                random.shuffle(combined_sql_skeletons)

                with open(output_file, 'w', encoding='utf-8') as f:
                    json.dump(combined_sql_skeletons, f, ensure_ascii=False, indent=2)

                print(f"Database '{database_name}' generated {len(combined_sql_skeletons)} valid SQL skeletons, saved in {output_file}")

    def run(self):
        """Run the main program"""
        # Process all databases
        self.process_all_databases()

if __name__ == '__main__':
    config = {
        'parsed_data_dir': 'path_to_parsed_data',
        'new_structure_dir': 'path_to_new_structure',
        'new_skeleton_dir': 'path_to_new_skeleton',
        'new_logs_file': 'path_to_new_logs',
        'old_data_file': 'path_to_old_data'
    }
    
    generator = SQLGenerator(config)
    generator.run() 