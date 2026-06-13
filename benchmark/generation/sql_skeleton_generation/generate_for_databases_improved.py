#!/usr/bin/env python3
"""
Generate CFG structures and SQL skeletons for each database in the beijing dataset (improved version)

Key improvements:
1. Combine old database CFG rules and new database expert examples to generate different results per database
2. Increase randomness and diversity (rotation, pruning, merging, and other transformation strategies)
3. Improved SQL skeleton generation supporting JOIN, subqueries, aggregates, and other complex structures
"""

import os
import sys
import json
import sqlglot
from sqlglot import parse_one
import random
import re
from tqdm import tqdm
from collections import defaultdict

# Node class definition
class Node:
    def __init__(self, symbol):
        self.symbol = symbol
        self.children = []

def get_cfg_rules(node, rules=None):
    """Recursively traverse AST and generate CFG rule sequence"""
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
                get_cfg_rules(arg_value, rules)
            elif isinstance(arg_value, list):
                for item in arg_value:
                    if isinstance(item, sqlglot.Expression):
                        get_cfg_rules(item, rules)
    else:
        rule_str = f"{rule} -> terminal"
        rules.append(rule_str)
    
    return rules

def ast_to_dict(node):
    """Convert AST node to dictionary"""
    if not node:
        return None
    node_dict = {
        'type': node.key,
        'args': {}
    }
    for key, value in node.args.items():
        if isinstance(value, sqlglot.Expression):
            node_dict['args'][key] = ast_to_dict(value)
        elif isinstance(value, list):
            node_dict['args'][key] = [ast_to_dict(item) if isinstance(item, sqlglot.Expression) else str(item) for item in value]
        else:
            node_dict['args'][key] = str(value)
    return node_dict

def generate_cfg_for_database(expert_file, old_cfg_file, output_file, db_name):
    """Generate AST/CFG file for a single database (combining old database CFG rules)"""
    # Load expert examples
    with open(expert_file, 'r', encoding='utf-8') as f:
        expert_data = json.load(f)
    
    # Load old database CFG rules (for augmentation)
    old_cfg_rules_list = []
    if old_cfg_file and os.path.exists(old_cfg_file):
        with open(old_cfg_file, 'r', encoding='utf-8') as f:
            old_cfg_data = json.load(f)
        for data in old_cfg_data:
            cfg_rules = data.get('cfg_rules', [])
            if cfg_rules:
                old_cfg_rules_list.append(cfg_rules)
    
    # Set different random seed per database (based on database name)
    random.seed(hash(db_name) % (2**32))
    
    processed_data = []
    for idx, data in enumerate(expert_data):
        sql_text = data.get('sql_framework', '')
        if not sql_text:
            continue
        try:
            ast = parse_one(sql_text)
            cfg_rules = get_cfg_rules(ast)
            ast_dict = ast_to_dict(ast)
            data['ast'] = ast_dict
            data['cfg_rules'] = cfg_rules
            processed_data.append(data)
        except Exception as e:
            continue
    
    # If old database CFG rules exist, add some augmented CFG rules
    if old_cfg_rules_list:
        # Randomly select some old database CFG rules as supplements
        num_old_to_add = min(10, len(old_cfg_rules_list))
        selected_old_rules = random.sample(old_cfg_rules_list, num_old_to_add)
        for old_cfg_rules in selected_old_rules:
            # Create supplement data item
            supplement_data = {
                'query': '',
                'sql': '',
                'sql_framework': '',
                'ast': None,
                'cfg_rules': old_cfg_rules
            }
            processed_data.append(supplement_data)
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(processed_data, f, ensure_ascii=False, indent=2)
    
    return len(processed_data)

def apply_advanced_transformation(seq, old_cfg_rules_list=None):
    """Apply advanced transformation strategies (rotation, pruning, merging, etc.)"""
    new_seq = seq.copy()
    
    # Strategy 1: swap rule order (30% probability)
    if random.random() < 0.3:
        indices = [i for i in range(len(new_seq)) if '->' in new_seq[i]]
        if len(indices) >= 2:
            i1, i2 = random.sample(indices, 2)
            new_seq[i1], new_seq[i2] = new_seq[i2], new_seq[i1]
    
    # Strategy 2: insert old database CFG rule fragments (20% probability)
    elif random.random() < 0.5 and old_cfg_rules_list:
        if random.random() < 0.2:
            old_rule_seq = random.choice(old_cfg_rules_list)
            # Randomly insert some old rules
            insert_pos = random.randint(0, len(new_seq))
            insert_rules = random.sample(old_rule_seq, min(3, len(old_rule_seq)))
            new_seq = new_seq[:insert_pos] + insert_rules + new_seq[insert_pos:]
    
    # Strategy 3: duplicate and modify rules (30% probability)
    elif random.random() < 0.8:
        indices = [i for i in range(len(new_seq)) if '->' in new_seq[i]]
        if indices:
            idx = random.choice(indices)
            rule = new_seq[idx]
            # Duplicate rule and possibly modify
            new_seq.insert(idx + 1, rule)
    
    # Strategy 4: delete rules (10% probability, but keep basic structure)
    elif random.random() < 0.9 and len(new_seq) > 5:
        indices = [i for i in range(len(new_seq)) if '->' in new_seq[i]]
        if len(indices) > 3:
            idx_to_remove = random.choice(indices[1:-1])  # Do not delete first and last
            new_seq.pop(idx_to_remove)
    
    return new_seq

def generate_structures_for_database(cfg_file, old_cfg_file, output_file, db_name, num_samples=100):
    """Generate SQL structures for a single database (combining old database CFG rules)"""
    with open(cfg_file, 'r', encoding='utf-8') as f:
        cfg_data = json.load(f)
    
    # Load old database CFG rules
    old_cfg_rules_list = []
    if old_cfg_file and os.path.exists(old_cfg_file):
        with open(old_cfg_file, 'r', encoding='utf-8') as f:
            old_cfg_data = json.load(f)
        for data in old_cfg_data:
            cfg_rules = data.get('cfg_rules', [])
            if cfg_rules:
                old_cfg_rules_list.append(cfg_rules)
    
    cfg_rules_list = []
    for data in cfg_data:
        cfg_rules = data.get('cfg_rules', [])
        if cfg_rules:
            cfg_rules_list.append(cfg_rules)
    
    if not cfg_rules_list:
        return []
    
    # Set different random seed per database
    random.seed(hash(db_name) % (2**32))
    
    # Collect all unique CFG rule sequences
    unique_rule_sequences = set(tuple(seq) for seq in cfg_rules_list)
    all_rule_sequences = list(unique_rule_sequences)
    
    # Initialize generated structure list
    generated_structures = []
    generated_structures.extend(all_rule_sequences)
    
    # Generate new structures (using advanced transformation strategies)
    max_attempts = num_samples * 30
    attempts = 0
    
    while len(generated_structures) < num_samples and attempts < max_attempts:
        attempts += 1
        seq = list(random.choice(all_rule_sequences))
        transformed_seq = apply_advanced_transformation(seq, old_cfg_rules_list)
        
        seq_tuple = tuple(transformed_seq)
        if seq_tuple not in unique_rule_sequences:
            generated_structures.append(transformed_seq)
            unique_rule_sequences.add(seq_tuple)
    
    # If still insufficient, repeat proportionally
    if len(generated_structures) < num_samples:
        additional = list(generated_structures)
        while len(generated_structures) < num_samples:
            generated_structures.extend(additional)
        generated_structures = generated_structures[:num_samples]
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(generated_structures, f, ensure_ascii=False, indent=2)
    
    return generated_structures

def build_parse_tree_from_cfg_rules(cfg_rules):
    """Build parse tree from CFG rules"""
    index = 0

    def build_node():
        nonlocal index
        if index >= len(cfg_rules):
            return None

        rule = cfg_rules[index]
        index += 1

        parts = rule.split('->')
        if len(parts) != 2:
            return None
        
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
                    child = Node(symbol)
                    node.children.append(child)
        return node

    return build_node()

def generate_sql_skeleton(node):
    """Generate SQL skeleton from parse tree (improved version, supports more structures, based on original code)"""
    if not node:
        return ''

    symbol = node.symbol.lower()

    if symbol == 'select_statement':
        select_part = generate_sql_skeleton(node.children[0]) if node.children else ''
        return select_part + ';'

    elif symbol == 'select':
        select_clause = 'SELECT ' + (generate_sql_skeleton(node.children[0]) if node.children else '_')
        from_clause = ''
        where_clause = ''
        group_by_clause = ''
        having_clause = ''
        order_by_clause = ''
        limit_clause = ''

        for child in node.children[1:]:
            child_sql = generate_sql_skeleton(child)
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
                # If nested SELECT appears, add parentheses
                if 'SELECT' in child_sql.upper():
                    child_sql = f'({child_sql})'
                select_clause += ' ' + child_sql

        return select_clause + from_clause + where_clause + group_by_clause + having_clause + order_by_clause + limit_clause

    elif symbol == 'select_elements':
        elements = [generate_sql_skeleton(child) for child in node.children]
        return ', '.join(elements) if elements else '_'

    elif symbol == 'column':
        return '_'

    elif symbol == 'from':
        return generate_sql_skeleton(node.children[0]) if node.children else '_'

    elif symbol == 'table_reference':
        return '_'

    elif symbol == 'join_clause':
        left_table = generate_sql_skeleton(node.children[0]) if len(node.children) > 0 else '_'
        join_type = node.children[1].symbol.upper() if len(node.children) > 1 else 'JOIN'
        right_table = generate_sql_skeleton(node.children[2]) if len(node.children) > 2 else '_'
        on_condition = generate_sql_skeleton(node.children[3]) if len(node.children) > 3 else '_'
        return f"{left_table} {join_type} {right_table} ON {on_condition}"

    elif symbol == 'where':
        return generate_sql_skeleton(node.children[0]) if node.children else '_'

    elif symbol == 'condition':
        return '_'

    elif symbol == 'group_by':
        return generate_sql_skeleton(node.children[0]) if node.children else '_'

    elif symbol == 'having':
        return generate_sql_skeleton(node.children[0]) if node.children else '_'

    elif symbol == 'order_by':
        return generate_sql_skeleton(node.children[0]) if node.children else '_'

    elif symbol == 'limit':
        return '_'

    elif symbol == 'aggregate_function':
        func_name = node.children[0].symbol.upper() if node.children else '_'
        column = generate_sql_skeleton(node.children[1]) if len(node.children) > 1 else '_'
        return f"{func_name}({column})"

    elif symbol == 'terminal':
        return '_'

    else:
        # Recursively process other symbols
        result = ' '.join(generate_sql_skeleton(child) for child in node.children)
        return result.strip()

def cfg_rules_to_sql_skeleton(cfg_rules):
    """Convert CFG rule sequence back to SQL skeleton"""
    parse_tree = build_parse_tree_from_cfg_rules(cfg_rules)
    if parse_tree:
        return generate_sql_skeleton(parse_tree)
    return ''

def is_valid_sql_skeleton(sql_skeleton):
    """Check if SQL skeleton is valid (improved version, fixes syntax errors)"""
    if not sql_skeleton or not sql_skeleton.upper().startswith('SELECT'):
        return False
    
    # Check basic syntax errors
    # 1. FROM must be followed by table name (cannot be empty)
    if 'FROM' in sql_skeleton.upper():
        from_pos = sql_skeleton.upper().find('FROM')
        after_from = sql_skeleton[from_pos+4:].strip()
        # Invalid if after FROM is WHERE, HAVING, GROUP BY, ORDER BY, LIMIT, or empty
        if not after_from or any(after_from.upper().startswith(kw) for kw in ['WHERE', 'HAVING', 'GROUP', 'ORDER', 'LIMIT', ';']):
            return False
    
    # 2. WHERE must have FROM and table name before it, and condition after it
    if 'WHERE' in sql_skeleton.upper():
        where_pos = sql_skeleton.upper().find('WHERE')
        before_where = sql_skeleton[:where_pos].strip()
        if 'FROM' not in before_where.upper():
            return False
        from_pos = before_where.upper().rfind('FROM')
        after_from = before_where[from_pos+4:].strip()
        if not after_from or after_from == '':
            return False
        # WHERE must have content after it
        after_where = sql_skeleton[where_pos+5:].strip()
        if not after_where or after_where in [';', '']:
            return False
    
    # 3. HAVING must have GROUP BY before it
    if 'HAVING' in sql_skeleton.upper():
        having_pos = sql_skeleton.upper().find('HAVING')
        before_having = sql_skeleton[:having_pos].upper()
        if 'GROUP BY' not in before_having:
            return False
    
    # 4. Check if subquery is inside parentheses
    select_positions = [m.start() for m in re.finditer(r'\bSELECT\b', sql_skeleton, re.IGNORECASE)]
    if len(select_positions) <= 1:
        return True
    
    for pos in select_positions[1:]:
        before_select = sql_skeleton[:pos]
        if before_select.count('(') <= before_select.count(')'):
            return False
    return True

def sql_query_to_sql_skeleton(sql_query):
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

def classify_skeleton_difficulty(skeleton):
    """Classify SQL skeleton difficulty
    
    Returns:
        'simple': single table, no JOIN, no subquery, no GROUP BY
        'medium': has JOIN (at most 1) or GROUP BY, but no subquery, UNION, or HAVING
        'complex': has subquery, UNION, HAVING, or multi-table JOIN (2 or more)
    """
    skeleton_upper = skeleton.upper()
    
    has_join = 'JOIN' in skeleton_upper
    has_subquery = '(' in skeleton and 'SELECT' in skeleton_upper and skeleton.count('SELECT') > 1
    has_group_by = 'GROUP BY' in skeleton_upper
    has_having = 'HAVING' in skeleton_upper
    has_union = 'UNION' in skeleton_upper
    join_count = skeleton_upper.count('JOIN')
    
    # Simple: single table, no JOIN, no subquery, no GROUP BY
    if not has_join and not has_subquery and not has_group_by:
        return 'simple'
    
    # Complex: has subquery, UNION, HAVING, or multi-table JOIN (2 or more)
    if has_subquery or has_union or has_having or join_count >= 2:
        return 'complex'
    
    # Medium: other cases (JOIN at most 1, or GROUP BY, but no subquery/UNION/HAVING)
    return 'medium'

def generate_skeletons_for_database(structure_file, output_file, old_data_file=None, new_logs_file=None, total_skeletons=200, db_name=None, simple_ratio=0.5, medium_ratio=0.35, complex_ratio=0.15):
    """Generate SQL skeletons for a single database (based on original code logic)"""
    with open(structure_file, 'r', encoding='utf-8') as f:
        structures = json.load(f)
    
    # Set different random seed per database
    if db_name:
        random.seed(hash(db_name) % (2**32))
    
    # Generate SQL skeletons from CFG rules
    generated_skeletons = []
    for cfg_rules in structures:
        sql_skeleton = cfg_rules_to_sql_skeleton(cfg_rules)
        # Ensure SQL skeleton starts with SELECT and is valid
        if sql_skeleton and sql_skeleton.lower().startswith('select') and is_valid_sql_skeleton(sql_skeleton):
            generated_skeletons.append(sql_skeleton)
    
    # Remove duplicate skeletons
    generated_skeletons = list(set(generated_skeletons))
    
    # Load old database data (based on original code)
    old_sql_skeletons = []
    if old_data_file and os.path.exists(old_data_file):
        with open(old_data_file, 'r', encoding='utf-8') as f:
            old_data = json.load(f)
        old_sql_queries = [entry.get('sql', '') for entry in old_data if entry.get('sql', '')]
        old_sql_skeletons = [sql_query_to_sql_skeleton(sql_query) for sql_query in old_sql_queries]
        old_sql_skeletons = [s for s in old_sql_skeletons if s.lower().startswith('select')]
    
    # Load new log file (based on original code)
    log_sql_skeletons = []
    if new_logs_file and os.path.exists(new_logs_file):
        with open(new_logs_file, 'r', encoding='utf-8') as f:
            new_logs = json.load(f)
        log_sql_skeletons = [sql_query_to_sql_skeleton(log_entry.get('sql', '')) for log_entry in new_logs if log_entry.get('sql', '')]
        log_sql_skeletons = [s for s in log_sql_skeletons if s.lower().startswith('select')]
    
    # Ensure old database skeletons are in final result and start with SELECT
    combined_sql_skeletons = generated_skeletons + [s for s in old_sql_skeletons if s.lower().startswith('select')]
    combined_sql_skeletons = list(set(combined_sql_skeletons))
    
    # Filter out invalid skeletons
    combined_sql_skeletons = [s for s in combined_sql_skeletons if is_valid_sql_skeleton(s) and s.lower().startswith('select')]
    
    # Calculate skeleton count to include from logs (based on original code)
    num_combined = len(combined_sql_skeletons)
    num_logs_to_include = total_skeletons - num_combined
    if num_logs_to_include > 0 and log_sql_skeletons:
        # Randomly select log skeletons
        log_sql_skeletons = [s for s in log_sql_skeletons if s.lower().startswith('select')]
        log_sql_skeletons = list(set(log_sql_skeletons))
        if len(log_sql_skeletons) > 0:
            if num_logs_to_include > len(log_sql_skeletons):
                # If needed count exceeds available log skeletons, use repeated selection
                selected_log_skeletons = random.choices(log_sql_skeletons, k=num_logs_to_include)
            else:
                selected_log_skeletons = random.sample(log_sql_skeletons, k=num_logs_to_include)
            combined_sql_skeletons.extend(selected_log_skeletons)
    
    # If total still insufficient, supplement to enough count first
    while len(combined_sql_skeletons) < total_skeletons:
        combined_sql_skeletons.extend(combined_sql_skeletons)
    combined_sql_skeletons = combined_sql_skeletons[:total_skeletons * 2]  # Generate extra for subsequent filtering
    
    # Classify all skeletons by difficulty
    skeletons_by_difficulty = {'simple': [], 'medium': [], 'complex': []}
    for skeleton in combined_sql_skeletons:
        difficulty = classify_skeleton_difficulty(skeleton)
        skeletons_by_difficulty[difficulty].append(skeleton)
    
    # Calculate target counts
    target_counts = {
        'simple': int(total_skeletons * simple_ratio),
        'medium': int(total_skeletons * medium_ratio),
        'complex': int(total_skeletons * complex_ratio)
    }
    
    # Ensure total is correct (handle rounding error)
    actual_total = sum(target_counts.values())
    if actual_total < total_skeletons:
        target_counts['simple'] += (total_skeletons - actual_total)
    
    # Select skeletons by target ratio
    final_skeletons = []
    for difficulty in ['simple', 'medium', 'complex']:
        available = skeletons_by_difficulty[difficulty]
        target = target_counts[difficulty]
        
        if len(available) >= target:
            # Randomly select target count
            selected = random.sample(available, target)
        elif len(available) > 0:
            # If insufficient, use all and repeat
            selected = available.copy()
            while len(selected) < target and len(available) > 0:
                selected.extend(available)
            selected = selected[:target]
        else:
            # If none available, skip (supplement from other difficulties later)
            selected = []
        
        final_skeletons.extend(selected)
    
    # If total insufficient, supplement from all available skeletons
    if len(final_skeletons) < total_skeletons:
        all_available = [s for s in combined_sql_skeletons if s not in final_skeletons]
        if len(all_available) > 0:
            needed = total_skeletons - len(final_skeletons)
            if needed <= len(all_available):
                final_skeletons.extend(random.sample(all_available, needed))
            else:
                final_skeletons.extend(all_available)
                # If still insufficient, repeat usage
                while len(final_skeletons) < total_skeletons:
                    final_skeletons.extend(random.choices(all_available, k=min(needed, len(all_available))))
                final_skeletons = final_skeletons[:total_skeletons]
    
    # Shuffle order to maintain randomness
    random.shuffle(final_skeletons)
    
    # Ensure total count is correct
    final_skeletons = final_skeletons[:total_skeletons]
    
    # Save results
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(final_skeletons, f, ensure_ascii=False, indent=2)
    
    return len(final_skeletons)

def process_all_databases(database_dir, expert_file, old_cfg_file, output_base_dir, old_data_file=None, new_logs_file=None, num_samples=100, total_skeletons=200, simple_ratio=0.5, medium_ratio=0.35, complex_ratio=0.15):
    """Process all databases"""
    # Create output directories
    cfg_dir = os.path.join(output_base_dir, 'ast_cfg')
    structure_dir = os.path.join(output_base_dir, 'sql_structure')
    skeleton_dir = os.path.join(output_base_dir, 'sql_skeleton')
    
    os.makedirs(cfg_dir, exist_ok=True)
    os.makedirs(structure_dir, exist_ok=True)
    os.makedirs(skeleton_dir, exist_ok=True)
    
    # Get all database names
    databases = []
    for item in os.listdir(database_dir):
        db_path = os.path.join(database_dir, item)
        if os.path.isdir(db_path):
            databases.append(item)
    
    print(f"Found {len(databases)} databases")
    print(f"Starting processing...\n")
    
    for db_name in tqdm(databases, desc="Processing databases"):
        print(f"\nProcessing database: {db_name}")
        
        # Step 1: generate CFG
        cfg_file = os.path.join(cfg_dir, f"{db_name}_ast_cfg.json")
        try:
            count = generate_cfg_for_database(expert_file, old_cfg_file, cfg_file, db_name)
            print(f"  ✓ Step 1: CFG generated, successfully parsed {count} entries")
        except Exception as e:
            print(f"  ✗ Step 1 failed: {e}")
            continue
        
        # Step 2: generate SQL structures
        structure_file = os.path.join(structure_dir, f"{db_name}_structure.json")
        try:
            structures = generate_structures_for_database(cfg_file, old_cfg_file, structure_file, db_name, num_samples)
            print(f"  ✓ Step 2: SQL structures generated, {len(structures)} total")
        except Exception as e:
            print(f"  ✗ Step 2 failed: {e}")
            continue
        
        # Step 3: generate SQL skeletons
        skeleton_file = os.path.join(skeleton_dir, f"{db_name}_sql_skeleton.json")
        try:
            count = generate_skeletons_for_database(structure_file, skeleton_file, old_data_file, new_logs_file, total_skeletons, db_name, simple_ratio, medium_ratio, complex_ratio)
            print(f"  ✓ Step 3: SQL skeletons generated, {count} total")
        except Exception as e:
            print(f"  ✗ Step 3 failed: {e}")
            continue

if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Generate CFG structures and SQL skeletons for each database in the beijing dataset (improved version)')
    # Set default paths (relative to script directory)
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(script_dir)))
    
    parser.add_argument('--database_dir', type=str, default=None, help='Database directory path (default: ../../data/beijing/database)')
    parser.add_argument('--expert_file', type=str, default=None, help='Expert examples file path (default: ../../data/target/expert_skeletons_beijing.json)')
    parser.add_argument('--old_cfg_file', type=str, default=None, help='Legacy CFG JSON (optional; skipped if missing)')
    parser.add_argument('--output_dir', type=str, default=None, help='Output directory path (default: benchmark/data/beijing/output)')
    parser.add_argument('--old_data_file', type=str, default=None, help='Legacy skeleton data JSON (optional; skipped if missing)')
    parser.add_argument('--new_logs_file', type=str, default=None, help='New log file (default: ../../data/target/expert_skeletons_beijing.json)')
    parser.add_argument('--num_samples', type=int, default=100, help='Number of structures to generate per database (default: 100)')
    parser.add_argument('--total_skeletons', type=int, default=200, help='Total skeletons to generate per database (default: 200)')
    parser.add_argument('--simple_ratio', type=float, default=0.5, help='Simple query ratio (default: 0.5, i.e. 50%%)')
    parser.add_argument('--medium_ratio', type=float, default=0.35, help='Medium query ratio (default: 0.35, i.e. 35%%)')
    parser.add_argument('--complex_ratio', type=float, default=0.15, help='Complex query ratio (default: 0.15, i.e. 15%%)')
    
    args = parser.parse_args()
    
    # Set default paths
    if args.database_dir is None:
        args.database_dir = os.path.join(project_root, 'benchmark', 'data', 'beijing', 'database')
    if args.expert_file is None:
        args.expert_file = os.path.join(project_root, 'benchmark', 'data', 'target', 'expert_skeletons_beijing.json')
    if args.output_dir is None:
        args.output_dir = os.path.join(project_root, 'benchmark', 'data', 'beijing', 'output')
    if args.new_logs_file is None:
        args.new_logs_file = os.path.join(project_root, 'benchmark', 'data', 'target', 'expert_skeletons_beijing.json')
    
    # Convert to absolute paths
    args.database_dir = os.path.abspath(args.database_dir)
    args.expert_file = os.path.abspath(args.expert_file)
    args.output_dir = os.path.abspath(args.output_dir)
    if args.old_cfg_file:
        args.old_cfg_file = os.path.abspath(args.old_cfg_file)
    if args.old_data_file:
        args.old_data_file = os.path.abspath(args.old_data_file)
    if args.new_logs_file:
        args.new_logs_file = os.path.abspath(args.new_logs_file)
    
    if not os.path.exists(args.database_dir):
        print(f"Error: Database directory does not exist: {args.database_dir}")
        sys.exit(1)
    
    if not os.path.exists(args.expert_file):
        print(f"Error: Expert examples file does not exist: {args.expert_file}")
        sys.exit(1)
    
    # old_cfg_file is optional; skip if not present
    if args.old_cfg_file and not os.path.exists(args.old_cfg_file):
        print(f"Warning: Old database CFG file does not exist: {args.old_cfg_file}, will skip old CFG data")
        args.old_cfg_file = None
    
    # Validate ratio sum
    ratio_sum = args.simple_ratio + args.medium_ratio + args.complex_ratio
    if abs(ratio_sum - 1.0) > 0.01:
        print(f"Warning: Difficulty ratio sum is {ratio_sum:.2f}, will auto-normalize")
        total = args.simple_ratio + args.medium_ratio + args.complex_ratio
        args.simple_ratio /= total
        args.medium_ratio /= total
        args.complex_ratio /= total
    
    process_all_databases(
        args.database_dir,
        args.expert_file,
        args.old_cfg_file,
        args.output_dir,
        args.old_data_file,
        args.new_logs_file,
        args.num_samples,
        args.total_skeletons,
        args.simple_ratio,
        args.medium_ratio,
        args.complex_ratio
    )
    
    print(f"\n{'='*60}")
    print("✓ All databases processed!")
    print(f"{'='*60}")

