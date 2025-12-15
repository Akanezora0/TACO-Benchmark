#!/usr/bin/env python3
"""
为beijing数据集的每个数据库生成CFG结构和SQL骨架（改进版）

关键改进：
1. 结合旧数据库CFG规则和新数据库专家例子，为每个数据库生成不同的结果
2. 增加随机性和多样性（旋转、剪枝、合并等转换策略）
3. 改进SQL骨架生成逻辑，支持JOIN、子查询、聚合等复杂结构
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

# Node类定义
class Node:
    def __init__(self, symbol):
        self.symbol = symbol
        self.children = []

def get_cfg_rules(node, rules=None):
    """递归遍历AST，生成CFG规则序列"""
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
    """将AST节点转换为字典"""
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
    """为单个数据库生成AST/CFG文件（结合旧数据库CFG规则）"""
    # 加载专家例子
    with open(expert_file, 'r', encoding='utf-8') as f:
        expert_data = json.load(f)
    
    # 加载旧数据库CFG规则（用于扩充）
    old_cfg_rules_list = []
    if old_cfg_file and os.path.exists(old_cfg_file):
        with open(old_cfg_file, 'r', encoding='utf-8') as f:
            old_cfg_data = json.load(f)
        for data in old_cfg_data:
            cfg_rules = data.get('cfg_rules', [])
            if cfg_rules:
                old_cfg_rules_list.append(cfg_rules)
    
    # 为每个数据库设置不同的随机种子（基于数据库名称）
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
    
    # 如果旧数据库CFG规则存在，可以添加一些扩充的CFG规则
    if old_cfg_rules_list:
        # 随机选择一些旧数据库的CFG规则作为补充
        num_old_to_add = min(10, len(old_cfg_rules_list))
        selected_old_rules = random.sample(old_cfg_rules_list, num_old_to_add)
        for old_cfg_rules in selected_old_rules:
            # 创建补充数据项
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
    """应用高级转换策略（旋转、剪枝、合并等）"""
    new_seq = seq.copy()
    
    # 策略1: 交换规则顺序（30%概率）
    if random.random() < 0.3:
        indices = [i for i in range(len(new_seq)) if '->' in new_seq[i]]
        if len(indices) >= 2:
            i1, i2 = random.sample(indices, 2)
            new_seq[i1], new_seq[i2] = new_seq[i2], new_seq[i1]
    
    # 策略2: 插入旧数据库的CFG规则片段（20%概率）
    elif random.random() < 0.5 and old_cfg_rules_list:
        if random.random() < 0.2:
            old_rule_seq = random.choice(old_cfg_rules_list)
            # 随机插入一些旧规则
            insert_pos = random.randint(0, len(new_seq))
            insert_rules = random.sample(old_rule_seq, min(3, len(old_rule_seq)))
            new_seq = new_seq[:insert_pos] + insert_rules + new_seq[insert_pos:]
    
    # 策略3: 复制并修改规则（30%概率）
    elif random.random() < 0.8:
        indices = [i for i in range(len(new_seq)) if '->' in new_seq[i]]
        if indices:
            idx = random.choice(indices)
            rule = new_seq[idx]
            # 复制规则并可能修改
            new_seq.insert(idx + 1, rule)
    
    # 策略4: 删除规则（10%概率，但确保至少保留基本结构）
    elif random.random() < 0.9 and len(new_seq) > 5:
        indices = [i for i in range(len(new_seq)) if '->' in new_seq[i]]
        if len(indices) > 3:
            idx_to_remove = random.choice(indices[1:-1])  # 不删除第一个和最后一个
            new_seq.pop(idx_to_remove)
    
    return new_seq

def generate_structures_for_database(cfg_file, old_cfg_file, output_file, db_name, num_samples=100):
    """为单个数据库生成SQL结构（结合旧数据库CFG规则）"""
    with open(cfg_file, 'r', encoding='utf-8') as f:
        cfg_data = json.load(f)
    
    # 加载旧数据库CFG规则
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
    
    # 为每个数据库设置不同的随机种子
    random.seed(hash(db_name) % (2**32))
    
    # 收集所有唯一的CFG规则序列
    unique_rule_sequences = set(tuple(seq) for seq in cfg_rules_list)
    all_rule_sequences = list(unique_rule_sequences)
    
    # 初始化生成的结构列表
    generated_structures = []
    generated_structures.extend(all_rule_sequences)
    
    # 生成新结构（使用高级转换策略）
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
    
    # 如果仍然不足，按比例重复
    if len(generated_structures) < num_samples:
        additional = list(generated_structures)
        while len(generated_structures) < num_samples:
            generated_structures.extend(additional)
        generated_structures = generated_structures[:num_samples]
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(generated_structures, f, ensure_ascii=False, indent=2)
    
    return generated_structures

def build_parse_tree_from_cfg_rules(cfg_rules):
    """从CFG规则构建解析树"""
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
    """从解析树生成SQL骨架（改进版，支持更多结构，参考原始代码）"""
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
                # 如果出现嵌套的 SELECT，添加括号
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
        # 递归处理其他符号
        result = ' '.join(generate_sql_skeleton(child) for child in node.children)
        return result.strip()

def cfg_rules_to_sql_skeleton(cfg_rules):
    """将CFG规则序列转换回SQL骨架"""
    parse_tree = build_parse_tree_from_cfg_rules(cfg_rules)
    if parse_tree:
        return generate_sql_skeleton(parse_tree)
    return ''

def is_valid_sql_skeleton(sql_skeleton):
    """检查SQL骨架是否有效（改进版，修复语法错误）"""
    if not sql_skeleton or not sql_skeleton.upper().startswith('SELECT'):
        return False
    
    # 检查基本语法错误
    # 1. FROM后面必须有表名（不能是空）
    if 'FROM' in sql_skeleton.upper():
        from_pos = sql_skeleton.upper().find('FROM')
        after_from = sql_skeleton[from_pos+4:].strip()
        # FROM后面如果是WHERE、HAVING、GROUP BY、ORDER BY、LIMIT或空，则无效
        if not after_from or any(after_from.upper().startswith(kw) for kw in ['WHERE', 'HAVING', 'GROUP', 'ORDER', 'LIMIT', ';']):
            return False
    
    # 2. WHERE前面必须有FROM和表名，且WHERE后面必须有条件
    if 'WHERE' in sql_skeleton.upper():
        where_pos = sql_skeleton.upper().find('WHERE')
        before_where = sql_skeleton[:where_pos].strip()
        if 'FROM' not in before_where.upper():
            return False
        from_pos = before_where.upper().rfind('FROM')
        after_from = before_where[from_pos+4:].strip()
        if not after_from or after_from == '':
            return False
        # WHERE后面必须有内容
        after_where = sql_skeleton[where_pos+5:].strip()
        if not after_where or after_where in [';', '']:
            return False
    
    # 3. HAVING前面必须有GROUP BY
    if 'HAVING' in sql_skeleton.upper():
        having_pos = sql_skeleton.upper().find('HAVING')
        before_having = sql_skeleton[:having_pos].upper()
        if 'GROUP BY' not in before_having:
            return False
    
    # 4. 检查子查询是否在括号内
    select_positions = [m.start() for m in re.finditer(r'\bSELECT\b', sql_skeleton, re.IGNORECASE)]
    if len(select_positions) <= 1:
        return True
    
    for pos in select_positions[1:]:
        before_select = sql_skeleton[:pos]
        if before_select.count('(') <= before_select.count(')'):
            return False
    return True

def sql_query_to_sql_skeleton(sql_query):
    """将SQL查询转换为SQL骨架"""
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
    """分类SQL骨架难度
    
    Returns:
        'simple': 单表，无JOIN，无子查询，无GROUP BY
        'medium': 有JOIN（最多1个）或GROUP BY，但无子查询、无UNION、无HAVING
        'complex': 有子查询、UNION、HAVING，或多表JOIN（2个以上）
    """
    skeleton_upper = skeleton.upper()
    
    has_join = 'JOIN' in skeleton_upper
    has_subquery = '(' in skeleton and 'SELECT' in skeleton_upper and skeleton.count('SELECT') > 1
    has_group_by = 'GROUP BY' in skeleton_upper
    has_having = 'HAVING' in skeleton_upper
    has_union = 'UNION' in skeleton_upper
    join_count = skeleton_upper.count('JOIN')
    
    # 简单：单表，无JOIN，无子查询，无GROUP BY
    if not has_join and not has_subquery and not has_group_by:
        return 'simple'
    
    # 复杂：有子查询、UNION、HAVING，或多表JOIN（2个以上）
    if has_subquery or has_union or has_having or join_count >= 2:
        return 'complex'
    
    # 中等：其他情况（有JOIN但最多1个，或有GROUP BY，但无子查询/UNION/HAVING）
    return 'medium'

def generate_skeletons_for_database(structure_file, output_file, old_data_file=None, new_logs_file=None, total_skeletons=200, db_name=None, simple_ratio=0.5, medium_ratio=0.35, complex_ratio=0.15):
    """为单个数据库生成SQL骨架（参考原始代码逻辑）"""
    with open(structure_file, 'r', encoding='utf-8') as f:
        structures = json.load(f)
    
    # 为每个数据库设置不同的随机种子
    if db_name:
        random.seed(hash(db_name) % (2**32))
    
    # 从CFG规则生成SQL骨架
    generated_skeletons = []
    for cfg_rules in structures:
        sql_skeleton = cfg_rules_to_sql_skeleton(cfg_rules)
        # 确保 SQL skeleton 以 SELECT 开头且有效
        if sql_skeleton and sql_skeleton.lower().startswith('select') and is_valid_sql_skeleton(sql_skeleton):
            generated_skeletons.append(sql_skeleton)
    
    # 移除重复的 skeleton
    generated_skeletons = list(set(generated_skeletons))
    
    # 加载旧数据库的数据（参考原始代码）
    old_sql_skeletons = []
    if old_data_file and os.path.exists(old_data_file):
        with open(old_data_file, 'r', encoding='utf-8') as f:
            old_data = json.load(f)
        old_sql_queries = [entry.get('sql', '') for entry in old_data if entry.get('sql', '')]
        old_sql_skeletons = [sql_query_to_sql_skeleton(sql_query) for sql_query in old_sql_queries]
        old_sql_skeletons = [s for s in old_sql_skeletons if s.lower().startswith('select')]
    
    # 加载新日志文件（参考原始代码）
    log_sql_skeletons = []
    if new_logs_file and os.path.exists(new_logs_file):
        with open(new_logs_file, 'r', encoding='utf-8') as f:
            new_logs = json.load(f)
        log_sql_skeletons = [sql_query_to_sql_skeleton(log_entry.get('sql', '')) for log_entry in new_logs if log_entry.get('sql', '')]
        log_sql_skeletons = [s for s in log_sql_skeletons if s.lower().startswith('select')]
    
    # 确保旧数据库的 skeleton 包含在最终结果中，并且以 SELECT 开头
    combined_sql_skeletons = generated_skeletons + [s for s in old_sql_skeletons if s.lower().startswith('select')]
    combined_sql_skeletons = list(set(combined_sql_skeletons))
    
    # 过滤掉无效的 skeleton
    combined_sql_skeletons = [s for s in combined_sql_skeletons if is_valid_sql_skeleton(s) and s.lower().startswith('select')]
    
    # 计算需要从日志中包含的 skeleton 数量（参考原始代码）
    num_combined = len(combined_sql_skeletons)
    num_logs_to_include = total_skeletons - num_combined
    if num_logs_to_include > 0 and log_sql_skeletons:
        # 随机选择日志 skeleton
        log_sql_skeletons = [s for s in log_sql_skeletons if s.lower().startswith('select')]
        log_sql_skeletons = list(set(log_sql_skeletons))
        if len(log_sql_skeletons) > 0:
            if num_logs_to_include > len(log_sql_skeletons):
                # 如果需要的数量大于可用的日志 skeleton 数量，使用重复选择
                selected_log_skeletons = random.choices(log_sql_skeletons, k=num_logs_to_include)
            else:
                selected_log_skeletons = random.sample(log_sql_skeletons, k=num_logs_to_include)
            combined_sql_skeletons.extend(selected_log_skeletons)
    
    # 如果总数仍不足，先补充到足够数量
    while len(combined_sql_skeletons) < total_skeletons:
        combined_sql_skeletons.extend(combined_sql_skeletons)
    combined_sql_skeletons = combined_sql_skeletons[:total_skeletons * 2]  # 多生成一些，以便后续筛选
    
    # 按难度分类所有骨架
    skeletons_by_difficulty = {'simple': [], 'medium': [], 'complex': []}
    for skeleton in combined_sql_skeletons:
        difficulty = classify_skeleton_difficulty(skeleton)
        skeletons_by_difficulty[difficulty].append(skeleton)
    
    # 计算目标数量
    target_counts = {
        'simple': int(total_skeletons * simple_ratio),
        'medium': int(total_skeletons * medium_ratio),
        'complex': int(total_skeletons * complex_ratio)
    }
    
    # 确保总数正确（处理舍入误差）
    actual_total = sum(target_counts.values())
    if actual_total < total_skeletons:
        target_counts['simple'] += (total_skeletons - actual_total)
    
    # 按目标比例选择骨架
    final_skeletons = []
    for difficulty in ['simple', 'medium', 'complex']:
        available = skeletons_by_difficulty[difficulty]
        target = target_counts[difficulty]
        
        if len(available) >= target:
            # 随机选择目标数量
            selected = random.sample(available, target)
        elif len(available) > 0:
            # 如果不足，全部使用并重复
            selected = available.copy()
            while len(selected) < target and len(available) > 0:
                selected.extend(available)
            selected = selected[:target]
        else:
            # 如果没有可用的，跳过（后续会从其他难度补充）
            selected = []
        
        final_skeletons.extend(selected)
    
    # 如果总数不足，从所有可用的骨架中补充
    if len(final_skeletons) < total_skeletons:
        all_available = [s for s in combined_sql_skeletons if s not in final_skeletons]
        if len(all_available) > 0:
            needed = total_skeletons - len(final_skeletons)
            if needed <= len(all_available):
                final_skeletons.extend(random.sample(all_available, needed))
            else:
                final_skeletons.extend(all_available)
                # 如果还不够，重复使用
                while len(final_skeletons) < total_skeletons:
                    final_skeletons.extend(random.choices(all_available, k=min(needed, len(all_available))))
                final_skeletons = final_skeletons[:total_skeletons]
    
    # 打乱顺序，保持随机性
    random.shuffle(final_skeletons)
    
    # 确保总数正确
    final_skeletons = final_skeletons[:total_skeletons]
    
    # 保存结果
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(final_skeletons, f, ensure_ascii=False, indent=2)
    
    return len(final_skeletons)

def process_all_databases(database_dir, expert_file, old_cfg_file, output_base_dir, old_data_file=None, new_logs_file=None, num_samples=100, total_skeletons=200, simple_ratio=0.5, medium_ratio=0.35, complex_ratio=0.15):
    """处理所有数据库"""
    # 创建输出目录
    cfg_dir = os.path.join(output_base_dir, 'ast_cfg')
    structure_dir = os.path.join(output_base_dir, 'sql_structure')
    skeleton_dir = os.path.join(output_base_dir, 'sql_skeleton')
    
    os.makedirs(cfg_dir, exist_ok=True)
    os.makedirs(structure_dir, exist_ok=True)
    os.makedirs(skeleton_dir, exist_ok=True)
    
    # 获取所有数据库名称
    databases = []
    for item in os.listdir(database_dir):
        db_path = os.path.join(database_dir, item)
        if os.path.isdir(db_path):
            databases.append(item)
    
    print(f"找到 {len(databases)} 个数据库")
    print(f"开始处理...\n")
    
    for db_name in tqdm(databases, desc="处理数据库"):
        print(f"\n处理数据库: {db_name}")
        
        # 步骤1: 生成CFG
        cfg_file = os.path.join(cfg_dir, f"{db_name}_ast_cfg.json")
        try:
            count = generate_cfg_for_database(expert_file, old_cfg_file, cfg_file, db_name)
            print(f"  ✓ 步骤1: 生成CFG，成功解析 {count} 条")
        except Exception as e:
            print(f"  ✗ 步骤1失败: {e}")
            continue
        
        # 步骤2: 生成SQL结构
        structure_file = os.path.join(structure_dir, f"{db_name}_structure.json")
        try:
            structures = generate_structures_for_database(cfg_file, old_cfg_file, structure_file, db_name, num_samples)
            print(f"  ✓ 步骤2: 生成SQL结构，共 {len(structures)} 个")
        except Exception as e:
            print(f"  ✗ 步骤2失败: {e}")
            continue
        
        # 步骤3: 生成SQL骨架
        skeleton_file = os.path.join(skeleton_dir, f"{db_name}_sql_skeleton.json")
        try:
            count = generate_skeletons_for_database(structure_file, skeleton_file, old_data_file, new_logs_file, total_skeletons, db_name, simple_ratio, medium_ratio, complex_ratio)
            print(f"  ✓ 步骤3: 生成SQL骨架，共 {count} 个")
        except Exception as e:
            print(f"  ✗ 步骤3失败: {e}")
            continue

if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='为beijing数据集的每个数据库生成CFG结构和SQL骨架（改进版）')
    # 设置默认路径（相对于脚本目录）
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(script_dir)))
    
    parser.add_argument('--database_dir', type=str, default=None, help='数据库目录路径（默认：../../data/beijing/database）')
    parser.add_argument('--expert_file', type=str, default=None, help='专家例子文件路径（默认：../../data/target/expert_skeletons_beijing.json）')
    parser.add_argument('--old_cfg_file', type=str, default=None, help='旧数据库CFG文件路径（默认：../../old/saturn/TACO-Benchmark-all/beijing/data/old_ast_cfg.json）')
    parser.add_argument('--output_dir', type=str, default=None, help='输出目录路径（默认：../../data/beijing/output）')
    parser.add_argument('--old_data_file', type=str, default=None, help='旧数据文件（默认：../../old/saturn/TACO-Benchmark-all/beijing/data/xcity_sql_skeletons.json）')
    parser.add_argument('--new_logs_file', type=str, default=None, help='新日志文件（默认：../../data/target/expert_skeletons_beijing.json）')
    parser.add_argument('--num_samples', type=int, default=100, help='每个数据库生成的结构数量（默认100）')
    parser.add_argument('--total_skeletons', type=int, default=200, help='每个数据库生成的骨架总数（默认200）')
    parser.add_argument('--simple_ratio', type=float, default=0.5, help='简单查询比例（默认0.5，即50%%）')
    parser.add_argument('--medium_ratio', type=float, default=0.35, help='中等查询比例（默认0.35，即35%%）')
    parser.add_argument('--complex_ratio', type=float, default=0.15, help='复杂查询比例（默认0.15，即15%%）')
    
    args = parser.parse_args()
    
    # 设置默认路径
    if args.database_dir is None:
        args.database_dir = os.path.join(project_root, 'benchmark', 'data', 'beijing', 'database')
    if args.expert_file is None:
        args.expert_file = os.path.join(project_root, 'benchmark', 'data', 'target', 'expert_skeletons_beijing.json')
    if args.old_cfg_file is None:
        args.old_cfg_file = os.path.join(project_root, 'old', 'saturn', 'TACO-Benchmark-all', 'beijing', 'data', 'old_ast_cfg.json')
    if args.output_dir is None:
        args.output_dir = os.path.join(project_root, 'benchmark', 'data', 'beijing', 'output')
    if args.old_data_file is None:
        args.old_data_file = os.path.join(project_root, 'old', 'saturn', 'TACO-Benchmark-all', 'beijing', 'data', 'xcity_sql_skeletons.json')
    if args.new_logs_file is None:
        args.new_logs_file = os.path.join(project_root, 'benchmark', 'data', 'target', 'expert_skeletons_beijing.json')
    
    # 转换为绝对路径
    args.database_dir = os.path.abspath(args.database_dir)
    args.expert_file = os.path.abspath(args.expert_file)
    args.old_cfg_file = os.path.abspath(args.old_cfg_file)
    args.output_dir = os.path.abspath(args.output_dir)
    if args.old_data_file:
        args.old_data_file = os.path.abspath(args.old_data_file)
    if args.new_logs_file:
        args.new_logs_file = os.path.abspath(args.new_logs_file)
    
    if not os.path.exists(args.database_dir):
        print(f"错误: 数据库目录不存在: {args.database_dir}")
        sys.exit(1)
    
    if not os.path.exists(args.expert_file):
        print(f"错误: 专家例子文件不存在: {args.expert_file}")
        sys.exit(1)
    
    # old_cfg_file是可选的，如果不存在则跳过
    if args.old_cfg_file and not os.path.exists(args.old_cfg_file):
        print(f"警告: 旧数据库CFG文件不存在: {args.old_cfg_file}，将跳过使用旧CFG数据")
        args.old_cfg_file = None
    
    # 验证比例总和
    ratio_sum = args.simple_ratio + args.medium_ratio + args.complex_ratio
    if abs(ratio_sum - 1.0) > 0.01:
        print(f"警告: 难度比例总和为 {ratio_sum:.2f}，将自动归一化")
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
    print("✓ 所有数据库处理完成！")
    print(f"{'='*60}")

