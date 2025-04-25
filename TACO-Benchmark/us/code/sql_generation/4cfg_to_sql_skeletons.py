import json
import os
import re
import random

class Node:
    def __init__(self, symbol):
        self.symbol = symbol
        self.children = []

def build_parse_tree_from_cfg_rules(cfg_rules):
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
                # 检查下一个规则是否与当前符号匹配
                if index < len(cfg_rules) and cfg_rules[index].startswith(symbol + ' ->'):
                    child = build_node()
                    if child:
                        node.children.append(child)
                    else:
                        # 如果无法构建子节点，返回 None
                        return None
                else:
                    # 如果没有匹配的规则，创建一个叶子节点
                    child = Node(symbol)
                    node.children.append(child)
        return node

    root = build_node()
    return root

def generate_sql_skeleton(node):
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
        return ', '.join(elements)

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
    parse_tree = build_parse_tree_from_cfg_rules(cfg_rules)
    if parse_tree:
        sql_skeleton = generate_sql_skeleton(parse_tree)
        return sql_skeleton
    else:
        return ''  # 如果无法构建解析树，返回空字符串

def is_valid_sql_skeleton(sql_skeleton):
    # 检查是否存在多个未被括号包含的 SELECT
    select_positions = [m.start() for m in re.finditer(r'\bSELECT\b', sql_skeleton, re.IGNORECASE)]
    if len(select_positions) <= 1:
        return True  # 只有一个 SELECT，肯定有效

    # 检查每个 SELECT 是否在括号内
    for pos in select_positions[1:]:
        before_select = sql_skeleton[:pos]
        open_parens = before_select.count('(')
        close_parens = before_select.count(')')
        if open_parens <= close_parens:
            # SELECT 不在括号内，语法无效
            return False
    return True

def sql_query_to_sql_skeleton(sql_query):
    # 替换引号中的字符串为 '_'
    sql_skeleton = re.sub(r"'[^']*'", '_', sql_query)
    sql_skeleton = re.sub(r'"[^"]*"', '_', sql_skeleton)
    # 替换数字为 '_'
    sql_skeleton = re.sub(r'\b\d+\b', '_', sql_skeleton)
    # 替换标识符（表名和列名）为 '_'
    sql_keywords = set(['SELECT', 'FROM', 'WHERE', 'GROUP', 'BY', 'HAVING', 'ORDER', 'LIMIT', 'JOIN', 'ON', 'AS', 'AND', 'OR', 'IN', 'NOT', 'NULL', 'IS', 'DISTINCT', 'COUNT', 'SUM', 'AVG', 'MIN', 'MAX', 'INNER', 'LEFT', 'RIGHT', 'FULL', 'OUTER', 'UNION', 'ALL'])
    # 函数替换标识符
    def replace_identifier(match):
        word = match.group(0)
        if word.upper() in sql_keywords or word == '*':
            return word
        else:
            return '_'
    sql_skeleton = re.sub(r'\b\w+\b', replace_identifier, sql_skeleton)
    # 替换多个连续的 '_' 为单个 '_'
    sql_skeleton = re.sub(r'(_\s*)+', '_ ', sql_skeleton)
    # 移除多余的空格
    sql_skeleton = ' '.join(sql_skeleton.strip().split())
    return sql_skeleton

def process_generated_structures(structures_file, output_file, new_logs_file, old_data_file, total_skeletons, log_ratio=0.3):
    # 加载生成的 CFG 规则序列
    with open(structures_file, 'r', encoding='utf-8') as f:
        structures = json.load(f)

    generated_sql_skeletons = []
    for idx, cfg_rules in enumerate(structures):
        sql_skeleton = cfg_rules_to_sql_skeleton(cfg_rules)
        # 新增检查：确保 SQL skeleton 以 SELECT 开头
        if sql_skeleton and sql_skeleton.lower().startswith('select') and is_valid_sql_skeleton(sql_skeleton):
            generated_sql_skeletons.append(sql_skeleton)
        else:
            # 跳过无效的 skeleton
            continue

    # 移除重复的 skeleton
    generated_sql_skeletons = list(set(generated_sql_skeletons))

    # 加载新数据库的 SQL 日志
    with open(new_logs_file, 'r', encoding='utf-8') as f:
        new_sql_logs = json.load(f)

    # 加载旧数据库的数据
    with open(old_data_file, 'r', encoding='utf-8') as f:
        old_data = json.load(f)

    # 提取旧数据库的 SQL 查询
    old_sql_queries = []
    for entry in old_data:
        sql_query = entry.get('sql', '')
        if sql_query:
            old_sql_queries.append(sql_query)

    # 提取 SQL skeleton
    log_sql_skeletons = [sql_query_to_sql_skeleton(log_entry['sql']) for log_entry in new_sql_logs]
    old_sql_skeletons = [sql_query_to_sql_skeleton(sql_query) for sql_query in old_sql_queries]

    # 确保旧数据库的 skeleton 包含在最终结果中，并且以 SELECT 开头
    combined_sql_skeletons = generated_sql_skeletons + [s for s in old_sql_skeletons if s.lower().startswith('select')]
    combined_sql_skeletons = list(set(combined_sql_skeletons))

    # 过滤掉无效的 skeleton
    combined_sql_skeletons = [s for s in combined_sql_skeletons if is_valid_sql_skeleton(s) and s.lower().startswith('select')]

    # 计算需要从日志中包含的 skeleton 数量
    num_combined = len(combined_sql_skeletons)
    num_logs_to_include = total_skeletons - num_combined
    if num_logs_to_include > 0:
        # 随机选择日志 skeleton
        log_sql_skeletons = [s for s in log_sql_skeletons if s.lower().startswith('select')]
        log_sql_skeletons = list(set(log_sql_skeletons))
        random.seed(42)
        if num_logs_to_include > len(log_sql_skeletons):
            # 如果需要的数量大于可用的日志 skeleton 数量，使用重复选择
            selected_log_skeletons = random.choices(log_sql_skeletons, k=num_logs_to_include)
        else:
            selected_log_skeletons = random.sample(log_sql_skeletons, k=num_logs_to_include)
        combined_sql_skeletons.extend(selected_log_skeletons)

    # 如果总数仍不足，则按照比例重复 skeleton
    while len(combined_sql_skeletons) < total_skeletons:
        combined_sql_skeletons.extend(combined_sql_skeletons)
    combined_sql_skeletons = combined_sql_skeletons[:total_skeletons]

    # 打乱顺序
    random.shuffle(combined_sql_skeletons)

    # 保存结果
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(combined_sql_skeletons, f, ensure_ascii=False, indent=2)

    print(f"生成了 {len(combined_sql_skeletons)} 个有效的 SQL skeleton，保存在 {output_file}")

def process_all_databases(new_structure_dir, new_skeleton_dir, new_logs_file, old_data_file, total_skeletons_per_db=1000, log_ratio=0.1):
    # 确保输出目录存在
    os.makedirs(new_skeleton_dir, exist_ok=True)

    # 加载新日志文件一次，避免重复读取
    with open(new_logs_file, 'r', encoding='utf-8') as f:
        new_sql_logs = json.load(f)

    # 加载旧数据文件一次，避免重复读取
    with open(old_data_file, 'r', encoding='utf-8') as f:
        old_data = json.load(f)

    # 提取旧数据库的 SQL skeletons
    old_sql_queries = [entry.get('sql', '') for entry in old_data if entry.get('sql', '')]
    old_sql_skeletons = [sql_query_to_sql_skeleton(sql_query) for sql_query in old_sql_queries]
    old_sql_skeletons = [s for s in old_sql_skeletons if s.lower().startswith('select')]

    # 提取日志中的 SQL skeletons
    log_sql_skeletons = [sql_query_to_sql_skeleton(log_entry['sql']) for log_entry in new_sql_logs]
    log_sql_skeletons = [s for s in log_sql_skeletons if s.lower().startswith('select')]

    # 遍历新结构目录下的所有 JSON 文件
    for file_name in os.listdir(new_structure_dir):
        if file_name.endswith('_structure.json'):
            database_name = file_name.replace('_structure.json', '')
            structure_file = os.path.join(new_structure_dir, file_name)
            output_file = os.path.join(new_skeleton_dir, f"{database_name}_sql_skeleton.json")

            # 加载 CFG 规则
            with open(structure_file, 'r', encoding='utf-8') as f:
                cfg_rules = json.load(f)

            # 生成 SQL skeletons
            generated_sql_skeletons = []
            for idx, cfg_rule in enumerate(cfg_rules):
                sql_skeleton = cfg_rules_to_sql_skeleton(cfg_rule)
                # 确保 skeleton 以 SELECT 开头且有效
                if sql_skeleton and sql_skeleton.lower().startswith('select') and is_valid_sql_skeleton(sql_skeleton):
                    generated_sql_skeletons.append(sql_skeleton)

            # 移除重复的 skeleton
            generated_sql_skeletons = list(set(generated_sql_skeletons))

            # 结合旧 skeletons 和日志 skeletons
            combined_sql_skeletons = generated_sql_skeletons + [s for s in old_sql_skeletons]
            combined_sql_skeletons = list(set(combined_sql_skeletons))
            combined_sql_skeletons = [s for s in combined_sql_skeletons if is_valid_sql_skeleton(s) and s.lower().startswith('select')]

            # 计算需要从日志中包含的 skeleton 数量
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

            # 如果总数仍不足，则按照比例重复 skeleton
            while len(combined_sql_skeletons) < total_skeletons_per_db:
                combined_sql_skeletons.extend(combined_sql_skeletons)
            combined_sql_skeletons = combined_sql_skeletons[:total_skeletons_per_db]

            # 打乱顺序
            random.shuffle(combined_sql_skeletons)

            # 保存结果
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(combined_sql_skeletons, f, ensure_ascii=False, indent=2)

            print(f"数据库 '{database_name}' 生成了 {len(combined_sql_skeletons)} 个有效的 SQL skeleton，保存在 {output_file}")

if __name__ == '__main__':
    # 获取脚本所在目录
    script_dir = os.path.dirname(os.path.abspath(__file__))

    # 定义新的输入和输出目录
    new_structure_dir = os.path.join(script_dir, '..', '..', 'data', 'new_sql_structure')
    new_skeleton_dir = os.path.join(script_dir, '..', '..', 'data', 'new_sql_skeleton')

    # 定义日志和旧数据文件路径
    new_logs_file = os.path.join(script_dir, '..', '..', 'data', 'new_sql_skeletons.json')
    old_data_file = os.path.join(script_dir, '..', '..', 'data', '12345_sql_skeletons.json')

    # 指定每个数据库生成的 SQL skeleton 总数
    total_skeletons_per_db = 200  # 可以根据需要调整

    # 处理所有数据库
    process_all_databases(new_structure_dir, new_skeleton_dir, new_logs_file, old_data_file, total_skeletons_per_db=total_skeletons_per_db, log_ratio=0.1)
