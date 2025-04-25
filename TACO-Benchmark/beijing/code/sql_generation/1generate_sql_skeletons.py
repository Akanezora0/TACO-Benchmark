import json
import sqlparse
from sqlparse.tokens import Keyword, Name, Literal, DML, Whitespace
from sqlparse.sql import TokenList, Token

def is_keyword(token):
    return token.ttype in Keyword

def is_identifier(token):
    return token.ttype in Name

def is_literal(token):
    return token.ttype in Literal

def is_whitespace(token):
    return token.ttype in Whitespace

def replace_tokens(tokens):
    new_tokens = []
    for token in tokens:
        if token.is_group:
            # 递归处理嵌套的Token
            replaced = replace_tokens(token.tokens)
            new_group = token.__class__(replaced)
            new_tokens.append(new_group)
        elif is_keyword(token) or token.ttype == DML:
            new_tokens.append(Token(token.ttype, token.value.upper()))
        elif is_identifier(token) or is_literal(token):
            new_tokens.append(Token(token.ttype, '_'))
        elif is_whitespace(token):
            new_tokens.append(Token(token.ttype, ' '))
        else:
            new_tokens.append(Token(token.ttype, token.value))
    return new_tokens

def extract_sql_framework(sql_text):
    # 解析SQL语句
    parsed = sqlparse.parse(sql_text)
    if not parsed:
        return ''
    statement = parsed[0]
    # 替换Tokens
    new_tokens = replace_tokens(statement.tokens)
    # 重组SQL结构
    sql_framework = ''.join(str(token) for token in new_tokens)
    # 移除多余的空格
    sql_framework = ' '.join(sql_framework.split())
    return sql_framework

def process_sql_file(input_file, output_file):
    with open(input_file, 'r', encoding='utf-8') as infile:
        try:
            data_list = json.load(infile)  # 一次性读取整个JSON数组
            print("文件是JSON数组格式，共有{}条记录。".format(len(data_list)))
        except json.JSONDecodeError as e:
            print("文件不是有效的JSON数组格式：", e)
            return

        result_list = []
        for data in data_list:
            original_sql = data.get('sql', '')
            sql_framework = extract_sql_framework(original_sql)
            # 构建新的JSON对象
            new_data = {
                'query': data.get('query', ''),
                'sql': original_sql,
                'sql_framework': sql_framework
            }
            result_list.append(new_data)

            # result_list.append(sql_framework)

        with open(output_file, 'w', encoding='utf-8') as outfile:
            json.dump(result_list, outfile, ensure_ascii=False, indent=4)

if __name__ == '__main__':
    import os
    script_dir = os.path.dirname(os.path.abspath(__file__))

    # input_file = os.path.join(script_dir, '..', '..', 'data', 'old_database', '12345_200.json')
    # output_file = os.path.join(script_dir, '..', '..', 'data', '12345_sql_skeletons.json')

    input_file = os.path.join(script_dir, '..', '..', 'data', 'logs', 'new_sql_logs.json')
    output_file = os.path.join(script_dir, '..', '..', 'data', 'new_sql_skeletons.json')

    process_sql_file(input_file, output_file)
    print("SQL骨架提取完成，结果保存在", output_file)