import json
import random
import os
from tqdm import tqdm

def load_data(data_file):
    """
    读取 JSON 数据文件
    """
    with open(data_file, 'r', encoding='utf-8') as f:
        data_list = json.load(f)
    return data_list

def get_cfg_rules_list(data_list):
    """
    提取数据中的 CFG 规则列表
    """
    cfg_rules_list = []
    for data in data_list:
        cfg_rules = data.get('cfg_rules', [])
        if cfg_rules:
            cfg_rules_list.append(cfg_rules)
    return cfg_rules_list

from tqdm import tqdm  # 导入 tqdm 库

def generate_new_sql_structures(cfg_rules_list, num_samples):
    """
    基于 CFG 规则生成新的 SQL 结构
    """
    # 收集所有唯一的 CFG 规则序列
    unique_rule_sequences = set(tuple(seq) for seq in cfg_rules_list)
    all_rule_sequences = list(unique_rule_sequences)

    # 初始化生成的结构列表
    generated_structures = []

    # 首先，添加旧数据库中的所有 CFG 规则序列，确保它们包含在生成的结构中
    generated_structures.extend(all_rule_sequences)

    # 使用 tqdm 包装循环，显示进度条
    with tqdm(total=num_samples, desc="Generating SQL structures") as pbar:
        # 更新进度条，因为已经添加了初始的 CFG 规则序列
        pbar.update(len(generated_structures))

        # 应用有效的转换来生成新的 CFG 规则序列
        while len(generated_structures) < num_samples:
            # 随机选择一个 CFG 规则序列
            seq = list(random.choice(all_rule_sequences))

            # 应用有效的转换（如交换两个非终结符）
            transformed_seq = apply_valid_transformation(seq)

            # 确保新序列是唯一的
            seq_tuple = tuple(transformed_seq)
            if seq_tuple not in unique_rule_sequences:
                generated_structures.append(transformed_seq)
                unique_rule_sequences.add(seq_tuple)
                pbar.update(1)  # 更新进度条

    # 如果仍然不足，则按照比例重复已有的 skeleton
    if len(generated_structures) < num_samples:
        additional_structures = list(generated_structures)
        while len(generated_structures) < num_samples:
            generated_structures.extend(additional_structures)
            pbar.update(len(additional_structures))  # 更新进度条
        # 截断到指定数量
        generated_structures = generated_structures[:num_samples]

    return generated_structures

def apply_valid_transformation(seq):
    """
    应用有效的转换，如交换两个非终结符
    """
    # 复制序列以避免修改原序列
    new_seq = seq.copy()
    # 定义可以交换的规则索引（避免交换产生无效结构）
    indices = [i for i in range(len(new_seq)) if '->' in new_seq[i]]
    if len(indices) >= 2:
        i1, i2 = random.sample(indices, 2)
        # 交换两个规则
        new_seq[i1], new_seq[i2] = new_seq[i2], new_seq[i1]
    return new_seq

def generate_sql_structure_for_databases(parsed_data_dir, cfg_files_dir, output_dir, num_samples=1000):
    """
    为每个数据库生成 SQL 结构并保存到文件
    """
    # 遍历 parsed_data/ 下的所有数据库文件夹
    for db_folder in os.listdir(parsed_data_dir):
        print(f"processing {db_folder}")
        db_folder_path = os.path.join(parsed_data_dir, db_folder)
        
        if os.path.isdir(db_folder_path):  # 如果是文件夹，每个文件夹代表一个数据库
            # 对应的 CFG 文件
            cfg_file = os.path.join(cfg_files_dir, f"{db_folder}_ast_cfg.json")
            
            # 加载该数据库的 CFG 规则
            cfg_data = load_data(cfg_file)
            
            # 提取 CFG 规则序列列表
            cfg_rules_list = get_cfg_rules_list(cfg_data)
            
            # 生成新的 SQL 结构
            generated_structures = generate_new_sql_structures(cfg_rules_list, num_samples)

            # 输出文件路径
            output_file = os.path.join(output_dir, f"{db_folder}_structure.json")

            # 将生成的 SQL 结构保存到文件
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(generated_structures, f, ensure_ascii=False, indent=2)
            
            print(f"{db_folder} 的 SQL 结构已生成并保存到 {output_file}")

if __name__ == '__main__':
    # 定义路径
    parsed_data_dir = os.path.join('..', '..', 'data', 'parsed_data')  # 使用已处理数据的 parsed_data 目录
    cfg_files_dir = os.path.join('..', '..', 'data', 'new_asf_cfg')  # 之前生成的 CFG 文件目录
    output_dir = os.path.join('..', '..', 'data', 'new_sql_structure')  # 生成的 SQL 结构文件保存目录

    # 创建保存 SQL 结构文件的目录（如果不存在）
    os.makedirs(output_dir, exist_ok=True)

    # 为每个数据库生成 SQL 结构，num_samples 表示每个数据库生成的 SQL 结构数量
    num_samples = 100  # 可以根据需要调整这个值
    generate_sql_structure_for_databases(parsed_data_dir, cfg_files_dir, output_dir, num_samples)