import json
import random
import os
from tqdm import tqdm

def load_data(data_file):
    """
    read the JSON data file
    """
    with open(data_file, 'r', encoding='utf-8') as f:
        data_list = json.load(f)
    return data_list

def get_cfg_rules_list(data_list):
    """
    extract the CFG rules list from the data
    """
    cfg_rules_list = []
    for data in data_list:
        cfg_rules = data.get('cfg_rules', [])
        if cfg_rules:
            cfg_rules_list.append(cfg_rules)
    return cfg_rules_list

def generate_new_sql_structures(cfg_rules_list, num_samples):
    """
    generate new SQL structures based on CFG rules
    """
    # collect all unique CFG rules sequences
    unique_rule_sequences = set(tuple(seq) for seq in cfg_rules_list)
    all_rule_sequences = list(unique_rule_sequences)

    # initialize the generated structures list
    generated_structures = []

    # first, add all CFG rules sequences from the old database, ensure they are included in the generated structures
    generated_structures.extend(all_rule_sequences)

    # use tqdm to wrap the loop, show the progress bar
    with tqdm(total=num_samples, desc="Generating SQL structures") as pbar:
        # update the progress bar, because the initial CFG rules sequences have been added
        pbar.update(len(generated_structures))

        # apply valid transformations to generate new CFG rules sequences
        while len(generated_structures) < num_samples:
            # randomly select a CFG rules sequence
            seq = list(random.choice(all_rule_sequences))

            # apply valid transformations (e.g., swap two non-terminals)
            transformed_seq = apply_valid_transformation(seq)

            # ensure the new sequence is unique
            seq_tuple = tuple(transformed_seq)
            if seq_tuple not in unique_rule_sequences:
                generated_structures.append(transformed_seq)
                unique_rule_sequences.add(seq_tuple)
                pbar.update(1)

    # if still insufficient, repeat the existing skeletons in proportion
    if len(generated_structures) < num_samples:
        additional_structures = list(generated_structures)
        while len(generated_structures) < num_samples:
            generated_structures.extend(additional_structures)
            pbar.update(len(additional_structures))
        # truncate to the specified number
        generated_structures = generated_structures[:num_samples]

    return generated_structures

def apply_valid_transformation(seq):
    """
    apply valid transformations, such as swapping two non-terminals
    """
    # copy the sequence to avoid modifying the original sequence
    new_seq = seq.copy()
    # define the indices of the rules that can be swapped (avoid swapping to generate invalid structures)
    indices = [i for i in range(len(new_seq)) if '->' in new_seq[i]]
    if len(indices) >= 2:
        i1, i2 = random.sample(indices, 2)
        # swap two rules
        new_seq[i1], new_seq[i2] = new_seq[i2], new_seq[i1]
    return new_seq

def generate_sql_structure_for_databases(parsed_data_dir, cfg_files_dir, output_dir, num_samples=1000):
    """
    generate SQL structures for each database and save to files
    """
    # traverse all database folders under parsed_data/
    for db_folder in os.listdir(parsed_data_dir):
        print(f"processing {db_folder}")
        db_folder_path = os.path.join(parsed_data_dir, db_folder)
        
        if os.path.isdir(db_folder_path):  # if it is a folder, each folder represents a database
            # the corresponding CFG file
            cfg_file = os.path.join(cfg_files_dir, f"{db_folder}_ast_cfg.json")
            
            # load the CFG rules of the database
            cfg_data = load_data(cfg_file)
            
            # extract the CFG rules sequence list
            cfg_rules_list = get_cfg_rules_list(cfg_data)
            
            # generate new SQL structures
            generated_structures = generate_new_sql_structures(cfg_rules_list, num_samples)

            # output file path
            output_file = os.path.join(output_dir, f"{db_folder}_structure.json")

            # save the generated SQL structures to the file
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(generated_structures, f, ensure_ascii=False, indent=2)
            
            print(f"{db_folder} SQL structures have been generated and saved to {output_file}")

if __name__ == '__main__':
    parsed_data_dir = os.path.join('..', '..', 'data', 'parsed_data')
    cfg_files_dir = os.path.join('..', '..', 'data', 'new_asf_cfg')
    output_dir = os.path.join('..', '..', 'data', 'new_sql_structure')

    os.makedirs(output_dir, exist_ok=True)

    # generate SQL structures for each database, num_samples represents the number of SQL structures to generate for each database
    num_samples = 100  # can be adjusted as needed
    generate_sql_structure_for_databases(parsed_data_dir, cfg_files_dir, output_dir, num_samples)