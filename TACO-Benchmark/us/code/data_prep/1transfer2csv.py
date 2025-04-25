import os
import shutil
import pandas as pd
from tqdm import tqdm

# 检查文件是否是有效的 Excel 文件
def is_valid_excel(excel_path):
    try:
        # 尝试读取文件，检查文件是否有效
        pd.read_excel(excel_path, engine='openpyxl')
        return True
    except Exception as e:
        print(f"Error reading {excel_path}: {e}")
        return False

# 将 Excel 文件转换为 CSV 格式，并确保编码为 UTF-8
def convert_excel_to_csv(excel_path, csv_path):
    try:
        # 读取 Excel 文件（忽略合并单元格）
        df = pd.read_excel(excel_path, engine='openpyxl', header=None)
        
        # 将 DataFrame 转换为 CSV 格式，并确保编码为 UTF-8
        df.to_csv(csv_path, index=False, header=False, encoding='utf-8')
        print(f"Converted {excel_path} to {csv_path}")
    except Exception as e:
        print(f"Error converting {excel_path}: {e}")
        return False
    return True

# 统一转换所有 Excel 和 CSV 文件为标准 CSV 格式，并确保编码为 UTF-8
def process_files(raw_data_dir, raw_csv_data_dir):
    if not os.path.exists(raw_csv_data_dir):
        os.makedirs(raw_csv_data_dir)

    for folder_name in tqdm(os.listdir(raw_data_dir), desc='Processing folders'):
        folder_path = os.path.join(raw_data_dir, folder_name)
        if os.path.isdir(folder_path):
            # 创建对应的文件夹
            raw_csv_folder_path = os.path.join(raw_csv_data_dir, folder_name)
            os.makedirs(raw_csv_folder_path, exist_ok=True)
            
            for file_name in os.listdir(folder_path):
                original_file_path = os.path.join(folder_path, file_name)
                # 判断文件类型
                if file_name.endswith(('.xls', '.xlsx')):
                    # 如果是 Excel 文件
                    csv_file_name = file_name.replace('.xls', '.csv').replace('.xlsx', '.csv')
                    csv_file_name = csv_file_name.replace('.csvx', '.csv')  # 强制修改扩展名为 .csv
                    csv_file_path = os.path.join(raw_csv_folder_path, csv_file_name)
                    if is_valid_excel(original_file_path):
                        convert_excel_to_csv(original_file_path, csv_file_path)
                    else:
                        print(f"Skipping invalid Excel file: {file_name}")
                elif file_name.endswith('.csv'):
                    # 如果是 CSV 文件，读取并重新保存为 UTF-8 编码
                    csv_file_path = os.path.join(raw_csv_folder_path, file_name)
                    try:
                        # 读取 CSV 文件，尝试自动检测编码
                        with open(original_file_path, 'rb') as f:
                            raw_data = f.read()
                            detected_encoding = 'utf-8'  # 默认使用 UTF-8
                            try:
                                # 尝试检测文件编码
                                import chardet
                                result = chardet.detect(raw_data)
                                if result['encoding']:  # 如果检测到编码
                                    detected_encoding = result['encoding']
                                else:
                                    print(f"Encoding detection failed for {file_name}, using default UTF-8.")
                            except ImportError:
                                print("chardet not installed, using default UTF-8 encoding.")
                            
                            # 读取文件内容并重新保存为 UTF-8
                            content = raw_data.decode(detected_encoding, errors='replace')
                            with open(csv_file_path, 'w', encoding='utf-8') as f_out:
                                f_out.write(content)
                        print(f"Converted encoding and copied {file_name} to {csv_file_path}")
                    except Exception as e:
                        print(f"Error processing CSV file {file_name}: {e}")
                else:
                    print(f"Skipping non-Excel, non-CSV file: {file_name}")

# 定义目录
raw_data_dir = '../../data/raw_data'
raw_csv_data_dir = '../../data/raw_csv_data'

# 处理文件并转换为 CSV
process_files(raw_data_dir, raw_csv_data_dir)