import os
import shutil
import pandas as pd
from pathlib import Path
from tqdm import tqdm

_BENCHMARK_DATA = Path(__file__).resolve().parents[2] / "data"

# Check whether a file is a valid Excel file
def is_valid_excel(excel_path):
    try:
        # Try reading the file to verify it is valid
        pd.read_excel(excel_path, engine='openpyxl')
        return True
    except Exception as e:
        print(f"Error reading {excel_path}: {e}")
        return False

# Convert Excel files to CSV with UTF-8 encoding
def convert_excel_to_csv(excel_path, csv_path):
    try:
        # Read Excel file (ignore merged cells)
        df = pd.read_excel(excel_path, engine='openpyxl', header=None)
        
        # Convert DataFrame to CSV with UTF-8 encoding
        df.to_csv(csv_path, index=False, header=False, encoding='utf-8')
        print(f"Converted {excel_path} to {csv_path}")
    except Exception as e:
        print(f"Error converting {excel_path}: {e}")
        return False
    return True

# Convert all Excel and CSV files to standard CSV format with UTF-8 encoding
def process_files(raw_data_dir, raw_csv_data_dir):
    if not os.path.exists(raw_csv_data_dir):
        os.makedirs(raw_csv_data_dir)

    for folder_name in tqdm(os.listdir(raw_data_dir), desc='Processing folders'):
        folder_path = os.path.join(raw_data_dir, folder_name)
        if os.path.isdir(folder_path):
            # Create corresponding folder
            raw_csv_folder_path = os.path.join(raw_csv_data_dir, folder_name)
            os.makedirs(raw_csv_folder_path, exist_ok=True)
            
            for file_name in os.listdir(folder_path):
                original_file_path = os.path.join(folder_path, file_name)
                # Determine file type
                if file_name.endswith(('.xls', '.xlsx')):
                    # Excel file
                    csv_file_name = file_name.replace('.xls', '.csv').replace('.xlsx', '.csv')
                    csv_file_name = csv_file_name.replace('.csvx', '.csv')  # force .csv extension
                    csv_file_path = os.path.join(raw_csv_folder_path, csv_file_name)
                    if is_valid_excel(original_file_path):
                        convert_excel_to_csv(original_file_path, csv_file_path)
                    else:
                        print(f"Skipping invalid Excel file: {file_name}")
                elif file_name.endswith('.csv'):
                    # CSV file: read and re-save as UTF-8
                    csv_file_path = os.path.join(raw_csv_folder_path, file_name)
                    try:
                        # Read CSV file and try to auto-detect encoding
                        with open(original_file_path, 'rb') as f:
                            raw_data = f.read()
                            detected_encoding = 'utf-8'  # default to UTF-8
                            try:
                                # Try to detect file encoding
                                import chardet
                                result = chardet.detect(raw_data)
                                if result['encoding']:  # encoding detected
                                    detected_encoding = result['encoding']
                                else:
                                    print(f"Encoding detection failed for {file_name}, using default UTF-8.")
                            except ImportError:
                                print("chardet not installed, using default UTF-8 encoding.")
                            
                            # Read file content and re-save as UTF-8
                            content = raw_data.decode(detected_encoding, errors='replace')
                            with open(csv_file_path, 'w', encoding='utf-8') as f_out:
                                f_out.write(content)
                        print(f"Converted encoding and copied {file_name} to {csv_file_path}")
                    except Exception as e:
                        print(f"Error processing CSV file {file_name}: {e}")
                else:
                    print(f"Skipping non-Excel, non-CSV file: {file_name}")

# Define directories
raw_data_dir = str(_BENCHMARK_DATA / "raw_data")
raw_csv_data_dir = str(_BENCHMARK_DATA / "raw_csv_data")

# Process files and convert to CSV
process_files(raw_data_dir, raw_csv_data_dir)
