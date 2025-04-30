import os
import sqlite3
import pandas as pd
import json

# CSV to SQLite
def csv_to_sqlite(csv_folder_path, sqlite_db_path):
    # SQLite
    conn = sqlite3.connect(sqlite_db_path)
    cursor = conn.cursor()

    # database structure
    db_structure = {}

    # CSV
    for file_name in os.listdir(csv_folder_path):
        if file_name.endswith('.csv'):
            csv_path = os.path.join(csv_folder_path, file_name)
            try:
                # read
                df = pd.read_csv(csv_path)
                
                # NaN to ''
                df = df.fillna('')

                # table name
                table_name = file_name.replace('.csv', '')
                
                # DataFrame to SQLite
                df.to_sql(table_name, conn, if_exists='replace', index=False)
                print(f"Added table: {table_name} from {csv_path}")
                
                # columns
                db_structure[table_name] = {
                    'columns': df.columns.tolist(),
                    'data': df.to_dict(orient='records')
                }
            except Exception as e:
                print(f"Error processing file {csv_path}: {e}")
                continue

    # commit
    conn.commit()
    conn.close()

    # database structure
    return db_structure

# database structure and data
def save_db_structure_and_data_as_json(db_structure, db_folder_path, db_name):
    json_file_path = os.path.join(db_folder_path, f"{db_name}.json")
    
    try:
        with open(json_file_path, 'w', encoding='utf-8') as json_file:
            json.dump(db_structure, json_file, ensure_ascii=False, indent=4)
        print(f"Database structure and data saved as JSON: {json_file_path}")
    except Exception as e:
        print(f"Error saving JSON file {json_file_path}: {e}")

# process SQLite
def process_parsed_data_to_sqlite(parsed_data_dir, database_dir):
    # parsed_data
    for folder_name in os.listdir(parsed_data_dir):
        folder_path = os.path.join(parsed_data_dir, folder_name)
        if os.path.isdir(folder_path):
            try:
                # database
                db_folder_path = os.path.join(database_dir, folder_name)
                os.makedirs(db_folder_path, exist_ok=True)
                
                sqlite_db_path = os.path.join(db_folder_path, folder_name + '.db')
                
                # CSV to SQLite
                db_structure = csv_to_sqlite(folder_path, sqlite_db_path)
                
                # database structure and data
                save_db_structure_and_data_as_json(db_structure, db_folder_path, folder_name)
                print(f"Database created and structure/data saved for {folder_name}")
            except Exception as e:
                print(f"Error processing folder {folder_path}: {e}")
                continue

parsed_data_dir = '../../data/parsed_data'
database_dir = '../../data/database'

os.makedirs(database_dir, exist_ok=True)

process_parsed_data_to_sqlite(parsed_data_dir, database_dir)