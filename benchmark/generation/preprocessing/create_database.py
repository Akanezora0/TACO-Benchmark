import os
import sqlite3
import pandas as pd
import json
from pathlib import Path

_BENCHMARK_DATA = Path(__file__).resolve().parents[2] / "data"

# Convert CSV files to SQLite tables
def csv_to_sqlite(csv_folder_path, sqlite_db_path):
    # Create SQLite database connection
    conn = sqlite3.connect(sqlite_db_path)
    cursor = conn.cursor()

    # Store database structure and table data
    db_structure = {}

    # Iterate CSV files and convert each to a SQLite table
    for file_name in os.listdir(csv_folder_path):
        if file_name.endswith('.csv'):
            csv_path = os.path.join(csv_folder_path, file_name)
            try:
                # Read CSV file
                df = pd.read_csv(csv_path)
                
                # Replace NaN with empty strings
                df = df.fillna('')

                # Get table name (remove .csv suffix)
                table_name = file_name.replace('.csv', '')
                
                # Write DataFrame to SQLite database
                df.to_sql(table_name, conn, if_exists='replace', index=False)
                print(f"Added table: {table_name} from {csv_path}")
                
                # Get column names and store in db_structure
                db_structure[table_name] = {
                    'columns': df.columns.tolist(),
                    'data': df.to_dict(orient='records')  # convert table data to dict form
                }
            except Exception as e:
                # On error, skip file and print error message
                print(f"Error processing file {csv_path}: {e}")
                continue  # skip current file and continue with next

    # Commit and close database connection
    conn.commit()
    conn.close()

    # Return database structure (including data)
    return db_structure

# Save database structure and data as JSON file
def save_db_structure_and_data_as_json(db_structure, db_folder_path, db_name):
    # JSON file path using database name with .json suffix
    json_file_path = os.path.join(db_folder_path, f"{db_name}.json")
    
    try:
        # Write structure and data to JSON file
        with open(json_file_path, 'w', encoding='utf-8') as json_file:
            json.dump(db_structure, json_file, ensure_ascii=False, indent=4)
        print(f"Database structure and data saved as JSON: {json_file_path}")
    except Exception as e:
        # On JSON save error, print error message
        print(f"Error saving JSON file {json_file_path}: {e}")

# Process all folders in parsed_data and generate SQLite databases plus JSON structure/data
def process_parsed_data_to_sqlite(parsed_data_dir, database_dir):
    # Iterate all folders under parsed_data/
    for folder_name in os.listdir(parsed_data_dir):
        folder_path = os.path.join(parsed_data_dir, folder_name)
        if os.path.isdir(folder_path):
            try:
                # Create a SQLite database for each folder
                db_folder_path = os.path.join(database_dir, folder_name)
                os.makedirs(db_folder_path, exist_ok=True)
                
                sqlite_db_path = os.path.join(db_folder_path, folder_name + '.db')
                
                # Convert all CSV files in folder to SQLite tables and return structure/data
                db_structure = csv_to_sqlite(folder_path, sqlite_db_path)
                
                # Save database structure and data as JSON using database name
                save_db_structure_and_data_as_json(db_structure, db_folder_path, folder_name)
                print(f"Database created and structure/data saved for {folder_name}")
            except Exception as e:
                # On folder processing error, print message and continue
                print(f"Error processing folder {folder_path}: {e}")
                continue  # skip current folder and continue with next

# Define directories
parsed_data_dir = str(_BENCHMARK_DATA / "parsed_data")
database_dir = str(_BENCHMARK_DATA / "database")

# Create directory for SQLite databases
os.makedirs(database_dir, exist_ok=True)

# Process all folders in parsed_data and generate SQLite databases plus structure/data
process_parsed_data_to_sqlite(parsed_data_dir, database_dir)
