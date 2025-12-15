import os
import json
from statistics import mean

# Get a list of all JSON files in the current directory
json_files = [file for file in os.listdir('.') if file.endswith('.json')]

# Initialize lists to store token counts
sql_token_lengths = []
nl_query_token_lengths = []

# Function to count tokens in a string (split by whitespace as an example)
def count_tokens(text):
    return len(text.split())

# Process each JSON file
for file in json_files:
    with open(file, 'r', encoding='utf-8') as f:
        data = json.load(f)
        # Assuming each file has 'sql' and 'natural_language_query' fields
        sql_tokens = count_tokens(data.get('sql', ''))
        nl_query_tokens = count_tokens(data.get('natural_language_query', ''))
        sql_token_lengths.append(sql_tokens)
        nl_query_token_lengths.append(nl_query_tokens)

# Calculate average token lengths
average_sql_tokens = mean(sql_token_lengths) if sql_token_lengths else 0
average_nl_query_tokens = mean(nl_query_token_lengths) if nl_query_token_lengths else 0

average_sql_tokens, average_nl_query_tokens
