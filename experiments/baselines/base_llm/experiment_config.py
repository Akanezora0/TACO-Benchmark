"""
Baseline Experiment Configuration

Ensures fair comparison across different models by standardizing:
1. Prompt templates
2. Schema formatting
3. Model parameters
4. Evaluation procedures
"""

from typing import Dict, Optional
from dataclasses import dataclass


@dataclass
class BaselineModelConfig:
    """Configuration for baseline model experiments"""
    model_name: str
    model_type: str  # "base_llm", "sft_based", "llm_based", "hybrid"
    
    # API configuration (for base LLMs)
    api_key: Optional[str] = None
    base_url: Optional[str] = None
    
    # Generation parameters (standardized for fair comparison)
    temperature: float = 0.1  # Low temperature for reproducibility
    max_tokens: int = 2000
    
    # Context window
    context_window: int = 128000
    
    # SFT model specific
    model_path: Optional[str] = None
    device: Optional[str] = None
    num_beams: int = 4
    
    # Schema filtering (for SFT models)
    schema_filter_enabled: bool = False
    schema_filter_model_path: Optional[str] = None
    max_tables: int = 7
    max_columns: int = 20


# Standardized model configurations for fair comparison
BASELINE_MODEL_CONFIGS: Dict[str, BaselineModelConfig] = {
    "gpt-4o": BaselineModelConfig(
        model_name="gpt-4o",
        model_type="base_llm",
        temperature=0.1,
        max_tokens=2000,
        context_window=128000
    ),
    "gpt-4o-mini": BaselineModelConfig(
        model_name="gpt-4o-mini",
        model_type="base_llm",
        temperature=0.1,
        max_tokens=2000,
        context_window=128000
    ),
    "gpt-o1": BaselineModelConfig(
        model_name="o1",
        model_type="base_llm",
        temperature=0.1,
        max_tokens=2000,
        context_window=200000
    ),
    "deepseek-r1": BaselineModelConfig(
        model_name="deepseek-r1",
        model_type="base_llm",
        temperature=0.1,
        max_tokens=2000,
        context_window=64000
    ),
    "codes-33b": BaselineModelConfig(
        model_name="codes-33b",
        model_type="sft_based",
        model_path="models/codes-33b",
        device="cuda:0",
        temperature=0.1,
        max_tokens=2000,
        num_beams=4,
        schema_filter_enabled=True,
        schema_filter_model_path="models/sic_ckpts/sic_spider",
        max_tables=7,
        max_columns=20
    ),
}


def get_baseline_prompt_template() -> str:
    """
    Get standardized baseline prompt template
    
    This template is used for all models to ensure fair comparison.
    """
    return """You are a SQL expert. Generate SQL queries based on natural language queries and database schema.

{schema_text}

Natural language query: {query}

Requirements:
1. Generate complete, executable SQL statements
2. All table and column names must be wrapped in double quotes (including Chinese and special characters)
3. Ensure SQL syntax is correct and can be executed on SQLite
4. Output only SQL statements, no explanations or comments

Database: {database}

SQL query:"""


def format_schema_for_baseline(
    schema: Dict,
    max_tables: Optional[int] = None,
    max_columns_per_table: Optional[int] = None
) -> str:
    """
    Format schema for baseline experiments
    
    Args:
        schema: Schema dictionary
        max_tables: Maximum number of tables (None = all tables)
        max_columns_per_table: Maximum columns per table (None = all columns)
        
    Returns:
        Formatted schema text
    """
    all_tables = schema.get('tables', [])
    
    if max_tables is None:
        selected_tables = all_tables
    else:
        selected_tables = all_tables[:max_tables]
    
    text = "Database Schema Information:\n\n"
    
    for table in selected_tables:
        table_name = table.get('table_name', '')
        columns = table.get('columns', [])
        
        if max_columns_per_table is not None:
            columns = columns[:max_columns_per_table]
        
        text += f"Table: {table_name}\n"
        text += "  Columns:\n"
        for col in columns:
            col_name = col.get('column_name', '')
            col_type = col.get('data_type', 'TEXT')
            text += f"    - {col_name} ({col_type})\n"
        text += "\n"
    
    return text


def clean_sql_output(sql: str) -> str:
    """
    Clean SQL output to ensure consistent format
    
    Args:
        sql: Raw SQL string from model
        
    Returns:
        Cleaned SQL string
    """
    sql = sql.strip()
    
    # Remove code block markers
    if sql.startswith('```'):
        lines = sql.split('\n')
        sql = '\n'.join(lines[1:-1]) if len(lines) > 2 else sql
    
    # Ensure semicolon at the end
    sql = sql.strip().rstrip(';') + ';'
    
    return sql

