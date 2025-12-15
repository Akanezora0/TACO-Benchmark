"""
Standardized Prompt Strategy for Base LLM Experiments

Ensures fair comparison by using the same prompt template across all base LLM models.
"""

from typing import Dict, Optional


class BaseLLMPromptStrategy:
    """Standardized prompt strategy for base LLM baseline experiments"""
    
    @staticmethod
    def build_baseline_prompt(
        query: str,
        schema_text: str,
        database: str
    ) -> str:
        """
        Build standardized baseline prompt for all base LLM models
        
        This prompt template is used consistently across all base LLM models
        to ensure fair comparison.
        
        Args:
            query: Natural language query
            schema_text: Formatted schema text
            database: Database name
            
        Returns:
            Complete prompt string
        """
        prompt = f"""You are a SQL expert. Generate SQL queries based on natural language queries and database schema.

{schema_text}

Natural language query: {query}

Requirements:
1. Generate complete, executable SQL statements
2. All table and column names must be wrapped in double quotes (including Chinese and special characters)
3. Ensure SQL syntax is correct and can be executed on SQLite
4. Output only SQL statements, no explanations or comments

Database: {database}

SQL query:"""
        
        return prompt
    
    @staticmethod
    def build_messages(
        query: str,
        schema_text: str,
        database: str,
        system_message: Optional[str] = None
    ) -> list:
        """
        Build OpenAI API format messages
        
        Args:
            query: Natural language query
            schema_text: Formatted schema text
            database: Database name
            system_message: Optional custom system message
            
        Returns:
            List of message dictionaries in OpenAI format
        """
        if system_message is None:
            system_message = "You are a SQL expert."
        
        prompt = BaseLLMPromptStrategy.build_baseline_prompt(
            query, schema_text, database
        )
        
        messages = [
            {"role": "system", "content": system_message},
            {"role": "user", "content": prompt}
        ]
        
        return messages
    
    @staticmethod
    def clean_sql_output(sql: str) -> str:
        """
        Clean SQL output to ensure consistent format
        
        Removes code block markers, ensures semicolon at end,
        and standardizes whitespace.
        
        Args:
            sql: Raw SQL string from model
            
        Returns:
            Cleaned SQL string
        """
        sql = sql.strip()
        
        # Remove code block markers (```sql, ```, etc.)
        if sql.startswith('```'):
            lines = sql.split('\n')
            # Remove first line (```sql or ```) and last line (```)
            if len(lines) > 2:
                sql = '\n'.join(lines[1:-1])
            elif len(lines) == 2:
                sql = lines[1] if lines[1].strip() else lines[0]
        
        # Remove any remaining markdown code block markers
        sql = sql.strip().lstrip('```').rstrip('```').strip()
        
        # Ensure semicolon at the end
        sql = sql.rstrip(';').strip() + ';'
        
        return sql
    
    @staticmethod
    def get_standard_config() -> Dict:
        """
        Get standard configuration for base LLM experiments
        
        These parameters are used consistently across all base LLM models
        to ensure fair comparison.
        
        Returns:
            Configuration dictionary
        """
        return {
            "temperature": 0.1,  # Low temperature for reproducibility
            "max_tokens": 2000,  # Sufficient for most SQL queries
        }


def create_baseline_prompt(
    query: str,
    schema_text: str,
    database: str
) -> str:
    """
    Convenience function to create baseline prompt
    
    Args:
        query: Natural language query
        schema_text: Formatted schema text
        database: Database name
        
    Returns:
        Complete prompt string
    """
    return BaseLLMPromptStrategy.build_baseline_prompt(query, schema_text, database)


# Example usage
if __name__ == "__main__":
    # Example schema
    example_schema = """Database Schema Information:

Table: enterprise_registration
  Columns:
    - enterprise_name (TEXT)
    - registration_date (DATE)
    - registered_capital (INTEGER)
    - registration_address (TEXT)

Table: enterprise_info
  Columns:
    - enterprise_name (TEXT)
    - industry_type (TEXT)
    - employee_count (INTEGER)
"""
    
    # Example query
    example_query = "Query enterprise registration data in Beijing: registration count and registered capital, grouped by year"
    
    # Build prompt
    prompt = BaseLLMPromptStrategy.build_baseline_prompt(
        query=example_query,
        schema_text=example_schema,
        database="enterprise_db"
    )
    
    print("Standardized Baseline Prompt:")
    print("=" * 80)
    print(prompt)
    print("=" * 80)
    
    # Build messages for API
    messages = BaseLLMPromptStrategy.build_messages(
        query=example_query,
        schema_text=example_schema,
        database="enterprise_db"
    )
    
    print("\nOpenAI API Format Messages:")
    for msg in messages:
        print(f"{msg['role']}: {msg['content'][:100]}...")
    
    # Test SQL cleaning
    example_sql = """```sql
SELECT "enterprise_name", "registered_capital" 
FROM "enterprise_registration" 
WHERE "registration_address" LIKE '%Beijing%';
```"""
    
    cleaned = BaseLLMPromptStrategy.clean_sql_output(example_sql)
    print(f"\nCleaned SQL:\n{cleaned}")
    
    # Get standard config
    config = BaseLLMPromptStrategy.get_standard_config()
    print(f"\nStandard Configuration:\n{config}")

