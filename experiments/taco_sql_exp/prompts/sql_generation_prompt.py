"""
SQL Generation Prompt strategy implementation

Implements SQL generation prompt construction logic based on the design document
Includes Baseline Prompt and TACO-SQL Prompt strategies
"""

from typing import Dict, List, Optional, Tuple


class SQLGenerationPrompt:
    """SQL Generation prompt builder"""
    
    def __init__(
        self, 
        temperature: float = 0.1, 
        max_tokens: int = 2000,
        use_filtered_schema: bool = False
    ):
        """
        Initialize prompt builder
        
        Args:
            temperature: Temperature parameter (default 0.1 for SQL accuracy)
            max_tokens: Maximum output tokens (default 2000)
            use_filtered_schema: Whether to use filtered schema (TACO-SQL mode)
        """
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.use_filtered_schema = use_filtered_schema
    
    def build_baseline_prompt(
        self, 
        query: str, 
        schema_text: str, 
        database: str
    ) -> str:
        """
        Build Baseline Prompt (original setting)
        
        Use case: Origin experiment setting (original query + full schema)
        
        Args:
            query: Natural language query
            schema_text: Full schema text
            database: Database name
            
        Returns:
            Formatted prompt string
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
    
    def build_taco_sql_prompt(
        self, 
        rewritten_query: str, 
        filtered_schema_text: str, 
        database: str
    ) -> str:
        """
        Build TACO-SQL Prompt (after Table Linking)
        
        Use case: QR+TL and QR+TL+QP experiment settings
        
        Args:
            rewritten_query: Query after Question Rewriting
            filtered_schema_text: Filtered schema text (relevant tables only)
            database: Database name
            
        Returns:
            Formatted prompt string
        """
        prompt = f"""You are a SQL expert. Generate SQL queries based on natural language queries and relevant database schema.

Relevant Tables Schema Information:
{filtered_schema_text}

Natural language query: {rewritten_query}

Requirements:
1. Generate complete, executable SQL statements
2. All table and column names must be wrapped in double quotes (including Chinese and special characters)
3. Ensure SQL syntax is correct and can be executed on SQLite
4. Output only SQL statements, no explanations or comments
5. Use only the relevant tables listed above; do not use unlisted tables

Database: {database}

SQL query:"""
        
        return prompt
    
    def build_prompt(
        self, 
        query: str, 
        schema_text: str, 
        database: str,
        rewritten_query: Optional[str] = None,
        is_filtered: bool = False
    ) -> str:
        """
        Automatically select prompt strategy based on setting
        
        Args:
            query: Original query
            schema_text: Schema text
            database: Database name
            rewritten_query: Rewritten query (if used)
            is_filtered: Whether schema has been filtered
            
        Returns:
            Formatted prompt string
        """
        if is_filtered and rewritten_query:
            # Use TACO-SQL Prompt
            return self.build_taco_sql_prompt(rewritten_query, schema_text, database)
        else:
            # Use Baseline Prompt
            return self.build_baseline_prompt(query, schema_text, database)
    
    def build_messages(
        self, 
        query: str, 
        schema_text: str, 
        database: str,
        rewritten_query: Optional[str] = None,
        is_filtered: bool = False
    ) -> List[Dict[str, str]]:
        """
        Build OpenAI-format message list
        
        Args:
            query: Original query
            schema_text: Schema text
            database: Database name
            rewritten_query: Rewritten query
            is_filtered: Whether schema has been filtered
            
        Returns:
            Message list
        """
        prompt = self.build_prompt(query, schema_text, database, rewritten_query, is_filtered)
        
        messages = [
            {
                "role": "system",
                "content": "You are a SQL expert."
            },
            {
                "role": "user",
                "content": prompt
            }
        ]
        
        return messages
    
    def get_config(self) -> Dict:
        """
        Get model call configuration
        
        Returns:
            Configuration dictionary
        """
        return {
            "temperature": self.temperature,
            "max_tokens": self.max_tokens
        }
    
    def clean_sql(self, sql: str) -> str:
        """
        Clean generated SQL (remove code block markers, etc.)
        
        Args:
            sql: Raw SQL string
            
        Returns:
            Cleaned SQL
        """
        sql = sql.strip()
        
        # Remove code block markers
        if sql.startswith('```'):
            lines = sql.split('\n')
            sql = '\n'.join(lines[1:-1]) if len(lines) > 2 else sql
        
        # Ensure trailing semicolon
        sql = sql.strip().rstrip(';') + ';'
        
        return sql


def create_sql_prompt_builder(
    temperature: float = 0.1,
    max_tokens: int = 2000,
    use_filtered_schema: bool = False
) -> SQLGenerationPrompt:
    """
    Create SQL Generation prompt builder
    
    Args:
        temperature: Temperature parameter
        max_tokens: Maximum output tokens
        use_filtered_schema: Whether to use filtered schema
        
    Returns:
        SQLGenerationPrompt instance
    """
    return SQLGenerationPrompt(
        temperature=temperature,
        max_tokens=max_tokens,
        use_filtered_schema=use_filtered_schema
    )


# Example usage
if __name__ == "__main__":
    # Create prompt builder
    prompt_builder = create_sql_prompt_builder()
    
    # Example schema text
    example_schema = """Database Schema Information:

Table: 企业注册表
  Columns:
    - 企业名称 (TEXT)
    - 注册时间 (DATE)
    - 注册资本 (INTEGER)
    - 注册地址 (TEXT)

Table: 企业信息表
  Columns:
    - 企业名称 (TEXT)
    - 行业类型 (TEXT)
    - 员工数量 (INTEGER)
"""
    
    # Build Baseline Prompt
    baseline_prompt = prompt_builder.build_baseline_prompt(
        query="查询北京地区企业注册情况",
        schema_text=example_schema,
        database="企业数据库"
    )
    print("Baseline Prompt:")
    print(baseline_prompt)
    print("\n" + "="*80 + "\n")
    
    # Build TACO-SQL Prompt
    filtered_schema = """Relevant Tables Schema Information:

Table: 企业注册表
  Columns:
    - 企业名称 (TEXT)
    - 注册时间 (DATE)
    - 注册资本 (INTEGER)
"""
    
    taco_sql_prompt = prompt_builder.build_taco_sql_prompt(
        rewritten_query="查询北京地区企业注册数据：注册数量、注册资本，按年份统计",
        filtered_schema_text=filtered_schema,
        database="企业数据库"
    )
    print("TACO-SQL Prompt:")
    print(taco_sql_prompt)
    
    # Test SQL cleaning
    example_sql = """```sql
SELECT "企业名称", "注册资本" FROM "企业注册表" WHERE "注册地址" LIKE '%北京%';
```"""
    cleaned = prompt_builder.clean_sql(example_sql)
    print("\nCleaned SQL:")
    print(cleaned)
