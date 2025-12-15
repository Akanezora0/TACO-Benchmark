"""
SQL Generation Prompt策略实现

根据文档中的设计，实现SQL生成的Prompt构建逻辑
包括Baseline Prompt和TACO-SQL Prompt两种策略
"""

from typing import Dict, List, Optional, Tuple


class SQLGenerationPrompt:
    """SQL Generation的Prompt构建器"""
    
    def __init__(
        self, 
        temperature: float = 0.1, 
        max_tokens: int = 2000,
        use_filtered_schema: bool = False
    ):
        """
        初始化Prompt构建器
        
        Args:
            temperature: 温度参数（默认0.1，保证SQL准确性）
            max_tokens: 最大输出token数（默认2000）
            use_filtered_schema: 是否使用过滤后的Schema（TACO-SQL模式）
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
        构建Baseline Prompt（原始设置）
        
        适用场景：Origin实验设置（原始查询 + 完整Schema）
        
        Args:
            query: 自然语言查询
            schema_text: 完整Schema文本
            database: 数据库名称
            
        Returns:
            格式化的Prompt字符串
        """
        prompt = f"""你是一个SQL专家。根据自然语言查询和数据库Schema，生成对应的SQL查询语句。

{schema_text}

自然语言查询：{query}

要求：
1. 生成完整、可执行的SQL语句
2. 所有表名和列名必须用双引号包裹（包括中文和特殊字符）
3. 确保SQL语法正确，可以在SQLite上执行
4. 只输出SQL语句，不要添加任何解释或注释

数据库：{database}

SQL查询："""
        
        return prompt
    
    def build_taco_sql_prompt(
        self, 
        rewritten_query: str, 
        filtered_schema_text: str, 
        database: str
    ) -> str:
        """
        构建TACO-SQL Prompt（使用Table Linking后）
        
        适用场景：QR+TL和QR+TL+QP实验设置
        
        Args:
            rewritten_query: Question Rewriting后的查询
            filtered_schema_text: 过滤后的Schema文本（仅包含相关表）
            database: 数据库名称
            
        Returns:
            格式化的Prompt字符串
        """
        prompt = f"""你是一个SQL专家。根据自然语言查询和相关数据库Schema，生成对应的SQL查询语句。

相关表Schema信息：
{filtered_schema_text}

自然语言查询：{rewritten_query}

要求：
1. 生成完整、可执行的SQL语句
2. 所有表名和列名必须用双引号包裹（包括中文和特殊字符）
3. 确保SQL语法正确，可以在SQLite上执行
4. 只输出SQL语句，不要添加任何解释或注释
5. 仅使用上述相关表，不要使用未列出的表

数据库：{database}

SQL查询："""
        
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
        根据设置自动选择Prompt策略
        
        Args:
            query: 原始查询
            schema_text: Schema文本
            database: 数据库名称
            rewritten_query: 转写后的查询（如果使用）
            is_filtered: Schema是否已过滤
            
        Returns:
            格式化的Prompt字符串
        """
        if is_filtered and rewritten_query:
            # 使用TACO-SQL Prompt
            return self.build_taco_sql_prompt(rewritten_query, schema_text, database)
        else:
            # 使用Baseline Prompt
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
        构建OpenAI格式的消息列表
        
        Args:
            query: 原始查询
            schema_text: Schema文本
            database: 数据库名称
            rewritten_query: 转写后的查询
            is_filtered: Schema是否已过滤
            
        Returns:
            消息列表
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
        获取模型调用配置
        
        Returns:
            配置字典
        """
        return {
            "temperature": self.temperature,
            "max_tokens": self.max_tokens
        }
    
    def clean_sql(self, sql: str) -> str:
        """
        清理生成的SQL（移除代码块标记等）
        
        Args:
            sql: 原始SQL字符串
            
        Returns:
            清理后的SQL
        """
        sql = sql.strip()
        
        # 移除代码块标记
        if sql.startswith('```'):
            lines = sql.split('\n')
            sql = '\n'.join(lines[1:-1]) if len(lines) > 2 else sql
        
        # 确保以分号结尾
        sql = sql.strip().rstrip(';') + ';'
        
        return sql


def create_sql_prompt_builder(
    temperature: float = 0.1,
    max_tokens: int = 2000,
    use_filtered_schema: bool = False
) -> SQLGenerationPrompt:
    """
    创建SQL Generation Prompt构建器
    
    Args:
        temperature: 温度参数
        max_tokens: 最大输出token数
        use_filtered_schema: 是否使用过滤后的Schema
        
    Returns:
        SQLGenerationPrompt实例
    """
    return SQLGenerationPrompt(
        temperature=temperature,
        max_tokens=max_tokens,
        use_filtered_schema=use_filtered_schema
    )


# 示例使用
if __name__ == "__main__":
    # 创建Prompt构建器
    prompt_builder = create_sql_prompt_builder()
    
    # 示例Schema文本
    example_schema = """数据库Schema信息：

表：企业注册表
  列：
    - 企业名称 (TEXT)
    - 注册时间 (DATE)
    - 注册资本 (INTEGER)
    - 注册地址 (TEXT)

表：企业信息表
  列：
    - 企业名称 (TEXT)
    - 行业类型 (TEXT)
    - 员工数量 (INTEGER)
"""
    
    # 构建Baseline Prompt
    baseline_prompt = prompt_builder.build_baseline_prompt(
        query="查询北京地区企业注册情况",
        schema_text=example_schema,
        database="企业数据库"
    )
    print("Baseline Prompt：")
    print(baseline_prompt)
    print("\n" + "="*80 + "\n")
    
    # 构建TACO-SQL Prompt
    filtered_schema = """相关表Schema信息：

表：企业注册表
  列：
    - 企业名称 (TEXT)
    - 注册时间 (DATE)
    - 注册资本 (INTEGER)
"""
    
    taco_sql_prompt = prompt_builder.build_taco_sql_prompt(
        rewritten_query="查询北京地区企业注册数据：注册数量、注册资本，按年份统计",
        filtered_schema_text=filtered_schema,
        database="企业数据库"
    )
    print("TACO-SQL Prompt：")
    print(taco_sql_prompt)
    
    # 测试SQL清理
    example_sql = """```sql
SELECT "企业名称", "注册资本" FROM "企业注册表" WHERE "注册地址" LIKE '%北京%';
```"""
    cleaned = prompt_builder.clean_sql(example_sql)
    print("\n清理后的SQL：")
    print(cleaned)

