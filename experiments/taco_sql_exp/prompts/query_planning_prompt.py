"""
Query Planning Prompt strategy implementation

Implements query planning prompt construction logic based on the design document
"""

from typing import List, Dict, Optional
import json


class QueryPlanningPrompt:
    """Query Planning prompt builder"""
    
    def __init__(self, temperature: float = 0.3, max_tokens: int = 1024):
        """
        Initialize prompt builder
        
        Args:
            temperature: Temperature parameter (default 0.3 for planning stability)
            max_tokens: Maximum output tokens (default 1024)
        """
        self.temperature = temperature
        self.max_tokens = max_tokens
    
    def build_prompt(
        self, 
        query: str, 
        relevant_tables: List[str], 
        schema_info: Optional[Dict] = None
    ) -> str:
        """
        Build query planning prompt
        
        Args:
            query: Rewritten query
            relevant_tables: List of relevant tables (from Table Linking)
            schema_info: Schema information (optional)
            
        Returns:
            Formatted prompt string
        """
        prompt = f"""Break down the following query into multiple simple subqueries and determine the execution order.

Original query: {query}

Relevant tables: {', '.join(relevant_tables)}

"""
        
        if schema_info:
            prompt += f"Schema information: {json.dumps(schema_info, ensure_ascii=False, indent=2)}\n\n"
        
        prompt += """Output the execution plan in JSON format as follows:
[
    {
        "subquery": "subquery description",
        "tables": ["table1", "table2"],
        "order": 1,
        "dependencies": []
    },
    ...
]

Execution plan:"""
        
        return prompt
    
    def build_messages(
        self, 
        query: str, 
        relevant_tables: List[str], 
        schema_info: Optional[Dict] = None
    ) -> List[Dict[str, str]]:
        """
        Build OpenAI-format message list
        
        Args:
            query: Rewritten query
            relevant_tables: List of relevant tables
            schema_info: Schema information
            
        Returns:
            Message list
        """
        prompt = self.build_prompt(query, relevant_tables, schema_info)
        
        messages = [
            {
                "role": "system",
                "content": "You are a query planning expert. Break down complex queries into simple subqueries and determine execution order."
            },
            {
                "role": "user",
                "content": prompt
            }
        ]
        
        return messages
    
    def parse_plan(self, plan_json: str) -> List[Dict]:
        """
        Parse execution plan JSON
        
        Args:
            plan_json: JSON-formatted execution plan string
            
        Returns:
            Parsed execution plan list
        """
        try:
            plan = json.loads(plan_json)
            # Validate plan format
            if not isinstance(plan, list):
                return self._default_plan()
            
            # Ensure each plan item has required fields
            for item in plan:
                if not all(key in item for key in ['subquery', 'tables', 'order']):
                    return self._default_plan()
            
            return plan
        except json.JSONDecodeError:
            # If parsing fails, return default plan
            return self._default_plan()
    
    def _default_plan(self, query: str = "", tables: List[str] = None) -> List[Dict]:
        """
        Generate default plan (single-step execution)
        
        Args:
            query: Query text
            tables: Table list
            
        Returns:
            Default execution plan
        """
        return [{
            "subquery": query or "Original query",
            "tables": tables or [],
            "order": 1,
            "dependencies": []
        }]
    
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


def create_planning_prompt_builder(
    temperature: float = 0.3, 
    max_tokens: int = 1024
) -> QueryPlanningPrompt:
    """
    Create Query Planning prompt builder
    
    Args:
        temperature: Temperature parameter
        max_tokens: Maximum output tokens
        
    Returns:
        QueryPlanningPrompt instance
    """
    return QueryPlanningPrompt(temperature=temperature, max_tokens=max_tokens)


# Example usage
if __name__ == "__main__":
    # Create prompt builder
    prompt_builder = create_planning_prompt_builder()
    
    # Example data
    example_query = "查询北京地区企业注册数据：注册数量、注册资本，按年份统计"
    example_tables = ["企业注册表", "企业信息表"]
    
    # Build prompt
    prompt = prompt_builder.build_prompt(example_query, example_tables)
    print("Query Planning Prompt:")
    print(prompt)
    
    # Build OpenAI-format messages
    messages = prompt_builder.build_messages(example_query, example_tables)
    print("\nOpenAI-format messages:")
    for msg in messages:
        print(f"{msg['role']}: {msg['content'][:200]}...")
    
    # Parse example plan
    example_plan_json = """[
        {
            "subquery": "查询企业注册基本信息",
            "tables": ["企业注册表"],
            "order": 1,
            "dependencies": []
        },
        {
            "subquery": "按年份统计注册数量和注册资本",
            "tables": ["企业注册表"],
            "order": 2,
            "dependencies": [1]
        }
    ]"""
    
    plan = prompt_builder.parse_plan(example_plan_json)
    print("\nParsed execution plan:")
    print(json.dumps(plan, ensure_ascii=False, indent=2))
