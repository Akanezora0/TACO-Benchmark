"""
Query Planning Prompt策略实现

根据文档中的设计，实现查询规划的Prompt构建逻辑
"""

from typing import List, Dict, Optional
import json


class QueryPlanningPrompt:
    """Query Planning的Prompt构建器"""
    
    def __init__(self, temperature: float = 0.3, max_tokens: int = 1024):
        """
        初始化Prompt构建器
        
        Args:
            temperature: 温度参数（默认0.3，保证规划稳定性）
            max_tokens: 最大输出token数（默认1024）
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
        构建查询规划Prompt
        
        Args:
            query: 转写后的查询
            relevant_tables: 相关表列表（来自Table Linking）
            schema_info: Schema信息（可选）
            
        Returns:
            格式化的Prompt字符串
        """
        prompt = f"""请将以下查询拆解为多个简单的子查询，并确定执行顺序。

原始查询：{query}

相关表：{', '.join(relevant_tables)}

"""
        
        if schema_info:
            prompt += f"Schema信息：{json.dumps(schema_info, ensure_ascii=False, indent=2)}\n\n"
        
        prompt += """请以JSON格式输出执行计划，格式如下：
[
    {
        "subquery": "子查询描述",
        "tables": ["表1", "表2"],
        "order": 1,
        "dependencies": []
    },
    ...
]

执行计划："""
        
        return prompt
    
    def build_messages(
        self, 
        query: str, 
        relevant_tables: List[str], 
        schema_info: Optional[Dict] = None
    ) -> List[Dict[str, str]]:
        """
        构建OpenAI格式的消息列表
        
        Args:
            query: 转写后的查询
            relevant_tables: 相关表列表
            schema_info: Schema信息
            
        Returns:
            消息列表
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
        解析执行计划JSON
        
        Args:
            plan_json: JSON格式的执行计划字符串
            
        Returns:
            解析后的执行计划列表
        """
        try:
            plan = json.loads(plan_json)
            # 验证计划格式
            if not isinstance(plan, list):
                return self._default_plan()
            
            # 确保每个计划项都有必要字段
            for item in plan:
                if not all(key in item for key in ['subquery', 'tables', 'order']):
                    return self._default_plan()
            
            return plan
        except json.JSONDecodeError:
            # 如果解析失败，返回默认计划
            return self._default_plan()
    
    def _default_plan(self, query: str = "", tables: List[str] = None) -> List[Dict]:
        """
        生成默认计划（单步执行）
        
        Args:
            query: 查询文本
            tables: 表列表
            
        Returns:
            默认执行计划
        """
        return [{
            "subquery": query or "原始查询",
            "tables": tables or [],
            "order": 1,
            "dependencies": []
        }]
    
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


def create_planning_prompt_builder(
    temperature: float = 0.3, 
    max_tokens: int = 1024
) -> QueryPlanningPrompt:
    """
    创建Query Planning Prompt构建器
    
    Args:
        temperature: 温度参数
        max_tokens: 最大输出token数
        
    Returns:
        QueryPlanningPrompt实例
    """
    return QueryPlanningPrompt(temperature=temperature, max_tokens=max_tokens)


# 示例使用
if __name__ == "__main__":
    # 创建Prompt构建器
    prompt_builder = create_planning_prompt_builder()
    
    # 示例数据
    example_query = "查询北京地区企业注册数据：注册数量、注册资本，按年份统计"
    example_tables = ["企业注册表", "企业信息表"]
    
    # 构建Prompt
    prompt = prompt_builder.build_prompt(example_query, example_tables)
    print("Query Planning Prompt：")
    print(prompt)
    
    # 构建OpenAI格式消息
    messages = prompt_builder.build_messages(example_query, example_tables)
    print("\nOpenAI格式消息：")
    for msg in messages:
        print(f"{msg['role']}: {msg['content'][:200]}...")
    
    # 解析示例计划
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
    print("\n解析后的执行计划：")
    print(json.dumps(plan, ensure_ascii=False, indent=2))

