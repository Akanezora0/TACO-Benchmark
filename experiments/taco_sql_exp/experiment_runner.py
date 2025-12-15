"""
TACO-SQL实验运行器

根据实验设置运行不同的实验配置
"""

from typing import Dict, List, Optional, Any
import json
import os
from pathlib import Path

from .config import ExperimentConfig, create_experiment_config
from .prompts import (
    create_rewriting_prompt_builder,
    create_planning_prompt_builder,
    create_sql_prompt_builder
)
from .utils import format_schema_simple, format_schema_filtered


class ExperimentRunner:
    """实验运行器"""
    
    def __init__(self, config: ExperimentConfig):
        """
        初始化实验运行器
        
        Args:
            config: 实验配置
        """
        self.config = config
        
        # 初始化Prompt构建器
        self.qr_prompt_builder = None
        self.qp_prompt_builder = None
        self.sql_prompt_builder = None
        
        if config.component_config.qr_enabled:
            self.qr_prompt_builder = create_rewriting_prompt_builder(
                temperature=config.component_config.qr_temperature,
                top_p=config.component_config.qr_top_p
            )
        
        if config.component_config.qp_enabled:
            self.qp_prompt_builder = create_planning_prompt_builder(
                temperature=config.component_config.qp_temperature,
                max_tokens=config.component_config.qp_max_tokens
            )
        
        self.sql_prompt_builder = create_sql_prompt_builder(
            temperature=config.component_config.sg_temperature,
            max_tokens=config.component_config.sg_max_tokens,
            use_filtered_schema=config.component_config.tl_enabled
        )
    
    def run_single_query(
        self,
        query: str,
        schema: Dict,
        database: str,
        ground_truth_sql: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        运行单个查询的实验
        
        Args:
            query: 原始查询
            schema: Schema字典
            database: 数据库名称
            ground_truth_sql: 标准答案SQL（可选）
            
        Returns:
            实验结果字典
        """
        result = {
            'original_query': query,
            'rewritten_query': None,
            'relevant_tables': None,
            'execution_plan': None,
            'generated_sql': None,
            'schema_info': {},
            'errors': []
        }
        
        try:
            # Step 1: Question Rewriting
            if self.config.component_config.qr_enabled and self.qr_prompt_builder:
                rewritten_query = self._rewrite_query(query)
                result['rewritten_query'] = rewritten_query
            else:
                result['rewritten_query'] = query
            
            # Step 2: Table Linking
            if self.config.component_config.tl_enabled:
                relevant_tables = self._retrieve_tables(result['rewritten_query'])
                result['relevant_tables'] = relevant_tables
                
                # 格式化过滤后的Schema
                schema_text, schema_info = format_schema_filtered(
                    schema,
                    relevant_tables
                )
            else:
                result['relevant_tables'] = []
                
                # 格式化完整Schema
                schema_text, schema_info = format_schema_simple(schema)
            
            result['schema_info'] = schema_info
            
            # Step 3: Query Planning
            if self.config.component_config.qp_enabled and self.qp_prompt_builder:
                execution_plan = self._plan_query(
                    result['rewritten_query'],
                    result['relevant_tables'],
                    schema
                )
                result['execution_plan'] = execution_plan
            else:
                # 默认单步计划
                result['execution_plan'] = [{
                    'subquery': result['rewritten_query'],
                    'tables': result['relevant_tables'],
                    'order': 1,
                    'dependencies': []
                }]
            
            # Step 4: SQL Generation
            if result['execution_plan']:
                # 使用第一个子查询生成SQL（简化实现）
                plan_item = result['execution_plan'][0]
                generated_sql = self._generate_sql(
                    query=result['rewritten_query'],
                    schema_text=schema_text,
                    database=database,
                    is_filtered=self.config.component_config.tl_enabled
                )
                result['generated_sql'] = generated_sql
        
        except Exception as e:
            result['errors'].append(str(e))
        
        return result
    
    def _rewrite_query(self, query: str) -> str:
        """
        转写查询（占位实现）
        
        Args:
            query: 原始查询
            
        Returns:
            转写后的查询
        """
        # TODO: 实现实际的LLM调用
        # messages = self.qr_prompt_builder.build_messages(query)
        # response = llm_client.chat.completions.create(...)
        # return response.choices[0].message.content
        
        # 占位实现：返回原始查询
        return query
    
    def _retrieve_tables(self, query: str) -> List[str]:
        """
        检索相关表（占位实现）
        
        Args:
            query: 查询文本
            
        Returns:
            相关表列表
        """
        # TODO: 实现实际的Table Linking
        # from taco_sql.table_linking.retrieval.table_retrieval import TableRetriever
        # retriever = TableRetriever(...)
        # tables = retriever.retrieve_top_k_tables([query], k=self.config.component_config.tl_top_k)
        # return tables[0]
        
        # 占位实现：返回空列表
        return []
    
    def _plan_query(
        self,
        query: str,
        relevant_tables: List[str],
        schema: Dict
    ) -> List[Dict]:
        """
        规划查询（占位实现）
        
        Args:
            query: 查询文本
            relevant_tables: 相关表列表
            schema: Schema字典
            
        Returns:
            执行计划列表
        """
        # TODO: 实现实际的Query Planning
        # prompt = self.qp_prompt_builder.build_prompt(query, relevant_tables, schema)
        # response = llm_client.chat.completions.create(...)
        # plan_json = response.choices[0].message.content
        # plan = self.qp_prompt_builder.parse_plan(plan_json)
        # return plan
        
        # 占位实现：返回默认计划
        return [{
            'subquery': query,
            'tables': relevant_tables,
            'order': 1,
            'dependencies': []
        }]
    
    def _generate_sql(
        self,
        query: str,
        schema_text: str,
        database: str,
        is_filtered: bool = False
    ) -> str:
        """
        生成SQL（占位实现）
        
        Args:
            query: 查询文本
            schema_text: Schema文本
            database: 数据库名称
            is_filtered: Schema是否已过滤
            
        Returns:
            生成的SQL
        """
        # TODO: 实现实际的SQL生成
        # prompt = self.sql_prompt_builder.build_prompt(
        #     query, schema_text, database, is_filtered=is_filtered
        # )
        # response = llm_client.chat.completions.create(...)
        # sql = response.choices[0].message.content
        # sql = self.sql_prompt_builder.clean_sql(sql)
        # return sql
        
        # 占位实现：返回占位SQL
        return "SELECT * FROM \"表名\";"
    
    def run_batch(
        self,
        test_data: List[Dict],
        output_path: Optional[str] = None
    ) -> List[Dict]:
        """
        批量运行实验
        
        Args:
            test_data: 测试数据列表
            output_path: 输出路径（可选）
            
        Returns:
            实验结果列表
        """
        results = []
        
        for item in test_data:
            query = item.get('natural_language_query', '')
            database = item.get('database', '')
            schema = item.get('schema', {})
            ground_truth_sql = item.get('sql', '')
            
            result = self.run_single_query(
                query=query,
                schema=schema,
                database=database,
                ground_truth_sql=ground_truth_sql
            )
            
            # 添加元数据
            result['item_id'] = item.get('id', '')
            result['database'] = database
            result['ground_truth_sql'] = ground_truth_sql
            
            results.append(result)
        
        # 保存结果
        if output_path:
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(results, f, ensure_ascii=False, indent=2)
        
        return results


def run_experiment(
    setting: str,
    model_name: str,
    dataset_name: str = "taco_beijing",
    test_data_path: Optional[str] = None,
    output_path: Optional[str] = None,
    **kwargs
) -> List[Dict]:
    """
    运行实验的主函数
    
    Args:
        setting: 实验设置（"origin", "qr", "qr_tl", "qr_tl_qp"）
        model_name: 模型名称
        dataset_name: 数据集名称
        test_data_path: 测试数据路径
        output_path: 输出路径
        **kwargs: 其他配置参数
        
    Returns:
        实验结果列表
    """
    # 创建配置
    config = create_experiment_config(
        setting=setting,
        model_name=model_name,
        dataset_name=dataset_name,
        **kwargs
    )
    
    # 创建运行器
    runner = ExperimentRunner(config)
    
    # 加载测试数据
    if test_data_path and os.path.exists(test_data_path):
        with open(test_data_path, 'r', encoding='utf-8') as f:
            test_data = json.load(f)
    else:
        # 使用默认路径
        default_path = f"benchmark/data/final/{dataset_name}/test.json"
        if os.path.exists(default_path):
            with open(default_path, 'r', encoding='utf-8') as f:
                test_data = json.load(f)
        else:
            raise FileNotFoundError(f"Test data not found: {test_data_path or default_path}")
    
    # 运行实验
    results = runner.run_batch(test_data, output_path)
    
    return results


# 示例使用
if __name__ == "__main__":
    # 示例：运行Origin设置实验
    print("运行Origin设置实验...")
    origin_results = run_experiment(
        setting="origin",
        model_name="gpt-4o",
        dataset_name="taco_beijing",
        output_path="results/origin_gpt4o.json"
    )
    print(f"完成，共处理 {len(origin_results)} 个查询")
    
    # 示例：运行完整TACO-SQL实验
    print("\n运行完整TACO-SQL实验...")
    full_results = run_experiment(
        setting="qr_tl_qp",
        model_name="gpt-4o",
        dataset_name="taco_beijing",
        output_path="results/qr_tl_qp_gpt4o.json"
    )
    print(f"完成，共处理 {len(full_results)} 个查询")

