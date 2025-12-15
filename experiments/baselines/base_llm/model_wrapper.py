"""
Unified Model Interface for Base LLM Experiments

Provides a consistent interface for all base LLM models to ensure fair comparison.
"""

from typing import Dict, List, Optional, Tuple
from abc import ABC, abstractmethod
import threading
from openai import OpenAI


class BaseLLMModel(ABC):
    """Abstract base class for base LLM models"""
    
    def __init__(self, config: Dict):
        """
        Initialize model
        
        Args:
            config: Model configuration dictionary
        """
        self.config = config
        self.model_name = config.get('model_name', '')
        self.temperature = config.get('temperature', 0.1)
        self.max_tokens = config.get('max_tokens', 2000)
    
    @abstractmethod
    def generate_sql(
        self,
        query: str,
        schema_text: str,
        database: str
    ) -> Tuple[str, Dict]:
        """
        Generate SQL from natural language query
        
        Args:
            query: Natural language query
            schema_text: Formatted schema text
            database: Database name
            
        Returns:
            Tuple of (generated_sql, generation_info)
        """
        pass
    
    def get_config(self) -> Dict:
        """Get model configuration"""
        return self.config


class OpenAIModel(BaseLLMModel):
    """Wrapper for OpenAI API models (GPT-4o, GPT-4o-mini, GPT-o1, etc.)"""
    
    def __init__(self, config: Dict):
        super().__init__(config)
        self.api_key = config.get('api_key', '')
        self.base_url = config.get('base_url', 'https://api.openai.com/v1')
        self.model = config.get('model', '')
        
        # Thread-local storage for clients (thread-safe)
        self._local = threading.local()
    
    def _get_client(self) -> OpenAI:
        """Get thread-local OpenAI client"""
        if not hasattr(self._local, 'client'):
            self._local.client = OpenAI(
                api_key=self.api_key,
                base_url=self.base_url
            )
        return self._local.client
    
    def generate_sql(
        self,
        query: str,
        schema_text: str,
        database: str
    ) -> Tuple[str, Dict]:
        """
        Generate SQL using OpenAI API
        
        Args:
            query: Natural language query
            schema_text: Formatted schema text
            database: Database name
            
        Returns:
            Tuple of (generated_sql, generation_info)
        """
        from .prompt_strategy import BaseLLMPromptStrategy
        
        # Build messages
        messages = BaseLLMPromptStrategy.build_messages(
            query, schema_text, database
        )
        
        # Call API
        client = self._get_client()
        try:
            response = client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=self.temperature,
                max_tokens=self.max_tokens
            )
            
            sql = response.choices[0].message.content.strip()
            
            # Clean SQL
            sql = BaseLLMPromptStrategy.clean_sql_output(sql)
            
            # Generation info
            generation_info = {
                'model': self.model,
                'prompt_tokens': response.usage.prompt_tokens if hasattr(response.usage, 'prompt_tokens') else None,
                'completion_tokens': response.usage.completion_tokens if hasattr(response.usage, 'completion_tokens') else None,
                'total_tokens': response.usage.total_tokens if hasattr(response.usage, 'total_tokens') else None,
            }
            
            return sql, generation_info
            
        except Exception as e:
            return "", {'error': str(e)}


class DeepSeekModel(OpenAIModel):
    """Wrapper for DeepSeek API models (compatible with OpenAI API)"""
    
    def __init__(self, config: Dict):
        # DeepSeek uses OpenAI-compatible API
        if 'base_url' not in config:
            config['base_url'] = 'https://api.deepseek.com'
        super().__init__(config)


def create_model(model_name: str, config: Dict) -> BaseLLMModel:
    """
    Factory function to create appropriate model wrapper
    
    Args:
        model_name: Model name (e.g., 'gpt-4o', 'gpt-o1', 'deepseek-r1')
        config: Model configuration dictionary
        
    Returns:
        BaseLLMModel instance
    """
    config['model_name'] = model_name
    
    # OpenAI models
    if model_name in ['gpt-4', 'gpt-4o', 'gpt-4o-mini', 'gpt-o1', 'o1']:
        return OpenAIModel(config)
    
    # DeepSeek models
    elif model_name in ['deepseek-r1', 'deepseek-chat']:
        return DeepSeekModel(config)
    
    # Add other model types as needed
    else:
        raise ValueError(f"Unknown model: {model_name}")


def create_model_from_config(model_config: Dict) -> BaseLLMModel:
    """
    Create model from configuration dictionary
    
    Args:
        model_config: Model configuration (from experiment_config.py)
        
    Returns:
        BaseLLMModel instance
    """
    from .experiment_config import BASELINE_MODEL_CONFIGS
    
    model_name = model_config.get('model_name', '')
    
    if model_name in BASELINE_MODEL_CONFIGS:
        config = BASELINE_MODEL_CONFIGS[model_name]
        config_dict = {
            'model_name': config.model_name,
            'model': config.model_name,
            'api_key': config.api_key,
            'base_url': config.base_url,
            'temperature': config.temperature,
            'max_tokens': config.max_tokens,
        }
        return create_model(model_name, config_dict)
    else:
        raise ValueError(f"Model {model_name} not found in BASELINE_MODEL_CONFIGS")


# Example usage
if __name__ == "__main__":
    # Example configuration
    example_config = {
        'model_name': 'gpt-4o',
        'model': 'gpt-4o',
        'api_key': 'your-api-key',
        'base_url': 'https://api.openai.com/v1',
        'temperature': 0.1,
        'max_tokens': 2000
    }
    
    # Create model
    model = create_model('gpt-4o', example_config)
    
    # Example usage
    example_query = "Query enterprise registration data in Beijing"
    example_schema = """Database Schema Information:

Table: enterprise_registration
  Columns:
    - enterprise_name (TEXT)
    - registration_date (DATE)
"""
    
    # Generate SQL
    sql, info = model.generate_sql(
        query=example_query,
        schema_text=example_schema,
        database="test_db"
    )
    
    print(f"Generated SQL: {sql}")
    print(f"Generation Info: {info}")

