"""
Question Rewriting Prompt Strategy Implementation

Implements prompt construction logic for question rewriting based on the design document.
"""

from typing import List, Tuple, Dict, Optional


class QuestionRewritingPrompt:
    """Question Rewriting Prompt Builder"""
    
    # Few-shot examples (from documentation)
    FEW_SHOTS: List[Tuple[str, str]] = [
        (
            "I need my employee records to finish a report. Please tell me where I can get my employee records. My employee ID is E12345, Thanks.",
            "Find storage locations for employee records with ID E12345.",
        ),
        (
            "嗨，我想知道 2023 年 5 月的销售数据，最好能按地区分组，谢谢！",
            "Retrieve 2023-05 sales figures grouped by region.",
        ),
        (
            "Our customer table keeps crashing! What are the emails of users registered after 2024-01-01?",
            "List emails of customers registered after 2024-01-01.",
        ),
    ]
    
    # System prompt (from documentation)
    SYSTEM_PROMPT = (
        "You rewrite user questions for SQL retrieval. Remove irrelevant chatter, disambiguate entities, "
        "and output one concise sentence expressing the core intent while preserving key filters."
    )
    
    def __init__(self, temperature: float = 0.3, top_p: float = 0.9):
        """
        Initialize prompt builder
        
        Args:
            temperature: Temperature parameter (default 0.3 for rewriting stability)
            top_p: Top-p sampling parameter (default 0.9)
        """
        self.temperature = temperature
        self.top_p = top_p
    
    def build_messages(self, query: str) -> List[Dict[str, str]]:
        """
        Build complete message list (System + Few-shot + Current query)
        
        Args:
            query: Original user query
            
        Returns:
            Message list in OpenAI API format
        """
        messages = [
            {
                "role": "system",
                "content": self.SYSTEM_PROMPT
            }
        ]
        
        # Add few-shot examples
        for src, tgt in self.FEW_SHOTS:
            messages.append({"role": "user", "content": src})
            messages.append({"role": "assistant", "content": tgt})
        
        # Add current query
        messages.append({"role": "user", "content": query})
        
        return messages
    
    def build_simple_prompt(self, query: str, context: Optional[Dict] = None) -> str:
        """
        Build simple format prompt (for non-OpenAI API models)
        
        Args:
            query: Original query
            context: Optional context information
            
        Returns:
            Formatted prompt string
        """
        prompt = f"""Rewrite the following user query into a clearer, more structured form, removing redundant information and clarifying query intent.

Original query: {query}
"""
        
        if context:
            import json
            prompt += f"\nContext information: {json.dumps(context, ensure_ascii=False)}"
        
        prompt += "\nRewritten query:"
        
        return prompt
    
    def get_config(self) -> Dict:
        """
        Get model call configuration
        
        Returns:
            Configuration dictionary
        """
        return {
            "temperature": self.temperature,
            "top_p": self.top_p,
            "max_tokens": 512
        }


def create_rewriting_prompt_builder(temperature: float = 0.3, top_p: float = 0.9) -> QuestionRewritingPrompt:
    """
    Create Question Rewriting Prompt builder
    
    Args:
        temperature: Temperature parameter
        top_p: Top-p sampling parameter
        
    Returns:
        QuestionRewritingPrompt instance
    """
    return QuestionRewritingPrompt(temperature=temperature, top_p=top_p)


# Example usage
if __name__ == "__main__":
    # Create prompt builder
    prompt_builder = create_rewriting_prompt_builder()
    
    # Example query
    example_query = "I want to see the enterprise registration situation in Beijing in recent years, including registration count and registered capital"
    
    # Build OpenAI format messages
    messages = prompt_builder.build_messages(example_query)
    print("OpenAI format messages:")
    for msg in messages:
        print(f"{msg['role']}: {msg['content'][:100]}...")
    
    # Build simple format prompt
    simple_prompt = prompt_builder.build_simple_prompt(example_query)
    print("\nSimple format prompt:")
    print(simple_prompt)
    
    # Get configuration
    config = prompt_builder.get_config()
    print("\nModel configuration:")
    print(config)

