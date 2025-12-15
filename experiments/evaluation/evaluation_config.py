"""
Evaluation Configuration

Standardized configuration for evaluation to ensure consistent and fair evaluation across all models.
"""

from typing import Dict, Optional
from dataclasses import dataclass


@dataclass
class EvaluationConfig:
    """Configuration for evaluation"""
    
    # Evaluation mode
    evaluation_mode: str = "execution"  # "execution" or "exact_match"
    
    # Execution evaluation settings
    normalize_results: bool = True  # Normalize result values for comparison
    ignore_order: bool = True  # Compare results as sets (order-independent)
    case_sensitive: bool = False  # Case-sensitive string comparison
    
    # Error handling
    treat_syntax_errors_as_wrong: bool = True
    treat_execution_errors_as_wrong: bool = True
    treat_timeout_as_wrong: bool = True
    timeout_seconds: Optional[int] = None
    
    # Result normalization
    normalize_nulls: bool = True  # Treat NULL, None, empty string consistently
    normalize_types: bool = True  # Convert all values to strings for comparison
    trim_whitespace: bool = True  # Trim whitespace from values
    
    # Statistical analysis
    calculate_confidence_intervals: bool = True
    confidence_level: float = 0.95  # 95% confidence interval
    
    # Output settings
    save_per_query_results: bool = True
    save_error_details: bool = True
    save_statistics: bool = True


# Standard evaluation configuration for fair comparison
STANDARD_EVAL_CONFIG = EvaluationConfig(
    evaluation_mode="execution",
    normalize_results=True,
    ignore_order=True,
    case_sensitive=False,
    treat_syntax_errors_as_wrong=True,
    treat_execution_errors_as_wrong=True,
    calculate_confidence_intervals=True,
    confidence_level=0.95
)


def get_evaluation_config(config_name: str = "standard") -> EvaluationConfig:
    """
    Get evaluation configuration
    
    Args:
        config_name: Configuration name ("standard", "strict", "lenient")
        
    Returns:
        EvaluationConfig instance
    """
    if config_name == "standard":
        return STANDARD_EVAL_CONFIG
    elif config_name == "strict":
        return EvaluationConfig(
            evaluation_mode="execution",
            normalize_results=False,
            ignore_order=False,
            case_sensitive=True
        )
    elif config_name == "lenient":
        return EvaluationConfig(
            evaluation_mode="execution",
            normalize_results=True,
            ignore_order=True,
            case_sensitive=False,
            normalize_nulls=True,
            trim_whitespace=True
        )
    else:
        raise ValueError(f"Unknown config name: {config_name}")


def create_custom_config(**kwargs) -> EvaluationConfig:
    """
    Create custom evaluation configuration
    
    Args:
        **kwargs: Configuration parameters
        
    Returns:
        EvaluationConfig instance
    """
    config = STANDARD_EVAL_CONFIG
    for key, value in kwargs.items():
        if hasattr(config, key):
            setattr(config, key, value)
    return config

