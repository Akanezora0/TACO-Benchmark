"""
TACO-SQL Experiment Configuration

Defines experimental settings, model configurations, and parameters
"""

from typing import Dict, List, Optional
from dataclasses import dataclass, field


@dataclass
class ModelConfig:
    """Model configuration"""
    model_name: str
    model_type: str  # "base_llm", "llm_based", "sft_based", "hybrid"
    api_key: Optional[str] = None
    base_url: Optional[str] = None
    temperature: float = 0.1
    max_tokens: int = 2000
    context_window: int = 128000
    
    # SFT model specific configuration
    model_path: Optional[str] = None
    device: Optional[str] = None
    num_beams: int = 4
    num_return_sequences: int = 4
    
    # Schema filtering configuration
    schema_filter_enabled: bool = False
    schema_filter_model_path: Optional[str] = None
    max_tables: int = 7
    max_columns: int = 20


@dataclass
class ComponentConfig:
    """Component configuration"""
    # Question Rewriting
    qr_enabled: bool = False
    qr_temperature: float = 0.3
    qr_top_p: float = 0.9
    qr_max_tokens: int = 512
    
    # Table Linking
    tl_enabled: bool = False
    tl_top_k: int = 5
    tl_query_model_path: Optional[str] = None
    tl_table_model_path: Optional[str] = None
    tl_merged_table_path: Optional[str] = None
    
    # Query Planning
    qp_enabled: bool = False
    qp_temperature: float = 0.3
    qp_max_tokens: int = 1024
    
    # SQL Generation
    sg_temperature: float = 0.1
    sg_max_tokens: int = 2000


@dataclass
class ExperimentConfig:
    """Experiment configuration"""
    # Experimental setting
    setting: str  # "origin", "qr", "qr_tl", "qr_tl_qp"
    model_config: ModelConfig
    component_config: ComponentConfig = field(default_factory=ComponentConfig)
    
    # Dataset configuration
    dataset_name: str = "taco_beijing"
    test_data_path: Optional[str] = None
    schema_dir: Optional[str] = None
    database_dir: Optional[str] = None
    
    # Output configuration
    output_dir: str = "results"
    save_intermediate: bool = True
    
    def __post_init__(self):
        """Automatically configure components based on setting"""
        if self.setting == "origin":
            self.component_config.qr_enabled = False
            self.component_config.tl_enabled = False
            self.component_config.qp_enabled = False
        elif self.setting == "qr":
            self.component_config.qr_enabled = True
            self.component_config.tl_enabled = False
            self.component_config.qp_enabled = False
        elif self.setting == "qr_tl":
            self.component_config.qr_enabled = True
            self.component_config.tl_enabled = True
            self.component_config.qp_enabled = False
        elif self.setting == "qr_tl_qp":
            self.component_config.qr_enabled = True
            self.component_config.tl_enabled = True
            self.component_config.qp_enabled = True
        else:
            raise ValueError(f"Unknown setting: {self.setting}")


# Predefined model configurations
MODEL_CONFIGS: Dict[str, ModelConfig] = {
    "gpt-4o": ModelConfig(
        model_name="gpt-4o",
        model_type="base_llm",
        temperature=0.1,
        max_tokens=2000,
        context_window=128000
    ),
    "gpt-4o-mini": ModelConfig(
        model_name="gpt-4o-mini",
        model_type="base_llm",
        temperature=0.1,
        max_tokens=2000,
        context_window=128000
    ),
    "gpt-o1": ModelConfig(
        model_name="o1",
        model_type="base_llm",
        temperature=0.1,
        max_tokens=2000,
        context_window=200000
    ),
    "deepseek-r1": ModelConfig(
        model_name="deepseek-r1",
        model_type="base_llm",
        temperature=0.1,
        max_tokens=2000,
        context_window=64000
    ),
    "codes-33b": ModelConfig(
        model_name="codes-33b",
        model_type="sft_based",
        model_path="models/codes-33b",
        device="cuda:0",
        num_beams=4,
        num_return_sequences=4,
        schema_filter_enabled=True,
        schema_filter_model_path="models/sic_ckpts/sic_spider",
        max_tables=7,
        max_columns=20
    ),
}


def create_experiment_config(
    setting: str,
    model_name: str,
    dataset_name: str = "taco_beijing",
    **kwargs
) -> ExperimentConfig:
    """
    Create experiment configuration
    
    Args:
        setting: Experimental setting ("origin", "qr", "qr_tl", "qr_tl_qp")
        model_name: Model name
        dataset_name: Dataset name
        **kwargs: Other configuration parameters
        
    Returns:
        ExperimentConfig instance
    """
    if model_name not in MODEL_CONFIGS:
        raise ValueError(f"Unknown model: {model_name}")
    
    model_config = MODEL_CONFIGS[model_name]
    
    # Create component configuration
    component_config = ComponentConfig()
    
    # Apply component configurations from kwargs
    if "qr_temperature" in kwargs:
        component_config.qr_temperature = kwargs["qr_temperature"]
    if "tl_top_k" in kwargs:
        component_config.tl_top_k = kwargs["tl_top_k"]
    if "qp_temperature" in kwargs:
        component_config.qp_temperature = kwargs["qp_temperature"]
    
    config = ExperimentConfig(
        setting=setting,
        model_config=model_config,
        component_config=component_config,
        dataset_name=dataset_name,
        **{k: v for k, v in kwargs.items() if k not in ["qr_temperature", "tl_top_k", "qp_temperature"]}
    )
    
    return config


# Example usage
if __name__ == "__main__":
    # Create Origin setting configuration
    origin_config = create_experiment_config(
        setting="origin",
        model_name="gpt-4o",
        dataset_name="taco_beijing"
    )
    print("Origin configuration:")
    print(f"  Setting: {origin_config.setting}")
    print(f"  QR Enabled: {origin_config.component_config.qr_enabled}")
    print(f"  TL Enabled: {origin_config.component_config.tl_enabled}")
    print(f"  QP Enabled: {origin_config.component_config.qp_enabled}")
    print()
    
    # Create full TACO-SQL configuration
    full_config = create_experiment_config(
        setting="qr_tl_qp",
        model_name="gpt-4o",
        dataset_name="taco_beijing",
        tl_top_k=5
    )
    print("Full TACO-SQL configuration:")
    print(f"  Setting: {full_config.setting}")
    print(f"  QR Enabled: {full_config.component_config.qr_enabled}")
    print(f"  TL Enabled: {full_config.component_config.tl_enabled}")
    print(f"  TL Top-K: {full_config.component_config.tl_top_k}")
    print(f"  QP Enabled: {full_config.component_config.qp_enabled}")

