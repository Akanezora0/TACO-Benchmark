"""
TACO-SQL消融实验模块

包含实验配置、Prompt策略和运行器
"""

from .config import (
    ExperimentConfig,
    ModelConfig,
    ComponentConfig,
    create_experiment_config,
    MODEL_CONFIGS
)

from .experiment_runner import (
    ExperimentRunner,
    run_experiment
)

__all__ = [
    'ExperimentConfig',
    'ModelConfig',
    'ComponentConfig',
    'create_experiment_config',
    'MODEL_CONFIGS',
    'ExperimentRunner',
    'run_experiment',
]

