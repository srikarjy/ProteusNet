"""
Configuration loading and validation utilities.
"""

import yaml
from dataclasses import dataclass
from typing import Dict, Any, Optional
from pathlib import Path


@dataclass
class ModelConfig:
    """Model architecture configuration."""
    vocab_size: int = 25
    d_model: int = 512
    n_heads: int = 8
    n_layers: int = 12
    max_seq_length: int = 1024
    hierarchical_levels: int = 3
    physics_features: int = 64


@dataclass
class TrainingConfig:
    """Training configuration."""
    batch_size: int = 32
    learning_rate: float = 1e-4
    weight_decay: float = 1e-5
    max_epochs: int = 100
    warmup_steps: int = 1000
    gradient_clip: float = 1.0
    
    # Loss weights for multi-task learning
    loss_weights: Dict[str, float] = None
    
    def __post_init__(self):
        if self.loss_weights is None:
            self.loss_weights = {
                'function_prediction': 1.0,
                'structure_prediction': 0.5,
                'stability_prediction': 0.3,
                'physics_consistency': 0.2,
                'evolutionary_similarity': 0.1
            }


@dataclass
class PhysicsConfig:
    """Physics-informed components configuration."""
    enable_distance_bias: bool = True
    enable_interaction_matrix: bool = True
    enable_secondary_structure_bias: bool = True
    physics_regularization: float = 0.01


@dataclass
class DataConfig:
    """Data processing configuration."""
    datasets: list = None
    max_sequence_length: int = 1024
    min_sequence_length: int = 50
    remove_ambiguous: bool = True
    
    def __post_init__(self):
        if self.datasets is None:
            self.datasets = ["UniProt/Swiss-Prot", "PDB", "GO"]


@dataclass
class ProteusNetConfig:
    """Complete ProteusNet configuration."""
    model: ModelConfig
    training: TrainingConfig
    physics: PhysicsConfig
    data: DataConfig


class ConfigLoader:
    """Utility class for loading and validating YAML configurations."""
    
    @staticmethod
    def load_config(config_path: str) -> ProteusNetConfig:
        """
        Load configuration from YAML file.
        
        Args:
            config_path: Path to YAML configuration file
            
        Returns:
            ProteusNetConfig object with validated parameters
        """
        config_path = Path(config_path)
        if not config_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {config_path}")
        
        with open(config_path, 'r') as f:
            config_dict = yaml.safe_load(f)
        
        return ConfigLoader._dict_to_config(config_dict)
    
    @staticmethod
    def _dict_to_config(config_dict: Dict[str, Any]) -> ProteusNetConfig:
        """Convert dictionary to ProteusNetConfig object."""
        model_config = ModelConfig(**config_dict.get('model', {}))
        training_config = TrainingConfig(**config_dict.get('training', {}))
        physics_config = PhysicsConfig(**config_dict.get('physics', {}))
        data_config = DataConfig(**config_dict.get('data', {}))
        
        return ProteusNetConfig(
            model=model_config,
            training=training_config,
            physics=physics_config,
            data=data_config
        )
    
    @staticmethod
    def save_config(config: ProteusNetConfig, config_path: str) -> None:
        """
        Save configuration to YAML file.
        
        Args:
            config: ProteusNetConfig object to save
            config_path: Path where to save the configuration
        """
        config_dict = {
            'model': config.model.__dict__,
            'training': config.training.__dict__,
            'physics': config.physics.__dict__,
            'data': config.data.__dict__
        }
        
        with open(config_path, 'w') as f:
            yaml.dump(config_dict, f, default_flow_style=False, indent=2)
    
    @staticmethod
    def validate_config(config: ProteusNetConfig) -> None:
        """
        Validate configuration parameters.
        
        Args:
            config: Configuration to validate
            
        Raises:
            ValueError: If configuration parameters are invalid
        """
        # Model validation
        if config.model.vocab_size <= 0:
            raise ValueError("vocab_size must be positive")
        if config.model.d_model <= 0:
            raise ValueError("d_model must be positive")
        if config.model.n_heads <= 0:
            raise ValueError("n_heads must be positive")
        if config.model.d_model % config.model.n_heads != 0:
            raise ValueError("d_model must be divisible by n_heads")
        
        # Training validation
        if config.training.batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if config.training.learning_rate <= 0:
            raise ValueError("learning_rate must be positive")
        if config.training.max_epochs <= 0:
            raise ValueError("max_epochs must be positive")
        
        # Loss weights validation
        if any(weight < 0 for weight in config.training.loss_weights.values()):
            raise ValueError("All loss weights must be non-negative")
        
        # Data validation
        if config.data.max_sequence_length <= 0:
            raise ValueError("max_sequence_length must be positive")
        if config.data.min_sequence_length <= 0:
            raise ValueError("min_sequence_length must be positive")
        if config.data.min_sequence_length >= config.data.max_sequence_length:
            raise ValueError("min_sequence_length must be less than max_sequence_length")