"""
Training pipeline and loss functions for ProteusNet.

This module contains:
- ProteusNetTrainer: Main training loop with multi-task objectives
- Loss functions: Physics-informed and standard loss functions
- Evaluation: Benchmark evaluation utilities
"""

from .trainer import ProteusNetTrainer
from .loss_functions import PhysicsConsistencyLoss
from .evaluation import ProteinBenchmarkEvaluator

__all__ = [
    "ProteusNetTrainer",
    "PhysicsConsistencyLoss",
    "ProteinBenchmarkEvaluator"
]