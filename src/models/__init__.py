"""
Neural network models and architectures for protein analysis.

This module contains the core ProteusNet architecture and its components:
- ProteusNet: Main physics-informed transformer model
- PhysicsInformedAttention: Attention mechanism with protein physics constraints
- HierarchicalEncoder: Multi-scale protein encoder
"""

from .proteus_net import ProteusNet
from .physics_attention import PhysicsInformedAttention
from .hierarchical_encoder import HierarchicalEncoder

__all__ = [
    "ProteusNet",
    "PhysicsInformedAttention", 
    "HierarchicalEncoder"
]