"""
ProteusNet: Physics-informed transformer for hierarchical protein understanding.

This package implements a novel transformer architecture that incorporates protein
physics constraints and hierarchical processing for comprehensive protein analysis.
"""

__version__ = "0.1.0"
__author__ = "ProteusNet Research Team"

from . import models
from . import training
from . import data
from . import utils

__all__ = ["models", "training", "data", "utils"]