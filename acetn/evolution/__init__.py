"""
Evolution module for iPEPS tensor network updates.

This module handles the evolution/optimization of tensor networks using
various update algorithms (full update, simple update, etc.).
"""
# Load compiled C++ extensions if available
from . import _extensions

# Import main classes and functions
from .als_solver import ALSSolver
from .evolve import evolve
from .gate import Gate
from .tensor_update import TensorUpdater
from .full_update import FullUpdater
from .fast_full_update import FastFullUpdater
from .simple_update import SimpleUpdater

__all__ = [
    'ALSSolver',
    'evolve',
    'Gate',
    'TensorUpdater',
    'FullUpdater',
    'FastFullUpdater',
    'SimpleUpdater',
]

