"""
Multi-Model Stock Prediction System

A comprehensive stock prediction system supporting multiple model architectures
including LSTM and Transformer models.
"""

from .core import MultiModelStockPredictor
from .model_registry import get_available_models, create_model, create_trainer
from .base_models import BaseStockModel, BaseModelTrainer
from .lstm_model import LSTMStockModel, LSTMTrainer
from .transformer_model import TransformerStockModel, TransformerTrainer

__version__ = "2.0.0"
__author__ = "Stock Prediction System Team"
__description__ = "Multi-Model Stock Prediction System"

__all__ = [
    'MultiModelStockPredictor',
    'get_available_models',
    'create_model',
    'create_trainer',
    'BaseStockModel',
    'BaseModelTrainer',
    'LSTMStockModel',
    'LSTMTrainer',
    'TransformerStockModel',
    'TransformerTrainer'
]