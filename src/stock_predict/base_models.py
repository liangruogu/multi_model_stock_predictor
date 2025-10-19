"""
Base Models Module

Provides base classes and registry mechanism for different model types.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, Tuple
import torch
import torch.nn as nn
from torch.utils.data import DataLoader


class BaseStockModel(nn.Module, ABC):
    """Base class for all stock prediction models."""

    def __init__(self, **kwargs):
        super().__init__()
        self.model_name = self.__class__.__name__

    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of the model."""
        pass

    @abstractmethod
    def get_model_params(self) -> Dict[str, Any]:
        """Get model parameters for saving."""
        pass

    @classmethod
    @abstractmethod
    def from_params(cls, params: Dict[str, Any]) -> 'BaseStockModel':
        """Create model from parameters."""
        pass


class BaseModelTrainer(ABC):
    """Base class for model trainers."""

    def __init__(self, model: BaseStockModel, device: torch.device):
        self.model = model
        self.device = device

    @abstractmethod
    def train(self, train_loader: DataLoader, val_loader: DataLoader,
              epochs: int, early_stopping_patience: int) -> Tuple[list, list]:
        """Train the model."""
        pass

    @abstractmethod
    def predict(self, data_loader: DataLoader) -> Tuple[torch.Tensor, torch.Tensor]:
        """Make predictions."""
        pass

    @abstractmethod
    def save_model(self, path: str, train_losses: list, val_losses: list,
                   model_params: Dict[str, Any], config: Dict[str, Any]):
        """Save model and training data."""
        pass

    @classmethod
    @abstractmethod
    def load_model(cls, path: str, device: torch.device) -> Tuple[BaseStockModel, list, list, Dict[str, Any], Dict[str, Any]]:
        """Load model and training data."""
        pass


class ModelRegistry:
    """Registry for different model types and their trainers."""

    _models = {}
    _trainers = {}

    @classmethod
    def register_model(cls, name: str, model_class: type, trainer_class: type):
        """Register a model and its trainer."""
        cls._models[name] = model_class
        cls._trainers[name] = trainer_class

    @classmethod
    def get_model_class(cls, name: str) -> type:
        """Get model class by name."""
        if name not in cls._models:
            raise ValueError(f"Model '{name}' not registered. Available models: {list(cls._models.keys())}")
        return cls._models[name]

    @classmethod
    def get_trainer_class(cls, name: str) -> type:
        """Get trainer class by name."""
        if name not in cls._trainers:
            raise ValueError(f"Trainer for model '{name}' not registered.")
        return cls._trainers[name]

    @classmethod
    def list_models(cls) -> list:
        """List all registered models."""
        return list(cls._models.keys())

    @classmethod
    def create_model(cls, name: str, **kwargs) -> BaseStockModel:
        """Create model instance by name."""
        model_class = cls.get_model_class(name)
        return model_class(**kwargs)

    @classmethod
    def create_trainer(cls, name: str, model: BaseStockModel, device: torch.device) -> BaseModelTrainer:
        """Create trainer instance by name."""
        trainer_class = cls.get_trainer_class(name)
        return trainer_class(model, device)