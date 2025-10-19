"""
Model Registry Module

Centralized registry for all available models and their trainers.
"""

from .base_models import ModelRegistry
from .lstm_model import LSTMStockModel, LSTMTrainer
from .transformer_model import TransformerStockModel, TransformerTrainer


def register_all_models():
    """Register all available models."""
    # Register LSTM model
    ModelRegistry.register_model(
        name="LSTM",
        model_class=LSTMStockModel,
        trainer_class=LSTMTrainer
    )

    # Register Transformer model
    ModelRegistry.register_model(
        name="Transformer",
        model_class=TransformerStockModel,
        trainer_class=TransformerTrainer
    )



def get_available_models():
    """Get list of available models."""
    return ModelRegistry.list_models()


def create_model(model_type: str, **kwargs):
    """Create model instance."""
    return ModelRegistry.create_model(model_type, **kwargs)


def create_trainer(model_type: str, model, device):
    """Create trainer instance."""
    return ModelRegistry.create_trainer(model_type, model, device)


# Initialize registry when module is imported
register_all_models()