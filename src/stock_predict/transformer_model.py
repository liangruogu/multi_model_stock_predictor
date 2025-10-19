"""
Transformer Model Implementation

Transformer-based stock prediction model for time series forecasting.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import os
import json
import math
from typing import Dict, Any, Tuple, List
from torch.utils.data import DataLoader

from .base_models import BaseStockModel, BaseModelTrainer


class PositionalEncoding(nn.Module):
    """Positional encoding for transformer."""

    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:x.size(0), :]


class TransformerStockModel(BaseStockModel):
    """Transformer model for stock prediction."""

    def __init__(self, input_size: int = 1, d_model: int = 128, nhead: int = 8,
                 num_encoder_layers: int = 6, dim_feedforward: int = 512,
                 dropout: float = 0.1, output_size: int = 1, max_seq_len: int = 1000):
        super().__init__()

        self.input_size = input_size
        self.d_model = d_model
        self.nhead = nhead
        self.num_encoder_layers = num_encoder_layers
        self.dim_feedforward = dim_feedforward
        self.dropout = dropout
        self.output_size = output_size
        self.max_seq_len = max_seq_len

        # Input projection
        self.input_projection = nn.Linear(input_size, d_model)

        # Positional encoding
        self.pos_encoder = PositionalEncoding(d_model, max_seq_len)

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation='relu',
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_encoder_layers)

        # Output layers
        self.output_projection = nn.Linear(d_model, output_size)

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        """Initialize model weights."""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, x: torch.Tensor, src_key_padding_mask: torch.Tensor = None) -> torch.Tensor:
        """Forward pass."""
        # x shape: (batch_size, seq_len, input_size)

        # Project input to d_model
        x = self.input_projection(x) * math.sqrt(self.d_model)

        # Add positional encoding
        x = x.transpose(0, 1)  # (seq_len, batch_size, d_model)
        x = self.pos_encoder(x)
        x = x.transpose(0, 1)  # (batch_size, seq_len, d_model)

        # Transformer encoder
        x = self.transformer_encoder(x, src_key_padding_mask=src_key_padding_mask)

        # Take the last output for prediction
        last_output = x[:, -1, :]  # (batch_size, d_model)

        # Project to output size
        output = self.output_projection(last_output)  # (batch_size, output_size)

        return output

    def generate_padding_mask(self, x: torch.Tensor, pad_value: float = 0.0) -> torch.Tensor:
        """Generate padding mask for transformer."""
        # x shape: (batch_size, seq_len, input_size)
        # Check if all features are padding value
        mask = (x == pad_value).all(dim=-1)  # (batch_size, seq_len)
        return mask

    def get_model_params(self) -> Dict[str, Any]:
        """Get model parameters for saving."""
        return {
            'model_type': 'Transformer',
            'input_size': self.input_size,
            'd_model': self.d_model,
            'nhead': self.nhead,
            'num_encoder_layers': self.num_encoder_layers,
            'dim_feedforward': self.dim_feedforward,
            'dropout': self.dropout,
            'output_size': self.output_size,
            'max_seq_len': self.max_seq_len
        }

    @classmethod
    def from_params(cls, params: Dict[str, Any]) -> 'TransformerStockModel':
        """Create model from parameters."""
        return cls(
            input_size=params['input_size'],
            d_model=params['d_model'],
            nhead=params['nhead'],
            num_encoder_layers=params['num_encoder_layers'],
            dim_feedforward=params['dim_feedforward'],
            dropout=params['dropout'],
            output_size=params['output_size'],
            max_seq_len=params['max_seq_len']
        )


class TransformerTrainer(BaseModelTrainer):
    """Trainer for Transformer model."""

    def __init__(self, model: TransformerStockModel, device: torch.device):
        super().__init__(model, device)
        self.criterion = nn.MSELoss()
        self.optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.5, patience=10
        )
        self.train_losses = []
        self.val_losses = []

    def train(self, train_loader: DataLoader, val_loader: DataLoader,
              epochs: int, early_stopping_patience: int, learning_rate: float = 0.001) -> Tuple[List[float], List[float]]:
        """Train the Transformer model."""
        # Update learning rate if provided
        if learning_rate != 0.001:
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = learning_rate

        self.train_losses = []
        self.val_losses = []
        best_val_loss = float('inf')
        patience_counter = 0

        for epoch in range(epochs):
            # Training phase
            self.model.train()
            train_loss = 0.0

            for batch_x, batch_y in train_loader:
                batch_x, batch_y = batch_x.to(self.device), batch_y.to(self.device)

                # Generate padding mask if needed
                padding_mask = self.model.generate_padding_mask(batch_x)

                self.optimizer.zero_grad()
                outputs = self.model(batch_x, src_key_padding_mask=padding_mask)
                loss = self.criterion(outputs, batch_y)
                loss.backward()

                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

                self.optimizer.step()
                train_loss += loss.item()

            avg_train_loss = train_loss / len(train_loader)
            self.train_losses.append(avg_train_loss)

            # Validation phase
            self.model.eval()
            val_loss = 0.0

            with torch.no_grad():
                for batch_x, batch_y in val_loader:
                    batch_x, batch_y = batch_x.to(self.device), batch_y.to(self.device)

                    # Generate padding mask if needed
                    padding_mask = self.model.generate_padding_mask(batch_x)

                    outputs = self.model(batch_x, src_key_padding_mask=padding_mask)
                    loss = self.criterion(outputs, batch_y)
                    val_loss += loss.item()

            avg_val_loss = val_loss / len(val_loader)
            self.val_losses.append(avg_val_loss)

            # Learning rate scheduling
            self.scheduler.step(avg_val_loss)

            # Early stopping
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                patience_counter = 0
                # Save best model state
                best_model_state = self.model.state_dict().copy()
            else:
                patience_counter += 1

            if patience_counter >= early_stopping_patience:
                print(f"Early stopping at epoch {epoch+1}")
                # Restore best model
                self.model.load_state_dict(best_model_state)
                break

            if (epoch + 1) % 10 == 0:
                current_lr = self.optimizer.param_groups[0]['lr']
                print(f'Epoch [{epoch+1}/{epochs}], Train Loss: {avg_train_loss:.6f}, Val Loss: {avg_val_loss:.6f}, LR: {current_lr:.6f}')

        return self.train_losses, self.val_losses

    def predict(self, data_loader: DataLoader) -> Tuple[torch.Tensor, torch.Tensor]:
        """Make predictions."""
        self.model.eval()
        predictions = []
        actuals = []

        with torch.no_grad():
            for batch_x, batch_y in data_loader:
                batch_x, batch_y = batch_x.to(self.device), batch_y.to(self.device)

                # Generate padding mask if needed
                padding_mask = self.model.generate_padding_mask(batch_x)

                outputs = self.model(batch_x, src_key_padding_mask=padding_mask)

                predictions.extend(outputs.cpu().numpy())
                actuals.extend(batch_y.cpu().numpy())

        return torch.tensor(predictions), torch.tensor(actuals)

    def save_model(self, path: str, train_losses: List[float], val_losses: List[float],
                   model_params: Dict[str, Any], config: Dict[str, Any]):
        """Save model and training data."""
        os.makedirs(os.path.dirname(path), exist_ok=True)

        save_data = {
            'model_state_dict': self.model.state_dict(),
            'train_losses': train_losses,
            'val_losses': val_losses,
            'model_params': model_params,
            'config': config,
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict()
        }

        torch.save(save_data, path)

    @classmethod
    def load_model(cls, path: str, device: torch.device) -> Tuple[TransformerStockModel, List[float], List[float], Dict[str, Any], Dict[str, Any]]:
        """Load model and training data."""
        save_data = torch.load(path, map_location=device)

        # Create model instance
        model = TransformerStockModel.from_params(save_data['model_params'])
        model.load_state_dict(save_data['model_state_dict'])
        model.to(device)

        # Create trainer instance
        trainer = cls(model, device)
        trainer.optimizer.load_state_dict(save_data['optimizer_state_dict'])
        trainer.scheduler.load_state_dict(save_data['scheduler_state_dict'])

        return model, save_data['train_losses'], save_data['val_losses'], save_data['model_params'], save_data['config']