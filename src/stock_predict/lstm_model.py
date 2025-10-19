"""
LSTM Model Implementation

LSTM-based stock prediction model with dual-layer architecture.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import os
import json
from typing import Dict, Any, Tuple, List
from torch.utils.data import DataLoader

from .base_models import BaseStockModel, BaseModelTrainer


class LSTMStockModel(BaseStockModel):
    """LSTM model for stock prediction with dual-layer architecture."""

    def __init__(self, input_size: int = 1, hidden_size1: int = 50, hidden_size2: int = 64,
                 fc1_size: int = 32, fc2_size: int = 16, output_size: int = 1, dropout: float = 0.0):
        super().__init__()

        self.input_size = input_size
        self.hidden_size1 = hidden_size1
        self.hidden_size2 = hidden_size2
        self.fc1_size = fc1_size
        self.fc2_size = fc2_size
        self.output_size = output_size
        self.dropout = dropout

        # LSTM layers
        self.lstm1 = nn.LSTM(input_size, hidden_size1, batch_first=True)
        self.lstm2 = nn.LSTM(hidden_size1, hidden_size2, batch_first=True)

        # Fully connected layers
        self.fc1 = nn.Linear(hidden_size2, fc1_size)
        self.fc2 = nn.Linear(fc1_size, fc2_size)
        self.fc3 = nn.Linear(fc2_size, output_size)

        self.dropout_layer = nn.Dropout(dropout)

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        """Initialize model weights."""
        for name, param in self.named_parameters():
            if 'weight_ih' in name:
                torch.nn.init.xavier_uniform_(param.data)
            elif 'weight_hh' in name:
                torch.nn.init.orthogonal_(param.data)
            elif 'bias' in name:
                param.data.fill_(0)
                # Set forget gate bias to 1
                n = param.size(0)
                param.data[n//4:n//2].fill_(1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        # First LSTM layer
        lstm1_out, (h1, c1) = self.lstm1(x)

        # Second LSTM layer
        lstm2_out, (h2, c2) = self.lstm2(lstm1_out)

        # Take the last output
        last_output = lstm2_out[:, -1, :]

        # Apply dropout
        last_output = self.dropout_layer(last_output)

        # Fully connected layers
        out = torch.relu(self.fc1(last_output))
        out = torch.relu(self.fc2(out))
        out = self.fc3(out)

        return out

    def get_model_params(self) -> Dict[str, Any]:
        """Get model parameters for saving."""
        return {
            'model_type': 'LSTM',
            'input_size': self.input_size,
            'hidden_size1': self.hidden_size1,
            'hidden_size2': self.hidden_size2,
            'fc1_size': self.fc1_size,
            'fc2_size': self.fc2_size,
            'output_size': self.output_size,
            'dropout': self.dropout
        }

    @classmethod
    def from_params(cls, params: Dict[str, Any]) -> 'LSTMStockModel':
        """Create model from parameters."""
        return cls(
            input_size=params['input_size'],
            hidden_size1=params['hidden_size1'],
            hidden_size2=params['hidden_size2'],
            fc1_size=params['fc1_size'],
            fc2_size=params['fc2_size'],
            output_size=params['output_size'],
            dropout=params['dropout']
        )


class LSTMTrainer(BaseModelTrainer):
    """Trainer for LSTM model."""

    def __init__(self, model: LSTMStockModel, device: torch.device):
        super().__init__(model, device)
        self.criterion = nn.MSELoss()
        self.optimizer = optim.Adam(model.parameters(), lr=0.001)
        self.train_losses = []
        self.val_losses = []

    def train(self, train_loader: DataLoader, val_loader: DataLoader,
              epochs: int, early_stopping_patience: int, learning_rate: float = 0.001) -> Tuple[List[float], List[float]]:
        """Train the LSTM model."""
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

                self.optimizer.zero_grad()
                outputs = self.model(batch_x)
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
                    outputs = self.model(batch_x)
                    loss = self.criterion(outputs, batch_y)
                    val_loss += loss.item()

            avg_val_loss = val_loss / len(val_loader)
            self.val_losses.append(avg_val_loss)

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
                print(f'Epoch [{epoch+1}/{epochs}], Train Loss: {avg_train_loss:.6f}, Val Loss: {avg_val_loss:.6f}')

        return self.train_losses, self.val_losses

    def predict(self, data_loader: DataLoader) -> Tuple[torch.Tensor, torch.Tensor]:
        """Make predictions."""
        self.model.eval()
        predictions = []
        actuals = []

        with torch.no_grad():
            for batch_x, batch_y in data_loader:
                batch_x, batch_y = batch_x.to(self.device), batch_y.to(self.device)
                outputs = self.model(batch_x)

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
            'optimizer_state_dict': self.optimizer.state_dict()
        }

        torch.save(save_data, path)

    @classmethod
    def load_model(cls, path: str, device: torch.device) -> Tuple[LSTMStockModel, List[float], List[float], Dict[str, Any], Dict[str, Any]]:
        """Load model and training data."""
        save_data = torch.load(path, map_location=device)

        # Create model instance
        model = LSTMStockModel.from_params(save_data['model_params'])
        model.load_state_dict(save_data['model_state_dict'])
        model.to(device)

        # Create trainer instance
        trainer = cls(model, device)
        trainer.optimizer.load_state_dict(save_data['optimizer_state_dict'])

        return model, save_data['train_losses'], save_data['val_losses'], save_data['model_params'], save_data['config']