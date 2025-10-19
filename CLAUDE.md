# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a comprehensive multi-model stock prediction system that supports both LSTM and Transformer architectures for time series forecasting of stock prices. The system is built with PyTorch and includes complete data preprocessing, model training, evaluation, and visualization pipelines.

## Development Commands

### Environment Setup
```bash
# Install dependencies
pip install pandas numpy scikit-learn torch matplotlib seaborn scipy ipykernel loguru pyyaml

# Or using uv (recommended)
uv add pandas numpy scikit-learn torch matplotlib seaborn scipy loguru pyyaml
```

### Running the System
```bash
# Complete pipeline (train + predict)
uv run python main.py --mode full

# Training only
uv run python main.py --mode train

# Prediction only (requires trained model)
uv run python main.py --mode predict

# Use different model types
uv run python main.py --model LSTM --mode full
uv run python main.py --model Transformer --mode full

# Specify different data files
uv run python main.py --data data/600519_daily_qfq_8y.csv

# List available models
uv run python main.py --list-models
```

### Code Quality Tools
```bash
# Code formatting
black src/ main.py --line-length 88

# Linting
flake8 src/ main.py

# Type checking
mypy src/
```

### Testing
```bash
# Run tests (when available)
pytest

# Run with coverage
pytest --cov=src
```

## Architecture Overview

### Multi-Model Architecture
The system uses a modular design with a model registry pattern that supports:

- **LSTM Model**: Dual-layer LSTM with three fully connected layers
- **Transformer Model**: Standard transformer architecture for sequence modeling
- **Model Registry**: Centralized registration and creation of models and trainers

### Core Components

#### Main Entry Points
- `main.py`: CLI interface with argparse for running different modes
- `src/stock_predict/core.py`: Core orchestrator `MultiModelStockPredictor` class

#### Model Architecture
- `src/stock_predict/base_models.py`: Abstract base classes and registry system
- `src/stock_predict/lstm_model.py`: LSTM model and trainer implementations
- `src/stock_predict/transformer_model.py`: Transformer model and trainer implementations
- `src/stock_predict/model_registry.py`: Model registration and factory functions

#### Data Pipeline
- `src/stock_predict/data_preprocessor.py`: Two preprocessing approaches:
  - Original sliding window method with return conversion
  - Blog-style MinMaxScaler with lag features (recommended)
- `src/stock_predict/prediction.py`: Future prediction with multiple methods (recursive, direct, hybrid)

#### Evaluation & Visualization
- `src/stock_predict/evaluation.py`: Financial metrics calculation (Sharpe ratio, Sortino ratio, Calmar ratio, etc.)
- `src/stock_predict/visualization.py`: Comprehensive visualization reports

#### Configuration
- `src/stock_predict/config_loader.py`: YAML configuration loading and validation
- `src/stock_predict/config_multi.py`: Flattened configuration for compatibility
- `config.yaml`: Multi-model configuration with LSTM/Transformer parameters

### Data Flow

1. **Data Loading**: CSV files with OHLCV data (columns: trade_date, open, close, high, low, volume)
2. **Preprocessing**: Two approaches available:
   - Blog-style: MinMaxScaler + lag features (preferred)
   - Original: Sliding window with return conversion
3. **Model Training**: Supports both LSTM and Transformer with early stopping
4. **Evaluation**: Financial metrics and prediction accuracy
5. **Future Prediction**: 20-day ahead forecasting with multiple methods
6. **Visualization**: Training history, predictions, financial metrics

### Key Features

#### Model Management
- **Model Registry**: Dynamic model registration and creation
- **Multi-Model Support**: Easy switching between LSTM and Transformer
- **Configuration-Driven**: YAML config with model-specific parameters

#### Output Organization
- **Timestamped Outputs**: Each run creates a unique timestamped directory
- **Comprehensive Artifacts**: Models, predictions, visualizations, metadata
- **Reproducibility**: Random seeds and configuration preservation

#### Data Handling
- **Flexible Preprocessing**: Choose between blog-style or original methods
- **Multiple Data Files**: Support for different stock datasets
- **Sequence Generation**: Sliding window for time series preparation

## Configuration System

### YAML Configuration Structure
```yaml
model:
  type: "LSTM"  # or "Transformer"
  lstm:         # LSTM-specific params
    hidden_size1: 50
    hidden_size2: 64
    # ...
  transformer:  # Transformer-specific params
    d_model: 128
    nhead: 8
    # ...

training:
  sequence_length: 60
  batch_size: 32
  learning_rate: 0.001
  epochs: 100

data:
  file: "data/000001_daily_qfq_8y.csv"
  use_blog_style: true
  split_ratio: 0.9
```

### Default Model Parameters

#### LSTM Architecture
- Dual-layer LSTM (hidden_size1=50, hidden_size2=64)
- Three fully connected layers (fc1=32, fc2=16, output=1)
- Recommended learning rate: 0.004

#### Transformer Architecture
- d_model=128, nhead=8, num_encoder_layers=6
- Recommended learning rate: 0.001

## Important Implementation Details

### Blog-Style vs Original Preprocessing
The blog-style method (`use_blog_style: true`) is the recommended approach:
- Uses MinMaxScaler for price normalization
- Creates lag features for sequence input
- Direct price prediction (no return conversion needed)

The original method uses:
- Return conversion and sliding window
- More complex inverse transformation
- Less stable training dynamics

### Prediction Methods
- **Recursive**: Uses predictions as input for next steps
- **Direct**: Independent predictions for each future step
- **Hybrid**: Combines both approaches (default)

### Output Structure
Each run creates a timestamped directory: `output/{stock_code}_{model_type}_{timestamp}/`
- Model file: `{model_type}_stock_model.pth`
- Predictions: `{model_type}_predictions.csv`
- Config: `config.json`
- Training history: `training_history.json`
- Visualizations: Multiple PNG files

## Development Guidelines

### Adding New Models
1. Create model class inheriting from `BaseStockModel`
2. Create trainer class inheriting from `BaseModelTrainer`
3. Register in `model_registry.py`
4. Add configuration parameters to `config.yaml`

### Working with Data
- Data files should be in `data/` directory
- Required columns: `trade_date`, `close`, `open`, `high`, `low`, `volume`
- Date format should be convertible to datetime

### Model Training Best Practices
- Use blog-style preprocessing for better stability
- Monitor validation loss for early stopping
- Adjust learning rates based on model type (LSTM: 0.004, Transformer: 0.001)
- Use GPU when available (CUDA)

### Configuration Management
- Use YAML files for persistent configuration
- Override with command-line arguments when needed
- Configurations are auto-saved with each run for reproducibility