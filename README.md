# Multi-Model Stock Prediction System

A comprehensive deep learning-based stock price prediction and quantitative analysis system supporting both LSTM and Transformer architectures for accurate stock price forecasting and financial metrics evaluation.

## 📋 Project Overview

This project implements a state-of-the-art multi-model stock prediction system with the following core features:

- **Multi-Model Architecture**: Support for both LSTM and Transformer neural networks
- **Data Preprocessing**: Advanced data loading, cleaning, and standardization with blog-style MinMaxScaler
- **Intelligent Training**: Complete workflow with early stopping, learning rate scheduling, and model validation
- **Financial Metrics**: Professional financial evaluation including Sharpe ratio, Sortino ratio, Calmar ratio, etc.
- **Rich Visualization**: Comprehensive charts for training history, predictions, and model analysis
- **Future Prediction**: Multiple prediction strategies (recursive, direct, hybrid) for robust forecasting

## 📊 Data Description

### Supported Data Files
- `data/000001_daily_qfq_8y.csv` - Ping An Bank (000001) 8-year daily data
- `data/000063_daily_qfq_8y.csv` - ZTE Corporation (000063) 8-year daily data
- `data/600031_daily_qfq_8y.csv` - Sany Heavy Industry (600031) 8-year daily data
- `data/600519_daily_qfq_8y.csv` - Kweichow Moutai (600519) 8-year daily data
- `data/601857_daily_qfq_8y.csv` - PetroChina (601857) 8-year daily data

### Data Schema
- `trade_date`: Trading date
- `open`: Opening price
- `close`: Closing price (primary prediction target)
- `high`: Highest price
- `low`: Lowest price
- `volume`: Trading volume

### Data Coverage
- **Historical Period**: 2017-10-13 to 2025-09-10
- **Prediction Horizon**: 20 trading days beyond the latest data
- **Update Frequency**: Daily trading data

## 🚀 Quick Start

### Environment Requirements
- Python 3.13+
- PyTorch 2.0+
- CUDA-enabled GPU recommended for faster training

### Install Dependencies
```bash
# Using uv package manager (recommended)
uv add pandas numpy scikit-learn torch matplotlib seaborn scipy loguru pyyaml

# Or using pip
pip install pandas numpy scikit-learn torch matplotlib seaborn scipy loguru pyyaml
```

### Running the System

#### 1. List Available Models
```bash
uv run python main.py --list-models
```

#### 2. Complete Pipeline (Recommended)
```bash
# LSTM Model (default)
uv run python main.py --mode full --model LSTM

# Transformer Model
uv run python main.py --mode full --model Transformer
```

#### 3. Training Only
```bash
uv run python main.py --mode train --model LSTM
uv run python main.py --mode train --model Transformer
```

#### 4. Prediction Only (requires trained model)
```bash
uv run python main.py --mode predict --model LSTM
uv run python main.py --mode predict --model Transformer
```

#### 5. Custom Data and Configuration
```bash
uv run python main.py --model LSTM --data data/600519_daily_qfq_8y.csv --config config.yaml
```

## 📁 Project Architecture

```
LSTM_predict/
├── src/                           # Source code directory
│   └── stock_predict/             # Main package
│       ├── __init__.py           # Package initialization
│       ├── core.py               # Multi-model coordinator
│       ├── model_registry.py     # Model factory and registry
│       ├── base_models.py        # Abstract base classes
│       ├── lstm_model.py         # LSTM model and trainer
│       ├── transformer_model.py  # Transformer model and trainer
│       ├── data_preprocessor.py  # Advanced data preprocessing
│       ├── prediction.py         # Multi-strategy prediction
│       ├── evaluation.py         # Financial metrics evaluation
│       ├── visualization.py      # Rich visualization suite
│       ├── config_loader.py      # YAML configuration management
│       └── config_multi.py       # Multi-model configuration
├── data/                         # Data files directory
│   ├── 000001_daily_qfq_8y.csv  # Sample stock data
│   └── [other stock files]
├── config.yaml                   # YAML configuration file
├── main.py                      # CLI entry point
├── pyproject.toml               # Python package configuration
└── README.md                    # Project documentation
```

## 🎯 Model Architectures

### LSTM Network Structure
- **Input Layer**: Sequence length 60 days, feature dimension 1 (closing price)
- **Architecture**: Dual-layer LSTM with three fully connected layers
  - LSTM1: 50 hidden units
  - LSTM2: 64 hidden units
  - FC1: 32 units, FC2: 16 units, Output: 1 unit
- **Regularization**: Configurable dropout (default: 0.0)
- **Recommended Learning Rate**: 0.004

### Transformer Network Structure
- **Input Layer**: Sequence length 60 days, feature dimension 1
- **Architecture**: Standard encoder-only Transformer
  - Model Dimension: 128 (configurable)
  - Attention Heads: 8
  - Encoder Layers: 6
  - Feedforward Dimension: 512
- **Positional Encoding**: Sinusoidal encoding for sequence position
- **Regularization**: Dropout (default: 0.1)
- **Recommended Learning Rate**: 0.001 with adaptive scheduling

## ⚙️ Configuration System

### YAML Configuration (`config.yaml`)
```yaml
# Model selection and parameters
model:
  type: "LSTM"  # or "Transformer"
  lstm:
    hidden_size1: 50
    hidden_size2: 64
    fc1_size: 32
    fc2_size: 16
    dropout: 0.0
  transformer:
    d_model: 128
    nhead: 8
    num_encoder_layers: 6
    dim_feedforward: 512
    dropout: 0.1

# Training parameters
training:
  sequence_length: 60
  batch_size: 32
  learning_rate: 0.0002
  epochs: 100
  early_stopping_patience: 30

# Data configuration
data:
  file: "data/000001_daily_qfq_8y.csv"
  use_blog_style: true
  split_ratio: 0.9

# Prediction settings
prediction:
  days: 20
  risk_free_rate: 0.03
```

### Output Organization
The system creates timestamped output directories for each run:
```
output/{stock_code}_{model_type}_{timestamp}/
├── {model_type}_stock_model.pth      # Trained model
├── {model_type}_predictions.csv      # Future predictions
├── config.json                       # Run configuration
├── training_history.json             # Training metrics
├── {model_type}_report_training_history.png
├── {model_type}_report_price_trend_with_test.png
├── training_fit.png
└── training_set_comparison.png
```

## 📈 Financial Metrics & Evaluation

### Prediction Accuracy Metrics
- **MSE**: Mean Squared Error
- **RMSE**: Root Mean Squared Error
- **MAE**: Mean Absolute Error
- **MAPE**: Mean Absolute Percentage Error
- **Directional Accuracy**: Price direction prediction accuracy

### Risk-Return Analysis
- **Sharpe Ratio**: Risk-adjusted return performance
- **Sortino Ratio**: Downside risk-adjusted returns
- **Calmar Ratio**: Annual return / maximum drawdown
- **Maximum Drawdown**: Peak-to-trough decline
- **Annualized Volatility**: Return standard deviation

## 🔮 Prediction Strategies

### Multiple Prediction Methods
1. **Recursive**: Traditional step-by-step prediction with error accumulation
2. **Direct**: Always predict from original sequence (no error accumulation)
3. **Hybrid**: Weighted combination balancing both approaches (recommended)

### Usage Examples
```python
from src.stock_predict import MultiModelStockPredictor

# LSTM with hybrid prediction
predictor = MultiModelStockPredictor({
    'model_type': 'LSTM',
    'data_file': 'data/600519_daily_qfq_8y.csv',
    'learning_rate': 0.004
})
predictor.run_full_pipeline()

# Transformer with custom config
transformer_config = {
    'model_type': 'Transformer',
    'd_model': 256,
    'nhead': 8,
    'num_encoder_layers': 8,
    'learning_rate': 0.001
}
predictor = MultiModelStockPredictor(transformer_config)
predictor.run_full_pipeline()
```

## 🎨 Advanced Features

### Model Registry System
- Dynamic model registration and creation
- Extensible architecture for adding new models
- Factory pattern for model instantiation

### Smart Learning Rate Scheduling
- **LSTM**: Fixed learning rate (configurable)
- **Transformer**: Adaptive `ReduceLROnPlateau` with 10-epoch patience
- Automatic learning rate adjustment based on validation performance

### Comprehensive Visualization
- Training history curves (train/val loss)
- Enhanced price trend with test data comparison
- Training fit analysis and displacement detection
- Historical + test + future prediction integration

### Timestamped Outputs
- Each run creates a unique timestamped directory
- Complete reproducibility with saved configurations
- Organized artifact management

## 🔧 Performance Optimization

### Model-Specific Tuning
- **LSTM**: Works well with higher learning rates (0.004)
- **Transformer**: Benefits from adaptive scheduling and lower initial rates
- **Memory Management**: Gradient clipping and efficient data loading

### Data Processing Efficiency
- Blog-style MinMaxScaler with lag features
- Optimized sequence generation
- Smart train/validation/test splitting

## ⚠️ Important Disclaimers

1. **Educational Purpose**: This system is for research and education only, not investment advice
2. **Market Uncertainty**: Stock prediction involves inherent uncertainty and risk
3. **Data Quality**: Results depend on high-quality, complete historical data
4. **Model Limitations**: No model can perfectly predict market movements
5. **Risk Management**: Always use proper risk management in real trading

## 🔍 Troubleshooting Guide

### Common Issues
1. **Memory Issues**: Reduce `batch_size` or use CPU training
2. **Slow Convergence**: Adjust learning rate or model architecture
3. **Poor Predictions**: Try different model types or prediction strategies
4. **Data Loading**: Verify file paths and CSV format compliance

### Performance Tips
- Use GPU acceleration when available
- Experiment with different sequence lengths
- Try hybrid prediction for better stability
- Monitor training loss for early stopping optimization

## 📝 Development History

- **2025-10-19**: Major multi-model architecture upgrade
  - Implemented Transformer model support
  - Added model registry and factory pattern
  - Enhanced configuration system with YAML support
  - Improved visualization and reporting
  - Optimized learning rate scheduling for both models

- **2025-10-16**: Complete system refactoring
  - Restructured to professional Python package layout
  - Added comprehensive financial metrics evaluation
  - Implemented blog-style data preprocessing
  - Created timestamped output management

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🏗️ Adding New Models

The system uses a **Model Registry Pattern** that makes it easy to add new model types. Here's how to register a new model:

### 📋 Registration Process

1. **Create Model File**: Create a new file in `src/stock_predict/` (e.g., `gru_model.py`)
2. **Implement Required Classes**:
   - Model class inheriting from `BaseStockModel`
   - Trainer class inheriting from `BaseModelTrainer`
3. **Register in Registry**: Add import and registration in `model_registry.py`
4. **Update Package Exports**: Add to `__init__.py` if needed

### 🔧 Required Methods

**Model Class must implement:**
- `forward(self, x: torch.Tensor) -> torch.Tensor`
- `get_model_params(self) -> Dict[str, Any]`
- `from_params(cls, params: Dict[str, Any]) -> BaseStockModel`

**Trainer Class must implement:**
- `train(self, train_loader, val_loader, epochs, early_stopping_patience)`
- `predict(self, data_loader) -> Tuple[torch.Tensor, torch.Tensor]`
- `save_model(self, path, train_losses, val_losses, model_params, config)`
- `load_model(cls, path, device)`

### 🚀 Registration Steps

1. **Import your classes** in `model_registry.py`
2. **Add registration** in `register_all_models()` function:
   ```python
   ModelRegistry.register_model(
       name="YourModel",
       model_class=YourModelClass,
       trainer_class=YourTrainerClass
   )
   ```
3. **Update package exports** (optional) in `__init__.py`

### 🎯 Usage After Registration

Your new model is immediately available:
```bash
# List models (includes yours)
uv run python main.py --list-models

# Use your model
uv run python main.py --model YourModel --mode full
```

### ✅ Benefits

- **Zero Configuration Changes**: Models work with existing CLI
- **Automatic Integration**: Full pipeline support (training, prediction, evaluation)
- **Consistent Interface**: Same methods as existing models
- **Extensible Architecture**: Easy to add more models

---

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request or open an Issue for discussion.

### Development Setup
```bash
# Clone repository
git clone <repository-url>
cd LSTM_predict

# Install development dependencies
uv add --dev pytest black flake8 mypy

# Run tests
pytest

# Code formatting
black src/ main.py

# Type checking
mypy src/
```

---

**Built with ❤️ for quantitative finance research and education**
