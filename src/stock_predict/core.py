"""
Core Module - Multi-Model Support

Main entry point for the stock prediction system supporting multiple model types.
Coordinates all components and provides high-level interface.
"""

import os
import sys
import torch
import numpy as np
import pandas as pd
from datetime import datetime
import argparse
import json
import shutil
from typing import Tuple, Dict, Any, Optional

from .data_preprocessor import DataPreprocessor, BlogStyleDataPreprocessor
from .evaluation import FinancialMetrics
from .visualization import StockVisualizer
from .prediction import Predictor
from .model_registry import get_available_models, create_model, create_trainer


class MultiModelStockPredictor:
    """Main class for multi-model stock price prediction system."""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the multi-model stock predictor.

        Args:
            config: Configuration parameters dictionary
        """
        self.config = config or self.get_default_config()

        # Debug: Print configuration
        print(f"🔧 Configuration loaded:")
        for key, value in self.config.items():
            print(f"   {key}: {value}")
        print()

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Set random seeds for reproducibility
        torch.manual_seed(self.config.get('random_seed', 42))
        np.random.seed(self.config.get('random_seed', 42))

        print(f"🚀 Multi-Model Stock Prediction System Initialized")
        print(f"   Device: {self.device}")
        print(f"   Model Type: {self.config.get('model_type', 'LSTM')}")
        print(f"   Data file: {self.config.get('data_file', 'data/000001_daily_qfq_8y.csv')}")

        # Initialize components
        self.preprocessor: Optional[DataPreprocessor] = None
        self.model = None
        self.trainer = None
        self.predictor: Optional[Predictor] = None

        # Create timestamp output directory
        self.create_timestamp_output_dir()

    def create_timestamp_output_dir(self):
        """创建带时间戳的输出目录"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        base_output_dir = self.config.get('output_dir', 'output')

        # 获取股票代码和模型类型用于目录命名
        data_file = self.config.get('data_file', 'data/000001_daily_qfq_8y.csv')
        stock_code = os.path.basename(data_file).split('_')[0]
        model_type = self.config.get('model_type', 'LSTM')

        # 创建时间戳目录
        self.timestamp_dir = os.path.join(base_output_dir, f"{stock_code}_{model_type}_{timestamp}")
        os.makedirs(self.timestamp_dir, exist_ok=True)

        print(f"📁 输出目录: {self.timestamp_dir}")

        # 保存配置到时间戳目录
        config_path = os.path.join(self.timestamp_dir, "config.json")
        with open(config_path, 'w', encoding='utf-8') as f:
            json.dump(self.config, f, indent=2, ensure_ascii=False, default=str)

        print(f"📋 配置已保存: {config_path}")

    def get_default_config(self) -> Dict[str, Any]:
        """Get default configuration parameters."""
        return {
            'model_type': 'LSTM',     # 模型类型: LSTM 或 Transformer
            'data_file': 'data/000001_daily_qfq_8y.csv',
            'sequence_length': 60,
            'batch_size': 32,
            'learning_rate': 0.004,
            'epochs': 100,
            'early_stopping_patience': 20,
            'random_seed': 42,
            'risk_free_rate': 0.03,
            'prediction_days': 20,
            'use_blog_style': True,
            'split_ratio': 0.9,
            'output_dir': 'output',
            # LSTM 特定参数
            'hidden_size1': 50,
            'hidden_size2': 64,
            'fc1_size': 32,
            'fc2_size': 16,
            'dropout': 0.0,
            # Transformer 特定参数
            'd_model': 128,
            'nhead': 8,
            'num_encoder_layers': 6,
            'dim_feedforward': 512,
            'transformer_dropout': 0.1
        }

    def prepare_data(self) -> Tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader, torch.utils.data.DataLoader, pd.DataFrame]:
        """
        Load and preprocess data.

        Returns:
            Tuple of (train_loader, val_loader, test_loader, data)
        """
        print("\n📊 Starting data preprocessing...")

        # Ensure we have default configuration
        if not self.config:
            self.config = self.get_default_config()

        # Choose data preprocessing method
        use_blog_style = self.config.get('use_blog_style', True)

        if use_blog_style:
            print("   Using blog-style preprocessing (MinMaxScaler + lag features)")
            self.preprocessor = BlogStyleDataPreprocessor(
                n_steps=self.config.get('sequence_length', 60)
            )

            # Load and preprocess data blog-style
            scaler, X_train, X_test, y_train, y_test = self.preprocessor.get_dataset(
                self.config.get('data_file', 'data/000001_daily_qfq_8y.csv'),
                lookback=self.config.get('sequence_length', 60),
                split_ratio=self.config.get('split_ratio', 0.9)
            )

            # Store scaler for inverse transform
            self.scaler = scaler

            # Create datasets manually
            from torch.utils.data import TensorDataset, DataLoader

            train_dataset = TensorDataset(X_train, y_train)
            test_dataset = TensorDataset(X_test, y_test)

            # Split training into train/val
            val_size = len(X_train) // 5  # 20% for validation
            train_size = len(X_train) - val_size

            train_subset, val_subset = torch.utils.data.random_split(
                train_dataset, [train_size, val_size],
                generator=torch.Generator().manual_seed(42)
            )

            train_loader = DataLoader(
                train_subset,
                batch_size=self.config.get('batch_size', 32),
                shuffle=True
            )
            val_loader = DataLoader(
                val_subset,
                batch_size=self.config.get('batch_size', 32),
                shuffle=False
            )
            test_loader = DataLoader(
                test_dataset,
                batch_size=1,
                shuffle=False
            )

            # Create a dummy data DataFrame for compatibility and store in preprocessor
            data = pd.read_csv(self.config.get('data_file', 'data/000001_daily_qfq_8y.csv'))
            if 'trade_date' in data.columns:
                data['date'] = pd.to_datetime(data['trade_date'])
                data.set_index('date', inplace=True)

            # Store data in preprocessor for compatibility
            self.preprocessor.data = data

        else:
            print("   Using original preprocessing (sliding window)")
            self.preprocessor = DataPreprocessor(
                sequence_length=self.config.get('sequence_length', 30)
            )

            # Load data
            data = self.preprocessor.load_data(self.config.get('data_file', 'data/000001_daily_qfq_8y.csv'))

            # Preprocess data
            train_data, test_data = self.preprocessor.preprocess_data()

            # Create data loaders
            train_loader, val_loader, test_loader = self.preprocessor.create_datasets(
                batch_size=self.config.get('batch_size', 32)
            )

        print(f"✅ Data preprocessing completed")
        print(f"   Training batches: {len(train_loader)}")
        print(f"   Validation batches: {len(val_loader)}")
        print(f"   Testing batches: {len(test_loader)}")

        return train_loader, val_loader, test_loader, data

    def create_model_and_trainer(self):
        """Create model and trainer based on configuration."""
        model_type = self.config.get('model_type', 'LSTM')
        print(f"\n🤖 Creating {model_type} model...")

        if model_type == 'LSTM':
            model_params = {
                'input_size': 1,
                'hidden_size1': self.config.get('hidden_size1', 50),
                'hidden_size2': self.config.get('hidden_size2', 64),
                'fc1_size': self.config.get('fc1_size', 32),
                'fc2_size': self.config.get('fc2_size', 16),
                'dropout': self.config.get('dropout', 0.0)
            }
            print(f"   Model configuration (双层LSTM+三层全连接):")
            print(f"     LSTM1 hidden size: {model_params['hidden_size1']}")
            print(f"     LSTM2 hidden size: {model_params['hidden_size2']}")
            print(f"     FC1 size: {model_params['fc1_size']}")
            print(f"     FC2 size: {model_params['fc2_size']}")
            print(f"     Dropout: {model_params['dropout']}")

        elif model_type == 'Transformer':
            model_params = {
                'input_size': 1,
                'd_model': self.config.get('d_model', 128),
                'nhead': self.config.get('nhead', 8),
                'num_encoder_layers': self.config.get('num_encoder_layers', 6),
                'dim_feedforward': self.config.get('dim_feedforward', 512),
                'dropout': self.config.get('transformer_dropout', 0.1),
                'max_seq_len': self.config.get('sequence_length', 60) + 100
            }
            print(f"   Model configuration (Transformer):")
            print(f"     Model dimension: {model_params['d_model']}")
            print(f"     Number of heads: {model_params['nhead']}")
            print(f"     Encoder layers: {model_params['num_encoder_layers']}")
            print(f"     Feedforward dimension: {model_params['dim_feedforward']}")
            print(f"     Dropout: {model_params['dropout']}")

        else:
            raise ValueError(f"Unknown model type: {model_type}. Available: {get_available_models()}")

        print(f"     Sequence length: {self.config.get('sequence_length', 60)}")

        # Create model
        self.model = create_model(model_type, **model_params)
        self.model.to(self.device)

        print(f"   Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")

        # Create trainer
        self.trainer = create_trainer(model_type, self.model, self.device)

        print(f"✅ {model_type} model and trainer created")

    def train_model(self, train_loader, val_loader) -> Tuple[list, list]:
        """
        Train the model.

        Args:
            train_loader: Training data loader
            val_loader: Validation data loader

        Returns:
            Tuple of (train_losses, val_losses)
        """
        print(f"\n🤖 Training {self.config.get('model_type', 'LSTM')} model...")

        # Create model and trainer if not already created
        if self.model is None or self.trainer is None:
            self.create_model_and_trainer()

        # Train model
        train_losses, val_losses = self.trainer.train(
            train_loader=train_loader,
            val_loader=val_loader,
            epochs=self.config.get('epochs', 100),
            early_stopping_patience=self.config.get('early_stopping_patience', 20),
            learning_rate=self.config.get('learning_rate', 0.004)
        )

        # Save model
        self.save_model(train_losses, val_losses)

        print(f"✅ Model training completed")

        return train_losses, val_losses

    def save_model(self, train_losses: list, val_losses: list):
        """Save the trained model to timestamp directory."""
        # Generate model path in timestamp directory
        model_filename = f'{self.config.get("model_type", "LSTM").lower()}_stock_model.pth'
        model_path = os.path.join(self.timestamp_dir, model_filename)

        model_params = self.model.get_model_params()

        self.trainer.save_model(
            model_path,
            train_losses,
            val_losses,
            model_params,
            self.config
        )

        # 保存训练历史
        history_data = {
            'train_losses': train_losses,
            'val_losses': val_losses,
            'epochs_trained': len(train_losses),
            'best_val_loss': min(val_losses),
            'final_train_loss': train_losses[-1] if train_losses else None,
            'timestamp': datetime.now().isoformat(),
            'model_type': self.config.get('model_type', 'LSTM')
        }

        history_path = os.path.join(self.timestamp_dir, 'training_history.json')
        with open(history_path, 'w', encoding='utf-8') as f:
            json.dump(history_data, f, indent=2)

        print(f"📊 训练历史已保存: {history_path}")

    def load_model(self, model_path: Optional[str] = None) -> Tuple[list, list]:
        """
        Load a trained model.

        Args:
            model_path: Optional model path, if not provided use config path

        Returns:
            Tuple of (train_losses, val_losses)
        """
        if model_path is None:
            # Try to find model in default output directory
            data_file = self.config.get('data_file', 'data/000001_daily_qfq_8y.csv')
            stock_code = os.path.basename(data_file).split('_')[0]
            model_type = self.config.get('model_type', 'LSTM').lower()
            model_filename = f'{model_type}_stock_model_{stock_code}.pth'
            model_path = os.path.join('output', model_filename)

            # If not found, try to find the latest timestamp directory
            if not os.path.exists(model_path):
                base_output_dir = self.config.get('output_dir', 'output')
                # 搜索匹配的目录，尝试不同的命名方式
                pattern_title = os.path.join(base_output_dir, f"{stock_code}_{model_type.title()}_*")  # Transformer
                pattern_upper = os.path.join(base_output_dir, f"{stock_code}_{model_type.upper()}_*")  # TRANSFORMER
                pattern_lower = os.path.join(base_output_dir, f"{stock_code}_{model_type}_*")      # transformer
                import glob
                possible_dirs = glob.glob(pattern_title) + glob.glob(pattern_upper) + glob.glob(pattern_lower)

                # 过滤出确实包含模型文件的目录
                valid_dirs = []
                for dir_path in possible_dirs:
                    potential_model_path = os.path.join(dir_path, f'{model_type}_stock_model.pth')
                    if os.path.exists(potential_model_path):
                        valid_dirs.append(dir_path)

                if valid_dirs:
                    # 按修改时间排序，取最新的
                    latest_dir = max(valid_dirs, key=os.path.getmtime)
                    model_path = os.path.join(latest_dir, f'{model_type}_stock_model.pth')
                    print(f"🔍 在最新时间戳目录中找到模型: {model_path}")

        if not os.path.exists(model_path):
            print(f"❌ 未找到训练好的模型文件!")
            print(f"🔍 查找路径: {model_path}")
            print(f"💡 请先训练模型或检查模型文件是否存在:")
            print(f"   训练命令: uv run python main_multi.py --model {model_type.upper()} --mode train")
            print(f"   或使用完整流程: uv run python main_multi.py --model {model_type.upper()} --mode full")
            raise FileNotFoundError(f"模型文件未找到: {model_path}\n请先训练模型: uv run python main_multi.py --model {model_type.upper()} --mode train")

        model_type = self.config.get('model_type', 'LSTM')
        trainer_class = self.trainer.__class__
        self.model, train_losses, val_losses, model_params, loaded_config = trainer_class.load_model(
            model_path, self.device
        )

        print(f"✅ {model_type} 模型已加载: {model_path}")

        # 如果有保存的配置，可以显示相关信息
        if loaded_config:
            print(f"📋 模型配置: epochs={loaded_config.get('epochs', 'N/A')}, "
                  f"batch_size={loaded_config.get('batch_size', 'N/A')}")

        return train_losses, val_losses

    def evaluate_model(self, test_loader) -> Tuple[Dict[str, Any], np.ndarray, np.ndarray]:
        """
        Evaluate the model on test data.

        Args:
            test_loader: Test data loader

        Returns:
            Tuple of (metrics, predictions, actuals)
        """
        print(f"\n📈 Evaluating {self.config.get('model_type', 'LSTM')} model performance...")

        # Make predictions on test set
        predictions, actuals = self.trainer.predict(test_loader)

        if isinstance(self.preprocessor, BlogStyleDataPreprocessor):
            # 博客风格的反归一化
            predicted_prices = self.preprocessor.inverse_transform(predictions).flatten()
            actual_prices = self.preprocessor.inverse_transform(actuals).flatten()
            predictions_orig = predicted_prices.reshape(-1, 1)
            actuals_orig = actual_prices.reshape(-1, 1)
        else:
            # 原始方法的反归一化
            predicted_returns = self.preprocessor.inverse_transform(predictions.flatten()).flatten()
            actual_returns = self.preprocessor.inverse_transform(actuals.flatten()).flatten()

            # Convert returns back to absolute prices for evaluation
            test_start_idx = len(self.preprocessor.original_prices) - len(predicted_returns)
            actual_prices_test = self.preprocessor.original_prices[test_start_idx:]

            predictions_orig = self.preprocessor.convert_returns_to_prices(predicted_returns, actual_prices_test[0])
            actuals_orig = actual_prices_test

        # Calculate financial metrics
        metrics_calculator = FinancialMetrics(
            risk_free_rate=self.config.get('risk_free_rate', 0.03)
        )

        metrics = metrics_calculator.evaluate_predictions(
            actuals_orig, predictions_orig
        )

        # Print evaluation report
        metrics_calculator.print_evaluation_report(metrics)

        return metrics, predictions_orig, actuals_orig

    def predict_future(self, num_days: Optional[int] = None, method: str = 'hybrid') -> Tuple[np.ndarray, list]:
        """
        Predict future stock prices - 支持多种预测方法

        Args:
            num_days: Number of days to predict (default from config)
            method: 'recursive', 'direct', or 'hybrid' prediction method

        Returns:
            Tuple of (predictions, prediction_dates)
        """
        if num_days is None:
            num_days = self.config.get('prediction_days', 20)

        print(f"\n🔮 Predicting future {num_days} days stock prices using {method} method...")

        # Ensure model is loaded
        if self.model is None:
            self.load_model()

        # Create predictor
        self.predictor = Predictor(self.model, self.preprocessor, self.device)

        # Get last sequence
        last_sequence = self.preprocessor.get_latest_sequence()

        # Predict
        predictions, prediction_dates = self.predictor.predict_with_dates(
            last_sequence, num_days, method=method
        )

        print(f"✅ Prediction completed")

        return predictions, prediction_dates

    def evaluate_on_training_data(self) -> Tuple[np.ndarray, np.ndarray, list]:
        """
        Evaluate model on training data to see fitting performance.
        Uses the same sliding window approach as training.

        Returns:
            Tuple of (train_predictions, train_actuals, train_dates)
        """
        print(f"\n📊 Evaluating model on training data...")

        if isinstance(self.preprocessor, BlogStyleDataPreprocessor):
            # 博客风格：使用已有的测试集的一部分作为验证评估
            from torch.utils.data import TensorDataset, DataLoader

            # 获取部分训练数据用于验证评估，使用与训练时相同的数据分割比例
            scaler, X_train_full, X_test_full, y_train_full, y_test_full = self.preprocessor.get_dataset(
                self.config.get('data_file', 'data/000001_daily_qfq_8y.csv'),
                lookback=self.config.get('sequence_length', 60),
                split_ratio=self.config.get('split_ratio', 0.9)  # 使用与训练时相同的分割比例
            )
            # 使用测试集的一部分作为验证评估
            X_val_partial = X_test_full[:min(100, len(X_test_full))]  # 最多100个样本
            y_val_partial = y_test_full[:min(100, len(y_test_full))]

            val_dataset = TensorDataset(X_val_partial, y_val_partial)
            val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)

            # 预测
            predictions, actuals = self.trainer.predict(val_loader)

            # 反归一化
            predicted_prices = self.preprocessor.inverse_transform(predictions).flatten()
            actual_prices = self.preprocessor.inverse_transform(actuals).flatten()
            predictions_orig = predicted_prices.reshape(-1, 1)
            actual_prices_val = actual_prices.reshape(-1, 1)

            # 获取对应的日期（使用数据末尾的日期）
            train_dates = self.preprocessor.data.index[-len(predictions_orig):]

        else:
            # 原始方法
            from sklearn.model_selection import train_test_split
            train_data, val_data = train_test_split(
                self.preprocessor.train_data,
                test_size=0.2,
                shuffle=False
            )

            from .data_preprocessor import StockDataset
            val_dataset = StockDataset(val_data, self.preprocessor.sequence_length)
            val_loader = torch.utils.data.DataLoader(
                val_dataset,
                batch_size=1,
                shuffle=False
            )

            predictions, actuals = self.trainer.predict(val_loader)

            predicted_returns = self.preprocessor.inverse_transform(predictions.flatten()).flatten()
            actual_returns = self.preprocessor.inverse_transform(actuals.flatten()).flatten()

            val_start_idx = len(self.preprocessor.train_data) - len(val_data)
            actual_prices_val = self.preprocessor.original_prices[val_start_idx:val_start_idx+len(predicted_returns)]
            predictions_orig = self.preprocessor.convert_returns_to_prices(predicted_returns, actual_prices_val[0])

            train_dates = self.preprocessor.data.index[val_start_idx:val_start_idx+len(predictions_orig)]

        print(f"✅ Training data evaluation completed")
        print(f"   Validation samples: {len(predictions_orig)}")
        print(f"   Date range: {train_dates[0]} to {train_dates[-1]}")

        return predictions_orig, actual_prices_val, train_dates

    def create_visualizations(self, data, predictions, prediction_dates,
                            train_losses, val_losses, metrics, test_actuals, test_predictions,
                            train_predictions=None, train_actuals=None, train_dates=None):
        """Create visualization reports in timestamp directory."""
        print(f"\n📊 生成可视化报告...")

        visualizer = StockVisualizer()

        # Get historical data for visualization
        historical_prices = data['close'].values
        historical_dates = data.index

        # Generate unique prefix based on timestamp and model type
        model_type = self.config.get('model_type', 'LSTM').lower()
        save_prefix = f'{model_type}_report'

        # 确保测试数据的日期与价格数量匹配
        if isinstance(self.preprocessor, BlogStyleDataPreprocessor):
            test_dates_viz = self.preprocessor.data.index[-len(test_actuals):]
        else:
            test_dates_viz = self.preprocessor.data.index[-120:]
            if len(test_dates_viz) > len(test_actuals):
                test_dates_viz = test_dates_viz[-len(test_actuals):]

        def create_timestamp_save_path(original_path):
            """为可视化文件创建时间戳路径"""
            if original_path.startswith('output/'):
                return os.path.join(self.timestamp_dir, original_path[7:])
            return os.path.join(self.timestamp_dir, original_path)

        # 保存训练历史图
        history_path = create_timestamp_save_path(f'{save_prefix}_training_history.png')
        visualizer.plot_training_history(
            train_losses, val_losses,
            save_path=history_path
        )

        # 修改可视化模块的保存路径
        import types
        original_plot_training_history = visualizer.plot_training_history
        visualizer.plot_training_history = lambda *args, **kwargs: original_plot_training_history(
            *args, **{k: create_timestamp_save_path(v) if k == 'save_path' else v for k, v in kwargs.items()}
        )

        visualizer.create_comprehensive_report_with_test(
            historical_prices=historical_prices,
            historical_dates=historical_dates,
            test_prices=test_actuals.flatten(),
            test_dates=test_dates_viz,
            test_predictions=test_predictions.flatten(),
            predicted_prices=predictions,
            prediction_dates=prediction_dates,
            train_losses=train_losses,
            val_losses=val_losses,
            metrics=metrics,
            save_prefix=save_prefix
        )

        # Create training fit visualization if data is available
        if train_predictions is not None and train_actuals is not None and train_dates is not None:
            training_fit_path = os.path.join(self.timestamp_dir, 'training_fit.png')
            visualizer.plot_training_fit(
                train_predictions=train_predictions,
                train_actuals=train_actuals,
                train_dates=train_dates,
                save_path=training_fit_path
            )

            # Create comprehensive training set comparison to analyze displacement
            training_comparison_path = os.path.join(self.timestamp_dir, 'training_set_comparison.png')
            model_type = self.config.get('model_type', 'Model')
            visualizer.plot_training_set_comparison(
                train_predictions=train_predictions,
                train_actuals=train_actuals,
                train_dates=train_dates,
                model_type=model_type,
                save_path=training_comparison_path
            )

  
        print(f"✅ 可视化报告已保存到: {self.timestamp_dir}")

    def save_predictions(self, predictions, prediction_dates, filename: str = None):
        """Save prediction results to timestamp directory."""
        if self.predictor is None:
            self.predictor = Predictor(self.model, self.preprocessor, self.device)

        # Generate filename in timestamp directory
        if filename is None:
            model_type = self.config.get('model_type', 'LSTM').lower()
            filename = f'{model_type}_predictions.csv'

        # 确保文件保存在时间戳目录中
        prediction_path = os.path.join(self.timestamp_dir, filename)

        # 创建预测结果DataFrame
        results_df = pd.DataFrame({
            'date': prediction_dates,
            'predicted_price': predictions.flatten()
        })

        # 保存到时间戳目录
        results_df.to_csv(prediction_path, index=False)
        print(f"📄 预测结果已保存: {prediction_path}")

        # 保存预测元数据
        prediction_metadata = {
            'model_type': self.config.get('model_type', 'LSTM'),
            'prediction_count': len(predictions),
            'prediction_dates': [date.isoformat() for date in prediction_dates],
            'price_range': [float(predictions.min()), float(predictions.max())],
            'price_change': float(predictions[-1] - predictions[0]),
            'price_change_percent': float((predictions[-1] - predictions[0]) / predictions[0] * 100),
            'timestamp': datetime.now().isoformat(),
            'config_used': self.config
        }

        metadata_path = os.path.join(self.timestamp_dir, 'prediction_metadata.json')
        with open(metadata_path, 'w', encoding='utf-8') as f:
            json.dump(prediction_metadata, f, indent=2, default=str)

        print(f"📋 预测元数据已保存: {metadata_path}")

        return results_df

    def run_full_pipeline(self):
        """Run the complete training and prediction pipeline."""
        print("=" * 80)
        print(f"🎯 Multi-Model Stock Prediction System - Full Pipeline")
        print(f"   Model Type: {self.config.get('model_type', 'LSTM')}")
        print("=" * 80)

        # 1. Data preprocessing
        train_loader, val_loader, test_loader, data = self.prepare_data()

        # 2. Model training
        train_losses, val_losses = self.train_model(train_loader, val_loader)

        # 3. Model evaluation
        metrics, test_predictions, test_actuals = self.evaluate_model(test_loader)

        # 4. Future prediction
        predictions, prediction_dates = self.predict_future()

        # 5. Save predictions
        self.save_predictions(predictions, prediction_dates)

        # 5.5. Training data evaluation for fitting visualization
        train_predictions, train_actuals, train_dates = self.evaluate_on_training_data()

        # 6. Generate visualization reports
        self.create_visualizations(
            data, predictions, prediction_dates, train_losses, val_losses, metrics, test_actuals, test_predictions,
            train_predictions, train_actuals, train_dates
        )

        # Get stock code and model type for file names
        data_file = self.config.get('data_file', 'data/000001_daily_qfq_8y.csv')
        stock_code = os.path.basename(data_file).split('_')[0]
        model_type = self.config.get('model_type', 'LSTM').lower()

        print(f"\n🎉 完整流程执行成功!")
        print(f"   股票代码: {stock_code}")
        print(f"   模型类型: {model_type}")
        print(f"   输出目录: {self.timestamp_dir}")
        print(f"   配置文件: {os.path.join(self.timestamp_dir, 'config.json')}")
        print(f"   模型文件: {os.path.join(self.timestamp_dir, f'{model_type}_stock_model.pth')}")
        print(f"   预测结果: {os.path.join(self.timestamp_dir, f'{model_type}_predictions.csv')}")
        print(f"   可视化图表: {self.timestamp_dir}/*.png")
        print(f"   训练历史: {os.path.join(self.timestamp_dir, 'training_history.json')}")

    def run_training_only(self):
        """Run training only."""
        print("=" * 80)
        print(f"🎯 Multi-Model Stock Prediction System - Training Only")
        print(f"   Model Type: {self.config.get('model_type', 'LSTM')}")
        print("=" * 80)

        # 1. Data preprocessing
        train_loader, val_loader, _, _ = self.prepare_data()

        # 2. Model training
        self.train_model(train_loader, val_loader)

        model_type = self.config.get('model_type', 'LSTM').lower()
        print(f"\n🎉 训练完成!")
        print(f"   模型类型: {model_type}")
        print(f"   输出目录: {self.timestamp_dir}")
        print(f"   模型文件: {os.path.join(self.timestamp_dir, f'{model_type}_stock_model.pth')}")
        print(f"   训练历史: {os.path.join(self.timestamp_dir, 'training_history.json')}")

    def run_prediction_only(self):
        """Run prediction only (requires trained model)."""
        print("=" * 80)
        print(f"🔮 Multi-Model Stock Prediction System - Prediction Only")
        print(f"   Model Type: {self.config.get('model_type', 'LSTM')}")
        print("=" * 80)

        # 1. Data preprocessing
        train_loader, val_loader, test_loader, data = self.prepare_data()

        # 2. Create model and trainer
        self.create_model_and_trainer()

        # 3. Load model
        train_losses, val_losses = self.load_model()

        # 4. Model evaluation to get test data
        metrics, test_predictions, test_actuals = self.evaluate_model(test_loader)

        # 5. Training data evaluation for fitting visualization
        train_predictions, train_actuals, train_dates = self.evaluate_on_training_data()

        # 6. Future prediction
        predictions, prediction_dates = self.predict_future()

        # 7. Save predictions
        self.save_predictions(predictions, prediction_dates)

        # 8. Generate visualization reports
        self.create_visualizations(
            data, predictions, prediction_dates, train_losses, val_losses, metrics, test_actuals, test_predictions,
            train_predictions, train_actuals, train_dates
        )

        model_type = self.config.get('model_type', 'LSTM').lower()
        print(f"\n🎉 预测完成!")
        print(f"   模型类型: {model_type}")
        print(f"   输出目录: {self.timestamp_dir}")
        print(f"   预测结果: {os.path.join(self.timestamp_dir, f'{model_type}_predictions.csv')}")
        print(f"   预测元数据: {os.path.join(self.timestamp_dir, 'prediction_metadata.json')}")
        print(f"   可视化图表: {self.timestamp_dir}/*.png")


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description='Multi-Model Stock Price Prediction System')
    parser.add_argument('--mode', choices=['train', 'predict', 'full'], default='full',
                       help='Run mode: train (training only), predict (prediction only), full (complete pipeline)')
    parser.add_argument('--model', choices=get_available_models(), default='LSTM',
                       help=f'Model type: {", ".join(get_available_models())}')
    parser.add_argument('--config', type=str, help='Configuration file path')
    parser.add_argument('--data', type=str, help='Data file path')
    parser.add_argument('--model-path', type=str, help='Model file path')
    parser.add_argument('--output-dir', type=str, default='output', help='Output directory')

    args = parser.parse_args()

    # Create configuration
    config = {}
    if args.data:
        config['data_file'] = args.data
    if args.model_path:
        config['model_path'] = args.model_path
    if args.output_dir:
        config['output_dir'] = args.output_dir
    if args.model:
        config['model_type'] = args.model

    # Create predictor
    predictor = MultiModelStockPredictor(config)

    # Run based on mode
    if args.mode == 'train':
        predictor.run_training_only()
    elif args.mode == 'predict':
        predictor.run_prediction_only()
    else:  # full
        predictor.run_full_pipeline()


if __name__ == "__main__":
    main()