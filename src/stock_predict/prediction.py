"""
Prediction Module

Handles future stock price prediction using trained LSTM models.
"""

import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from datetime import datetime, timedelta
from typing import Tuple, List, Optional
from .data_preprocessor import DataPreprocessor, BlogStyleDataPreprocessor
from .base_models import BaseStockModel
from .lstm_model import LSTMStockModel
from .transformer_model import TransformerStockModel


class Predictor:
    """Handles stock price prediction using trained models."""

    def __init__(self, model: BaseStockModel, preprocessor: DataPreprocessor,
                 device: torch.device):
        """
        Initialize the predictor.

        Args:
            model: Trained LSTM model
            preprocessor: Data preprocessor with fitted scaler
            device: Device for computation
        """
        self.model = model
        self.preprocessor = preprocessor
        self.device = device
        self.model.eval()

    def predict_future_days(self, last_sequence: np.ndarray,
                           num_days: int = 20, method: str = 'recursive') -> np.ndarray:
        """
        Predict stock returns for future days - 改进版支持多种预测方法

        Args:
            last_sequence: Last known sequence of returns
            num_days: Number of days to predict
            method: 'recursive' (递归) or 'direct' (直接预测) or 'hybrid' (混合)

        Returns:
            Array of predicted returns (scaled)
        """
        if method == 'recursive':
            return self._predict_recursive(last_sequence, num_days)
        elif method == 'direct':
            return self._predict_direct(last_sequence, num_days)
        elif method == 'hybrid':
            return self._predict_hybrid(last_sequence, num_days)
        else:
            raise ValueError(f"Unknown prediction method: {method}")

    def _predict_recursive(self, last_sequence: np.ndarray, num_days: int) -> np.ndarray:
        """
        递归预测 - 传统方法但加入噪声避免直线化
        """
        predictions = []
        current_sequence = last_sequence.copy()
        sequence_length = len(last_sequence)

        self.model.eval()
        with torch.no_grad():
            for day in range(num_days):
                # Prepare input data
                input_seq = torch.FloatTensor(
                    current_sequence[-sequence_length:]
                ).view(1, sequence_length, 1).to(self.device)

                # Predict next return
                next_pred = self.model(input_seq)

                # 关键改进: 添加合理的小幅随机噪声，避免完全直线化
                pred_value = next_pred.cpu().numpy()[0, 0]
                noise = np.random.normal(0, 0.001)  # 添加微小噪声
                pred_value = pred_value + noise

                # Save prediction
                predictions.append(pred_value)

                # Update sequence with noisy prediction
                current_sequence = np.append(current_sequence, pred_value)

        return np.array(predictions)

    def _predict_direct(self, last_sequence: np.ndarray, num_days: int) -> np.ndarray:
        """
        直接预测 - 每次都使用原始序列，避免递归误差累积
        """
        predictions = []
        original_sequence = last_sequence.flatten()  # 确保是一维数组
        sequence_length = len(last_sequence)

        self.model.eval()
        with torch.no_grad():
            for day in range(num_days):
                # 始终使用原始序列的最后sequence_length个值作为输入
                # 这样避免了递归误差累积
                input_seq = original_sequence[-sequence_length:]

                # Prepare input data
                input_tensor = torch.FloatTensor(input_seq).view(1, sequence_length, 1).to(self.device)

                # Predict next return
                next_pred = self.model(input_tensor)
                pred_value = next_pred.cpu().numpy()[0, 0]

                # Save prediction
                predictions.append(pred_value)

        return np.array(predictions)

    def _predict_hybrid(self, last_sequence: np.ndarray, num_days: int) -> np.ndarray:
        """
        混合预测 - 结合递归和直接预测的优势
        """
        recursive_preds = self._predict_recursive(last_sequence, num_days)
        direct_preds = self._predict_direct(last_sequence, num_days)

        # 加权平均，前期更多依赖直接预测，后期更多依赖递归预测
        weights = np.linspace(0.7, 0.3, num_days)
        hybrid_preds = weights * direct_preds + (1 - weights) * recursive_preds

        return hybrid_preds

    def predict_with_dates(self, last_sequence: np.ndarray,
                          num_days: int = 20,
                          start_date: Optional[datetime] = None,
                          method: str = 'hybrid') -> Tuple[np.ndarray, List[datetime]]:
        """
        Predict future prices with corresponding dates - 支持收益率和原始价格两种模式

        Args:
            last_sequence: Last known sequence of returns or prices
            num_days: Number of days to predict
            start_date: Starting date for predictions (default: last date + 1 day)
            method: 'recursive', 'direct', or 'hybrid' prediction method

        Returns:
            Tuple of (predictions, prediction_dates)
        """
        # Get predictions using improved method
        predictions_scaled = self.predict_future_days(last_sequence, num_days, method=method)

        # Check if preprocessor is for returns or prices
        if hasattr(self.preprocessor, 'convert_returns_to_prices'):
            # 收益率模式: 需要转换为价格
            predicted_returns = self.preprocessor.inverse_transform(predictions_scaled).flatten()
            predictions = self.preprocessor.convert_returns_to_prices(predicted_returns)
        else:
            # 原始价格模式: 直接反标准化
            predictions = self.preprocessor.inverse_transform(predictions_scaled)

        # Generate prediction dates
        if start_date is None:
            # Use the last date from the data + 1 day
            if hasattr(self.preprocessor.data, 'index'):
                last_date = self.preprocessor.data.index[-1]
            else:
                last_date = datetime(2025, 9, 10)
            start_date = last_date + timedelta(days=1)

        prediction_dates = []
        current_date = start_date

        # Skip weekends
        for _ in range(num_days):
            while current_date.weekday() >= 5:  # Saturday = 5, Sunday = 6
                current_date += timedelta(days=1)
            prediction_dates.append(current_date)
            current_date += timedelta(days=1)

        return predictions.flatten(), prediction_dates

    def predict_multiple_scenarios(self, last_sequence: np.ndarray,
                                 num_days: int = 20,
                                 num_simulations: int = 100) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate multiple prediction scenarios with Monte Carlo simulation.

        Args:
            last_sequence: Last known sequence of prices
            num_days: Number of days to predict
            num_simulations: Number of simulation runs

        Returns:
            Tuple of (mean_predictions, std_predictions)
        """
        all_predictions = []

        # Enable dropout for uncertainty estimation
        self.model.train()  # Enable dropout

        for _ in range(num_simulations):
            with torch.no_grad():
                predictions = self.predict_future_days(last_sequence, num_days)
                predictions_actual = self.preprocessor.inverse_transform(predictions)
                all_predictions.append(predictions_actual.flatten())

        self.model.eval()  # Disable dropout

        all_predictions = np.array(all_predictions)
        mean_predictions = np.mean(all_predictions, axis=0)
        std_predictions = np.std(all_predictions, axis=0)

        return mean_predictions, std_predictions

    def predict_confidence_intervals(self, last_sequence: np.ndarray,
                                   num_days: int = 20,
                                   confidence_level: float = 0.95) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Generate predictions with confidence intervals.

        Args:
            last_sequence: Last known sequence of prices
            num_days: Number of days to predict
            confidence_level: Confidence level for intervals (0.0-1.0)

        Returns:
            Tuple of (lower_bound, mean_prediction, upper_bound)
        """
        # Generate multiple scenarios
        mean_pred, std_pred = self.predict_multiple_scenarios(
            last_sequence, num_days, num_simulations=1000
        )

        # Calculate confidence intervals
        from scipy import stats
        z_score = stats.norm.ppf((1 + confidence_level) / 2)
        margin_error = z_score * std_pred

        lower_bound = mean_pred - margin_error
        upper_bound = mean_pred + margin_error

        return lower_bound, mean_pred, upper_bound

    def save_predictions(self, predictions: np.ndarray,
                        prediction_dates: List[datetime],
                        filename: str = 'predictions.csv',
                        output_dir: str = 'output') -> pd.DataFrame:
        """
        Save predictions to CSV file.

        Args:
            predictions: Predicted prices
            prediction_dates: Corresponding dates
            filename: Output filename
            output_dir: Output directory

        Returns:
            DataFrame with predictions
        """
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)

        # Create DataFrame
        results_df = pd.DataFrame({
            'date': prediction_dates,
            'predicted_price': predictions
        })

        # Save to file
        filepath = os.path.join(output_dir, filename)
        results_df.to_csv(filepath, index=False)
        print(f"Predictions saved to {filepath}")

        return results_df

    def plot_predictions_with_confidence(self, last_sequence: np.ndarray,
                                       num_days: int = 20,
                                       save_path: Optional[str] = None) -> None:
        """
        Plot predictions with confidence intervals.

        Args:
            last_sequence: Last known sequence of prices
            num_days: Number of days to predict
            save_path: Path to save the plot
        """
        # Get predictions with confidence intervals
        lower_bound, mean_pred, upper_bound = self.predict_confidence_intervals(
            last_sequence, num_days, confidence_level=0.95
        )

        # Get dates
        _, prediction_dates = self.predict_with_dates(last_sequence, num_days)

        # Create plot
        plt.figure(figsize=(14, 8))

        # Plot confidence interval
        plt.fill_between(prediction_dates, lower_bound, upper_bound,
                        alpha=0.3, color='blue', label='95% Confidence Interval')

        # Plot mean prediction
        plt.plot(prediction_dates, mean_pred, 'b-', linewidth=2, label='Mean Prediction')

        # Add historical data for context
        if hasattr(self.preprocessor, 'data'):
            historical_prices = self.preprocessor.data['close'].values[-30:]
            historical_dates = self.preprocessor.data.index[-30:]
            plt.plot(historical_dates, historical_prices, 'g-',
                    linewidth=2, label='Historical Prices')

        plt.xlabel('Date')
        plt.ylabel('Price')
        plt.title('Stock Price Predictions with Confidence Intervals')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.xticks(rotation=45)
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Confidence interval plot saved to {save_path}")
        plt.show()

    def evaluate_prediction_quality(self, actual_prices: np.ndarray,
                                  predicted_prices: np.ndarray) -> dict:
        """
        Evaluate the quality of predictions.

        Args:
            actual_prices: Actual price values
            predicted_prices: Predicted price values

        Returns:
            Dictionary of quality metrics
        """
        from .evaluation import FinancialMetrics

        metrics_calculator = FinancialMetrics()
        metrics = metrics_calculator.evaluate_predictions(actual_prices, predicted_prices)

        return metrics

    def load_trained_model(self, model_path: str, device: torch.device) -> Tuple[BaseStockModel, List[float], List[float], dict]:
        """
        Load a trained model from file.

        Args:
            model_path: Path to the saved model
            device: Device to load the model on

        Returns:
            Tuple of (model, train_losses, val_losses, model_params)
        """
        # This method needs to be updated to work with the new model registry
        # For now, return a placeholder
        raise NotImplementedError("This method needs to be updated for multi-model support")

    def create_prediction_report(self, last_sequence: np.ndarray,
                               num_days: int = 20,
                               save_prefix: str = 'prediction_report') -> dict:
        """
        Create a comprehensive prediction report.

        Args:
            last_sequence: Last known sequence of prices
            num_days: Number of days to predict
            save_prefix: Prefix for saved files

        Returns:
            Dictionary with prediction results
        """
        print("=" * 80)
        print("LSTM Stock Price Prediction Report")
        print("=" * 80)

        # Generate predictions
        predictions, prediction_dates = self.predict_with_dates(last_sequence, num_days)

        # Calculate statistics
        price_mean = np.mean(predictions)
        price_std = np.std(predictions)
        price_min = np.min(predictions)
        price_max = np.max(predictions)

        print(f"\nPrediction Statistics:")
        print(f"Mean Price: ${price_mean:.2f}")
        print(f"Standard Deviation: ${price_std:.2f}")
        print(f"Minimum Price: ${price_min:.2f}")
        print(f"Maximum Price: ${price_max:.2f}")
        print(f"Price Range: ${price_max - price_min:.2f}")

        # Calculate daily changes
        daily_changes = np.diff(predictions)
        avg_change = np.mean(daily_changes)
        volatility = np.std(daily_changes)

        print(f"\nDaily Changes:")
        print(f"Average Daily Change: ${avg_change:.2f}")
        print(f"Daily Volatility: ${volatility:.2f}")

        # Save predictions
        results_df = self.save_predictions(predictions, prediction_dates,
                                         f'{save_prefix}.csv')

        # Create visualization
        self.plot_predictions_with_confidence(
            last_sequence, num_days,
            save_path=f'output/{save_prefix}_confidence.png'
        )

        return {
            'predictions': predictions,
            'dates': prediction_dates,
            'statistics': {
                'mean': price_mean,
                'std': price_std,
                'min': price_min,
                'max': price_max,
                'avg_daily_change': avg_change,
                'daily_volatility': volatility
            },
            'dataframe': results_df
        }


if __name__ == "__main__":
    # Test prediction functionality
    print("Testing LSTM prediction functionality...")

    # This would normally be called with a trained model and preprocessor
    print("Prediction module loaded successfully!")