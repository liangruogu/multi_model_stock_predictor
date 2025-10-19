"""
Visualization Module

Contains various plotting and visualization functions for stock price analysis
and model evaluation results.
"""

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from typing import List, Optional, Tuple
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Set style
sns.set_style("whitegrid")
try:
    plt.style.use('seaborn-v0_8')
except OSError:
    try:
        plt.style.use('seaborn')
    except OSError:
        plt.style.use('default')


class StockVisualizer:
    """Handles visualization of stock data and prediction results."""

    def __init__(self, figsize: Tuple[int, int] = (12, 8)):
        """
        Initialize the visualizer.

        Args:
            figsize: Default figure size for plots
        """
        self.figsize = figsize
        self.colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']

    def plot_training_history(self, train_losses: List[float], val_losses: List[float],
                            save_path: Optional[str] = None) -> None:
        """
        Plot training and validation loss history.

        Args:
            train_losses: List of training losses
            val_losses: List of validation losses
            save_path: Path to save the plot
        """
        fig, ax = plt.subplots(figsize=self.figsize)

        epochs = range(1, len(train_losses) + 1)

        ax.plot(epochs, train_losses, 'b-', label='Training Loss', linewidth=2, alpha=0.8)
        ax.plot(epochs, val_losses, 'r-', label='Validation Loss', linewidth=2, alpha=0.8)

        ax.set_xlabel('Epoch', fontsize=12)
        ax.set_ylabel('MSE Loss', fontsize=12)
        ax.set_title('LSTM Model Training Loss Curve', fontsize=14, fontweight='bold')
        ax.legend(fontsize=12)
        ax.grid(True, alpha=0.3)

        # Beautify plot
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Training history plot saved to {save_path}")
        plt.show()

    def plot_training_fit(self, train_predictions: np.ndarray, train_actuals: np.ndarray,
                         train_dates: list, save_path: str = 'training_fit.png'):
        """
        Plot model fitting performance on training data.

        Args:
            train_predictions: Predicted prices on training data
            train_actuals: Actual prices from training data
            train_dates: Corresponding dates
            save_path: Path to save the plot
        """
        fig, ax = plt.subplots(figsize=(16, 10))

        # Use only last 200 days for better visualization
        max_points = 200
        if len(train_predictions) > max_points:
            start_idx = len(train_predictions) - max_points
            train_pred_vis = train_predictions[start_idx:]
            train_actual_vis = train_actuals[start_idx:]
            train_dates_vis = train_dates[start_idx:]
        else:
            train_pred_vis = train_predictions
            train_actual_vis = train_actuals
            train_dates_vis = train_dates

        # Plot actual training prices
        ax.plot(train_dates_vis, train_actual_vis, 'b-',
                color=self.colors[0], label='Actual Training Price', linewidth=2, alpha=0.8)

        # Plot predicted training prices
        ax.plot(train_dates_vis, train_pred_vis, 'r--',
                color=self.colors[2], label='Model Predicted Price', linewidth=2, alpha=0.8)

        # Calculate training fit metrics
        train_mse = np.mean((train_actual_vis - train_pred_vis) ** 2)
        train_mae = np.mean(np.abs(train_actual_vis - train_pred_vis))
        train_mape = np.mean(np.abs((train_actual_vis - train_pred_vis) / train_actual_vis)) * 100

        # Add training fit statistics
        fit_text = (f'Training Fit Performance:\n'
                   f'MSE: {train_mse:.4f}\n'
                   f'MAE: {train_mae:.4f}\n'
                   f'MAPE: {train_mape:.2f}%')

        ax.text(0.98, 0.98, fit_text, transform=ax.transAxes,
                verticalalignment='top', horizontalalignment='right',
                bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.9),
                fontsize=10, fontfamily='monospace')

        ax.set_xlabel('Date', fontsize=14)
        ax.set_ylabel('Price ($)', fontsize=14)
        ax.set_title('Model Fitting Performance on Training Data (Last 200 Days)',
                    fontsize=16, fontweight='bold', pad=20)
        ax.legend(fontsize=12, loc='upper left')
        ax.grid(True, alpha=0.3)

        # Beautify plot
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

        # Format x-axis
        fig.autofmt_xdate()

        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()

        print(f"Training fit plot saved to {save_path}")

    def plot_prediction_comparison(self, actual_prices: np.ndarray,
                                 predicted_prices: np.ndarray,
                                 dates: Optional[List] = None,
                                 save_path: Optional[str] = None) -> None:
        """
        Plot comparison of actual vs predicted prices.

        Args:
            actual_prices: Actual price values
            predicted_prices: Predicted price values
            dates: Date labels for x-axis
            save_path: Path to save the plot
        """
        fig, ax = plt.subplots(figsize=(14, 8))

        if dates is None:
            dates = range(len(actual_prices))

        # Plot actual prices
        ax.plot(dates, actual_prices, 'o-', color=self.colors[0],
                label='Actual Price', linewidth=2, markersize=4, alpha=0.8)

        # Plot predicted prices
        ax.plot(dates, predicted_prices, 's--', color=self.colors[1],
                label='Predicted Price', linewidth=2, markersize=4, alpha=0.8)

        ax.set_xlabel('Time', fontsize=12)
        ax.set_ylabel('Price', fontsize=12)
        ax.set_title('Stock Price Prediction Results Comparison', fontsize=14, fontweight='bold')
        ax.legend(fontsize=12)
        ax.grid(True, alpha=0.3)

        # Beautify plot
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

        # Rotate x-axis labels if needed
        if len(dates) > 20:
            plt.xticks(rotation=45)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Prediction comparison plot saved to {save_path}")
        plt.show()

    def plot_price_trend_with_prediction(self, historical_prices: np.ndarray,
                                       historical_dates: List,
                                       predicted_prices: np.ndarray,
                                       prediction_dates: List,
                                       save_path: Optional[str] = None) -> None:
        """
        Plot historical prices with future predictions.

        Args:
            historical_prices: Historical price data
            historical_dates: Historical date labels
            predicted_prices: Predicted price data
            prediction_dates: Prediction date labels
            save_path: Path to save the plot
        """
        fig, ax = plt.subplots(figsize=(16, 9))

        # Plot historical prices
        ax.plot(historical_dates, historical_prices, 'o-',
                color=self.colors[0], label='Historical Price', linewidth=2, markersize=3)

        # Plot predicted prices
        ax.plot(prediction_dates, predicted_prices, 's--',
                color=self.colors[1], label='Predicted Price', linewidth=2, markersize=5)

        # Add division line
        ax.axvline(x=historical_dates[-1], color='red', linestyle='--',
                  alpha=0.7, label='Prediction Start')

        ax.set_xlabel('Date', fontsize=12)
        ax.set_ylabel('Price', fontsize=12)
        ax.set_title('Stock Price Historical Data and Future Predictions', fontsize=16, fontweight='bold')
        ax.legend(fontsize=12)
        ax.grid(True, alpha=0.3)

        # Beautify plot
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

        # Rotate x-axis labels
        plt.xticks(rotation=45)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Price trend plot saved to {save_path}")
        plt.show()

    def plot_prediction_errors(self, actual_prices: np.ndarray,
                             predicted_prices: np.ndarray,
                             save_path: Optional[str] = None) -> None:
        """
        Plot prediction error analysis.

        Args:
            actual_prices: Actual price values
            predicted_prices: Predicted price values
            save_path: Path to save the plot
        """
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))

        # 1. Scatter plot: Actual vs Predicted prices
        axes[0, 0].scatter(actual_prices, predicted_prices, alpha=0.6, color=self.colors[0])
        axes[0, 0].plot([actual_prices.min(), actual_prices.max()],
                       [actual_prices.min(), actual_prices.max()], 'r--', lw=2)
        axes[0, 0].set_xlabel('Actual Price')
        axes[0, 0].set_ylabel('Predicted Price')
        axes[0, 0].set_title('Actual vs Predicted Prices')
        axes[0, 0].grid(True, alpha=0.3)

        # 2. Error distribution histogram
        errors = predicted_prices - actual_prices
        axes[0, 1].hist(errors, bins=30, alpha=0.7, color=self.colors[1], edgecolor='black')
        axes[0, 1].set_xlabel('Prediction Error')
        axes[0, 1].set_ylabel('Frequency')
        axes[0, 1].set_title('Prediction Error Distribution')
        axes[0, 1].grid(True, alpha=0.3)
        axes[0, 1].axvline(x=0, color='red', linestyle='--', alpha=0.7)

        # 3. Time series errors
        axes[1, 0].plot(errors, color=self.colors[2], linewidth=1.5)
        axes[1, 0].axhline(y=0, color='red', linestyle='--', alpha=0.7)
        axes[1, 0].set_xlabel('Time')
        axes[1, 0].set_ylabel('Prediction Error')
        axes[1, 0].set_title('Time Series Prediction Errors')
        axes[1, 0].grid(True, alpha=0.3)

        # 4. Percentage errors
        percentage_errors = (predicted_prices - actual_prices) / actual_prices * 100
        axes[1, 1].plot(percentage_errors, color=self.colors[3], linewidth=1.5)
        axes[1, 1].axhline(y=0, color='red', linestyle='--', alpha=0.7)
        axes[1, 1].set_xlabel('Time')
        axes[1, 1].set_ylabel('Percentage Error (%)')
        axes[1, 1].set_title('Time Series Percentage Errors')
        axes[1, 1].grid(True, alpha=0.3)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Prediction errors plot saved to {save_path}")
        plt.show()

    def plot_financial_metrics(self, metrics: dict, save_path: Optional[str] = None) -> None:
        """
        Plot financial metrics radar chart.

        Args:
            metrics: Dictionary of financial metrics
            save_path: Path to save the plot
        """
        # Select metrics to display
        metric_names = ['Sharpe Ratio', 'Directional Accuracy', 'Sortino Ratio',
                       'Total Return', 'Volatility Control']

        # Normalize metric values to 0-1 range
        sharpe = float(metrics.get('sharpe_ratio', 0))
        direction = float(metrics.get('direction_accuracy', 0))
        sortino = float(metrics.get('sortino_ratio', 0))
        total_return = float(metrics.get('total_return_percent', 0))
        volatility = float(metrics.get('volatility', 0))

        values = [
            min(max(sharpe / 2, 0), 1),  # Sharpe ratio, assume 2 is excellent
            min(max(direction / 100, 0), 1),  # Directional accuracy
            min(max(sortino / 2, 0), 1),  # Sortino ratio
            min(max(abs(total_return) / 20, 0), 1),  # Total return, assume 20% is excellent
            min(max(1 - abs(volatility), 0), 1)  # Volatility control (lower is better)
        ]

        # Create radar chart
        angles = np.linspace(0, 2 * np.pi, len(metric_names), endpoint=False)
        values += values[:1]  # Close the shape
        angles = np.concatenate((angles, [angles[0]]))

        fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(projection='polar'))

        ax.plot(angles, values, 'o-', linewidth=2, color=self.colors[0])
        ax.fill(angles, values, alpha=0.25, color=self.colors[0])
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(metric_names)
        ax.set_ylim(0, 1)
        ax.set_title('LSTM Model Financial Metrics Evaluation', fontsize=16, fontweight='bold', pad=20)

        # Add grid
        ax.grid(True)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Financial metrics plot saved to {save_path}")
        plt.show()

    def plot_returns_distribution(self, actual_prices: np.ndarray,
                                predicted_prices: np.ndarray,
                                save_path: Optional[str] = None) -> None:
        """
        Plot returns distribution comparison.

        Args:
            actual_prices: Actual price values
            predicted_prices: Predicted price values
            save_path: Path to save the plot
        """
        # Calculate returns
        actual_returns = np.diff(actual_prices) / actual_prices[:-1] * 100
        predicted_returns = np.diff(predicted_prices) / predicted_prices[:-1] * 100

        fig, axes = plt.subplots(1, 2, figsize=(15, 6))

        # Returns distribution histogram
        axes[0].hist(actual_returns, bins=30, alpha=0.7, label='Actual Returns',
                    color=self.colors[0], density=True)
        axes[0].hist(predicted_returns, bins=30, alpha=0.7, label='Predicted Returns',
                    color=self.colors[1], density=True)
        axes[0].set_xlabel('Returns (%)')
        axes[0].set_ylabel('Density')
        axes[0].set_title('Returns Distribution Comparison')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)

        # Q-Q plot
        try:
            from scipy import stats
            stats.probplot(actual_returns, dist="norm", plot=axes[1])
            axes[1].set_title('Actual Returns Q-Q Plot')
            axes[1].grid(True, alpha=0.3)
        except ImportError:
            axes[1].text(0.5, 0.5, 'scipy not available\nfor Q-Q plot',
                        ha='center', va='center', transform=axes[1].transAxes)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Returns distribution plot saved to {save_path}")
        plt.show()

    def create_comprehensive_report(self, historical_prices: np.ndarray,
                                  historical_dates: List,
                                  predicted_prices: np.ndarray,
                                  prediction_dates: List,
                                  train_losses: List[float],
                                  val_losses: List[float],
                                  metrics: dict,
                                  save_prefix: str = 'lstm_report') -> None:
        """
        Create comprehensive visualization report.

        Args:
            historical_prices: Historical price data
            historical_dates: Historical date labels
            predicted_prices: Predicted price data
            prediction_dates: Prediction date labels
            train_losses: Training loss history
            val_losses: Validation loss history
            metrics: Financial metrics dictionary
            save_prefix: Prefix for saved files
        """
        print("📊 Generating comprehensive visualization report...")

        # 1. Training history
        self.plot_training_history(
            train_losses, val_losses,
            save_path=f'output/{save_prefix}_training_history.png'
        )

        # 2. Price trend comparison
        self.plot_price_trend_with_prediction(
            historical_prices, historical_dates,
            predicted_prices, prediction_dates,
            save_path=f'output/{save_prefix}_price_trend.png'
        )

  
        print(f"✅ Comprehensive report generated with prefix: {save_prefix}")

    def plot_prediction_comparison(self, predicted_prices: np.ndarray,
                                 prediction_dates: List,
                                 actual_prices: np.ndarray = None,
                                 actual_dates: List = None,
                                 save_path: str = 'prediction_comparison.png'):
        """
        Create a comparison plot showing predicted vs actual prices for the prediction period.

        Args:
            predicted_prices: Predicted future prices
            prediction_dates: Prediction date range
            actual_prices: Actual prices for the same period (if available)
            actual_dates: Actual dates for the same period (if available)
            save_path: Path to save the plot
        """
        fig, ax = plt.subplots(figsize=(14, 8))

        # Plot predicted prices
        ax.plot(prediction_dates, predicted_prices, 'o-',
                color=self.colors[3], label='Predicted Prices',
                linewidth=3, markersize=8, alpha=0.9)

        # Plot actual prices if available
        if actual_prices is not None and actual_dates is not None:
            ax.plot(actual_dates, actual_prices, 's-',
                    color=self.colors[1], label='Actual Prices',
                    linewidth=3, markersize=6, alpha=0.9)

        # Combine all information in top-right corner
        pred_change = predicted_prices[-1] - predicted_prices[0]
        pred_change_pct = (pred_change / predicted_prices[0]) * 100

        # Combine prediction accuracy and change statistics
        if actual_prices is not None and len(actual_prices) == len(predicted_prices):
            mse = np.mean((actual_prices - predicted_prices) ** 2)
            mae = np.mean(np.abs(actual_prices - predicted_prices))
            mape = np.mean(np.abs((actual_prices - predicted_prices) / actual_prices)) * 100

            combined_text = (f'Prediction Accuracy:\n'
                           f'MSE: {mse:.4f}\n'
                           f'MAE: {mae:.4f}\n'
                           f'MAPE: {mape:.2f}%\n'
                           f'Change: {pred_change_pct:+.2f}%')
        else:
            combined_text = f'Prediction Change: {pred_change_pct:+.2f}%'

        ax.text(0.98, 0.98, combined_text, transform=ax.transAxes,
                verticalalignment='top', horizontalalignment='right',
                bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.9),
                fontsize=9, fontfamily='monospace')

        ax.set_xlabel('Date', fontsize=14)
        ax.set_ylabel('Price ($)', fontsize=14)
        ax.set_title('Stock Price Prediction vs Actual Results',
                    fontsize=16, fontweight='bold', pad=20)
        ax.legend(fontsize=12, loc='upper left')
        ax.grid(True, alpha=0.3)

        # Beautify plot
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

        # Format x-axis
        fig.autofmt_xdate()

        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()

        print(f"Prediction comparison plot saved to {save_path}")

    def create_comprehensive_report_with_test(self, historical_prices: np.ndarray,
                                            historical_dates: List,
                                            test_prices: np.ndarray,
                                            test_dates: List,
                                            test_predictions: np.ndarray,
                                            predicted_prices: np.ndarray,
                                            prediction_dates: List,
                                            train_losses: List[float],
                                            val_losses: List[float],
                                            metrics: dict,
                                            save_prefix: str = 'lstm_report') -> None:
        """
        Create comprehensive visualization report with test data comparison.

        Args:
            historical_prices: Historical price data
            historical_dates: Historical date labels
            test_prices: Test set actual prices
            test_dates: Test set date labels
            test_predictions: Test set predicted prices
            predicted_prices: Future predicted prices
            prediction_dates: Prediction date labels
            train_losses: Training loss history
            val_losses: Validation loss history
            metrics: Financial metrics dictionary
            save_prefix: Prefix for saved files
        """
        print("📊 Generating comprehensive visualization report with test data...")

        # 1. Training history
        self.plot_training_history(
            train_losses, val_losses,
            save_path=f'output/{save_prefix}_training_history.png'
        )

        # 2. Enhanced price trend comparison with test data
        self.plot_enhanced_price_trend(
            historical_prices, historical_dates,
            test_prices, test_dates, test_predictions,
            predicted_prices, prediction_dates,
            save_path=f'output/{save_prefix}_price_trend_with_test.png'
        )

  
        print(f"✅ Comprehensive report with test data generated with prefix: {save_prefix}")

    def plot_enhanced_price_trend(self, historical_prices: np.ndarray,
                                 historical_dates: List,
                                 test_prices: np.ndarray,
                                 test_dates: List,
                                 test_predictions: np.ndarray,
                                 predicted_prices: np.ndarray,
                                 prediction_dates: List,
                                 save_path: Optional[str] = None) -> None:
        """
        Plot comprehensive price trend with test data and future predictions.

        Args:
            historical_prices: Historical price data
            historical_dates: Historical date labels
            test_prices: Test set actual prices
            test_dates: Test set date labels
            test_predictions: Test set predicted prices
            predicted_prices: Future predicted prices
            prediction_dates: Prediction date labels
            save_path: Path to save the plot
        """
        fig, ax = plt.subplots(figsize=(18, 10))

        # Plot historical prices (last 60 days for context)
        context_days = 60
        start_idx = len(historical_prices) - context_days
        context_prices = historical_prices[start_idx:]
        context_dates = historical_dates[start_idx:]

        ax.plot(context_dates, context_prices, 'o-',
                color=self.colors[0], label='Historical Price', linewidth=2,
                markersize=4, alpha=0.7)

        # Plot test set actual prices
        ax.plot(test_dates, test_prices, 'o-',
                color=self.colors[1], label='Test Actual Price', linewidth=3,
                markersize=6, alpha=0.9)

        # Plot test set predicted prices
        ax.plot(test_dates, test_predictions, 's--',
                color=self.colors[2], label='Test Predicted Price', linewidth=2,
                markersize=5, alpha=0.8)

        # Plot future predicted prices
        ax.plot(prediction_dates, predicted_prices, '^--',
                color=self.colors[3], label='Future Prediction', linewidth=3,
                markersize=7, alpha=0.9)

        # Add vertical lines to mark different phases
        ax.axvline(x=test_dates[0], color='red', linestyle='--',
                  alpha=0.7, label='Test Period Start', linewidth=2)
        ax.axvline(x=prediction_dates[0], color='green', linestyle='--',
                  alpha=0.7, label='Prediction Period Start', linewidth=2)

        # Add prediction change statistics to top-right corner
        first_pred_price = predicted_prices[0]
        last_pred_price = predicted_prices[-1]
        pred_change = last_pred_price - first_pred_price
        pred_change_pct = (pred_change / first_pred_price) * 100

        # Move change statistics to right corner
        change_text = f'Prediction: {pred_change_pct:+.2f}%'
        ax.text(0.98, 0.98, change_text, transform=ax.transAxes,
                verticalalignment='top', horizontalalignment='right',
                bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.7),
                fontsize=10)

        ax.set_xlabel('Date', fontsize=14)
        ax.set_ylabel('Price ($)', fontsize=14)
        ax.set_title('Comprehensive Stock Price Analysis: Historical + Test + Future Prediction',
                    fontsize=16, fontweight='bold', pad=20)
        ax.legend(fontsize=12, loc='upper left')
        ax.grid(True, alpha=0.3)

        # Beautify plot
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

        # Rotate x-axis labels
        plt.xticks(rotation=45)

        # Add error statistics annotation to top-right corner
        test_mae = np.mean(np.abs(test_prices - test_predictions))
        test_mape = np.mean(np.abs((test_prices - test_predictions) / test_prices)) * 100

        error_text = f'Test Set Performance:\nMAE: ${test_mae:.2f}\nMAPE: {test_mape:.2f}%'
        ax.text(0.98, 0.02, error_text, transform=ax.transAxes,
                fontsize=11, verticalalignment='bottom', horizontalalignment='right',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Enhanced price trend plot saved to {save_path}")
        plt.show()

    def plot_training_set_comparison(self, train_predictions: np.ndarray,
                                     train_actuals: np.ndarray,
                                     train_dates: list,
                                     model_type: str = 'Model',
                                     save_path: Optional[str] = None) -> None:
        """
        Plot comprehensive training set prediction vs actual comparison.
        This method helps identify displacement issues in predictions.

        Args:
            train_predictions: Predicted prices on training set
            train_actuals: Actual prices from training set
            train_dates: Corresponding dates
            model_type: Type of model for title
            save_path: Path to save the plot
        """
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 12))

        # Convert to 1D arrays if needed
        train_pred_flat = train_predictions.flatten()
        train_actual_flat = train_actuals.flatten()

        # 1. Full time series plot (top subplot)
        ax1.plot(train_dates, train_actual_flat, 'b-',
                label='Actual Price', linewidth=2, alpha=0.8)
        ax1.plot(train_dates, train_pred_flat, 'r--',
                label='Predicted Price', linewidth=2, alpha=0.8)

        # Calculate and display displacement
        displacement = np.mean(train_pred_flat - train_actual_flat)
        displacement_pct = (displacement / np.mean(train_actual_flat)) * 100

        # Add displacement information
        displacement_text = (f'{model_type} Training Analysis:\n'
                            f'Mean Displacement: {displacement:+.4f}\n'
                            f'Displacement %: {displacement_pct:+.2f}%')

        ax1.text(0.02, 0.98, displacement_text, transform=ax1.transAxes,
                verticalalignment='top', horizontalalignment='left',
                bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.9),
                fontsize=10, fontfamily='monospace')

        ax1.set_xlabel('Date', fontsize=12)
        ax1.set_ylabel('Price ($)', fontsize=12)
        ax1.set_title(f'{model_type} Training Set: Predicted vs Actual Prices',
                      fontsize=14, fontweight='bold')
        ax1.legend(fontsize=12)
        ax1.grid(True, alpha=0.3)
        ax1.spines['top'].set_visible(False)
        ax1.spines['right'].set_visible(False)

        # Format x-axis for top plot
        fig.autofmt_xdate(rotation=45)

        # 2. Scatter plot with diagonal (bottom subplot)
        ax2.scatter(train_actual_flat, train_pred_flat, alpha=0.6, s=20, color=self.colors[0])

        # Add perfect prediction line (y=x)
        min_val = min(train_actual_flat.min(), train_pred_flat.min())
        max_val = max(train_actual_flat.max(), train_pred_flat.max())
        ax2.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2,
                label='Perfect Prediction (y=x)')

        # Add displacement line (y=x+displacement)
        ax2.plot([min_val, max_val], [min_val + displacement, max_val + displacement],
                'g--', linewidth=2, alpha=0.7,
                label=f'Mean Prediction (y=x+{displacement:.3f})')

        # Calculate metrics
        mse = np.mean((train_actual_flat - train_pred_flat) ** 2)
        mae = np.mean(np.abs(train_actual_flat - train_pred_flat))
        mape = np.mean(np.abs((train_actual_flat - train_pred_flat) / train_actual_flat)) * 100
        correlation = np.corrcoef(train_actual_flat, train_pred_flat)[0, 1]

        # Add metrics
        metrics_text = (f'Metrics:\n'
                       f'MSE: {mse:.4f}\n'
                       f'MAE: {mae:.4f}\n'
                       f'MAPE: {mape:.2f}%\n'
                       f'Correlation: {correlation:.3f}')

        ax2.text(0.98, 0.02, metrics_text, transform=ax2.transAxes,
                verticalalignment='bottom', horizontalalignment='right',
                bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.9),
                fontsize=9, fontfamily='monospace')

        ax2.set_xlabel('Actual Price ($)', fontsize=12)
        ax2.set_ylabel('Predicted Price ($)', fontsize=12)
        ax2.set_title(f'{model_type} Training Set: Predicted vs Actual Scatter',
                      fontsize=14, fontweight='bold')
        ax2.legend(fontsize=10)
        ax2.grid(True, alpha=0.3)
        ax2.spines['top'].set_visible(False)
        ax2.spines['right'].set_visible(False)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Training set comparison plot saved to {save_path}")
        else:
            plt.show()

    def plot_multiple_predictions(self, predictions_dict: dict,
                                 save_path: Optional[str] = None) -> None:
        """
        Plot multiple prediction strategies comparison.

        Args:
            predictions_dict: Dictionary of {name: (dates, prices)}
            save_path: Path to save the plot
        """
        fig, ax = plt.subplots(figsize=(16, 9))

        colors = self.colors[:len(predictions_dict)]

        for i, (name, (dates, prices)) in enumerate(predictions_dict.items()):
            ax.plot(dates, prices, '-o', label=name, linewidth=2,
                   markersize=3, color=colors[i], alpha=0.8)

        ax.set_xlabel('Date', fontsize=12)
        ax.set_ylabel('Price', fontsize=12)
        ax.set_title('Multiple Prediction Strategies Comparison', fontsize=16, fontweight='bold')
        ax.legend(fontsize=12)
        ax.grid(True, alpha=0.3)

        # Beautify plot
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

        # Rotate x-axis labels
        plt.xticks(rotation=45)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Multiple predictions plot saved to {save_path}")
        plt.show()


if __name__ == "__main__":
    # Test visualization functions
    visualizer = StockVisualizer()

    # Generate test data
    np.random.seed(42)
    dates = pd.date_range('2025-01-01', periods=100)
    actual_prices = 100 + np.cumsum(np.random.randn(100) * 2)
    predicted_prices = actual_prices + np.random.randn(100) * 1

    # Plot prediction comparison
    visualizer.plot_prediction_comparison(
        actual_prices, predicted_prices, dates
    )

    # Plot error analysis
    visualizer.plot_prediction_errors(
        actual_prices, predicted_prices
    )