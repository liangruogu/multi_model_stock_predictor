"""
Financial Evaluation Metrics Module

Contains various financial metrics for evaluating stock prediction performance
including Sharpe ratio, Sortino ratio, maximum drawdown, and other risk-adjusted measures.
"""

import numpy as np
import pandas as pd
from typing import Tuple, Dict, Any, List


class FinancialMetrics:
    """Calculates various financial metrics for stock prediction evaluation."""

    def __init__(self, risk_free_rate: float = 0.03):
        """
        Initialize financial metrics calculator.

        Args:
            risk_free_rate: Annual risk-free rate (default: 3%)
        """
        self.risk_free_rate = risk_free_rate

    def calculate_returns(self, prices: np.ndarray) -> np.ndarray:
        """
        Calculate daily returns from price series.

        Args:
            prices: Array of prices

        Returns:
            Array of daily returns
        """
        # Ensure prices is a 1D array
        if prices.ndim > 1:
            prices = prices.flatten()

        if len(prices) <= 1:
            return np.array([])

        returns = np.diff(prices) / prices[:-1]
        return returns

    def calculate_sharpe_ratio(self, returns: np.ndarray, trading_days: int = 252) -> float:
        """
        Calculate Sharpe ratio (risk-adjusted return).

        Args:
            returns: Array of returns
            trading_days: Number of trading days per year

        Returns:
            Sharpe ratio
        """
        if len(returns) == 0:
            return 0.0

        # Annual return
        annual_return = np.mean(returns) * trading_days

        # Annual volatility
        annual_volatility = np.std(returns) * np.sqrt(trading_days)

        # Sharpe ratio
        if annual_volatility == 0:
            return 0.0

        sharpe_ratio = (annual_return - self.risk_free_rate) / annual_volatility

        return sharpe_ratio

    def calculate_max_drawdown(self, prices: np.ndarray) -> Tuple[float, int, int]:
        """
        Calculate maximum drawdown.

        Args:
            prices: Array of prices

        Returns:
            Tuple of (max_drawdown, start_position, end_position)
        """
        # Ensure prices is a 1D array
        if prices.ndim > 1:
            prices = prices.flatten()

        if len(prices) == 0:
            return 0.0, 0, 0

        peak = np.maximum.accumulate(prices)
        drawdown = (prices - peak) / peak
        max_drawdown = np.min(drawdown)

        # Find start and end positions of max drawdown
        end_position = np.argmin(drawdown)
        start_position = np.argmax(prices[:end_position+1])

        return max_drawdown, start_position, end_position

    def calculate_volatility(self, returns: np.ndarray, trading_days: int = 252) -> float:
        """
        Calculate annualized volatility.

        Args:
            returns: Array of returns
            trading_days: Number of trading days per year

        Returns:
            Annualized volatility
        """
        if len(returns) == 0:
            return 0.0

        return np.std(returns) * np.sqrt(trading_days)

    def calculate_calmar_ratio(self, total_return: float, max_drawdown: float) -> float:
        """
        Calculate Calmar ratio (annual return / maximum drawdown).

        Args:
            total_return: Total return as percentage
            max_drawdown: Maximum drawdown (absolute value)

        Returns:
            Calmar ratio
        """
        if max_drawdown == 0:
            return 0.0

        return total_return / abs(max_drawdown)

    def calculate_sortino_ratio(self, returns: np.ndarray, trading_days: int = 252) -> float:
        """
        Calculate Sortino ratio (considers only downside volatility).

        Args:
            returns: Array of returns
            trading_days: Number of trading days per year

        Returns:
            Sortino ratio
        """
        if len(returns) == 0:
            return 0.0

        # Annual return
        annual_return = np.mean(returns) * trading_days

        # Downside volatility (only negative returns)
        negative_returns = returns[returns < 0]
        if len(negative_returns) == 0:
            return float('inf')  # No downside risk

        downside_volatility = np.std(negative_returns) * np.sqrt(trading_days)

        if downside_volatility == 0:
            return 0.0

        # Sortino ratio
        sortino_ratio = (annual_return - self.risk_free_rate) / downside_volatility

        return sortino_ratio

    def calculate_information_ratio(self, portfolio_returns: np.ndarray,
                                  benchmark_returns: np.ndarray) -> float:
        """
        Calculate Information ratio (excess return / tracking error).

        Args:
            portfolio_returns: Portfolio returns
            benchmark_returns: Benchmark returns

        Returns:
            Information ratio
        """
        if len(portfolio_returns) != len(benchmark_returns) or len(portfolio_returns) == 0:
            return 0.0

        # Excess returns
        excess_returns = portfolio_returns - benchmark_returns

        # Tracking error (standard deviation of excess returns)
        tracking_error = np.std(excess_returns)

        if tracking_error == 0:
            return 0.0

        # Information ratio
        information_ratio = np.mean(excess_returns) / tracking_error

        return information_ratio

    def calculate_beta(self, asset_returns: np.ndarray, market_returns: np.ndarray) -> float:
        """
        Calculate Beta coefficient.

        Args:
            asset_returns: Asset returns
            market_returns: Market returns

        Returns:
            Beta coefficient
        """
        if len(asset_returns) != len(market_returns) or len(asset_returns) == 0:
            return 0.0

        covariance = np.cov(asset_returns, market_returns)[0, 1]
        market_variance = np.var(market_returns)

        if market_variance == 0:
            return 0.0

        beta = covariance / market_variance

        return beta

    def evaluate_predictions(self, actual_prices: np.ndarray,
                           predicted_prices: np.ndarray) -> Dict[str, Any]:
        """
        Comprehensively evaluate prediction results.

        Args:
            actual_prices: Actual prices
            predicted_prices: Predicted prices

        Returns:
            Dictionary containing various evaluation metrics
        """
        if len(actual_prices) != len(predicted_prices):
            raise ValueError("Actual and predicted prices must have the same length")

        # Basic prediction error metrics
        mse = np.mean((actual_prices - predicted_prices) ** 2)
        rmse = np.sqrt(mse)
        mae = np.mean(np.abs(actual_prices - predicted_prices))
        mape = np.mean(np.abs((actual_prices - predicted_prices) / actual_prices)) * 100

        # Directional accuracy (whether prediction of price direction is correct)
        actual_returns = self.calculate_returns(actual_prices)
        predicted_returns = self.calculate_returns(predicted_prices)

        if len(actual_returns) > 0 and len(predicted_returns) > 0:
            direction_accuracy = np.mean(
                (actual_returns > 0) == (predicted_returns > 0)
            ) * 100
        else:
            direction_accuracy = 0.0

        # Financial metrics based on predicted prices
        predicted_returns_full = self.calculate_returns(predicted_prices)

        sharpe_ratio = self.calculate_sharpe_ratio(predicted_returns_full)
        max_drawdown, _, _ = self.calculate_max_drawdown(predicted_prices)
        volatility = self.calculate_volatility(predicted_returns_full)
        sortino_ratio = self.calculate_sortino_ratio(predicted_returns_full)

        # Total return
        total_return = (predicted_prices[-1] - predicted_prices[0]) / predicted_prices[0] * 100
        calmar_ratio = self.calculate_calmar_ratio(total_return, max_drawdown)

        return {
            'mse': float(mse),
            'rmse': float(rmse),
            'mae': float(mae),
            'mape': float(mape),
            'direction_accuracy': float(direction_accuracy),
            'sharpe_ratio': float(sharpe_ratio),
            'max_drawdown': float(max_drawdown),
            'volatility': float(volatility),
            'sortino_ratio': float(sortino_ratio),
            'total_return_percent': float(total_return),
            'calmar_ratio': float(calmar_ratio)
        }

    def print_evaluation_report(self, metrics: Dict[str, Any]) -> None:
        """
        Print a comprehensive evaluation report.

        Args:
            metrics: Dictionary of evaluation metrics
        """
        print("=" * 80)
        print("LSTM Stock Prediction Model - Financial Metrics Evaluation Report")
        print("=" * 80)

        print("\n📊 Prediction Accuracy Metrics:")
        print(f"Mean Squared Error (MSE): {metrics['mse']:.6f}")
        print(f"Root Mean Squared Error (RMSE): {metrics['rmse']:.6f}")
        print(f"Mean Absolute Error (MAE): {metrics['mae']:.6f}")
        print(f"Mean Absolute Percentage Error (MAPE): {metrics['mape']:.2f}%")
        print(f"Directional Accuracy: {metrics['direction_accuracy']:.2f}%")

        print("\n💰 Financial Risk-Return Metrics:")
        print(f"Sharpe Ratio: {metrics['sharpe_ratio']:.4f}")
        print(f"Sortino Ratio: {metrics['sortino_ratio']:.4f}")
        print(f"Calmar Ratio: {metrics['calmar_ratio']:.4f}")
        print(f"Total Return: {metrics['total_return_percent']:.2f}%")

        print("\n⚠️ Risk Metrics:")
        print(f"Maximum Drawdown: {metrics['max_drawdown']:.2%}")
        print(f"Annualized Volatility: {metrics['volatility']:.2%}")

        # Metric interpretation
        print("\n📝 Metric Interpretation:")
        if metrics['sharpe_ratio'] > 1:
            print("✅ Sharpe Ratio > 1: Good risk-adjusted returns")
        elif metrics['sharpe_ratio'] > 0:
            print("⚠️ Sharpe Ratio > 0: Positive but low risk-adjusted returns")
        else:
            print("❌ Sharpe Ratio < 0: Negative risk-adjusted returns")

        if metrics['direction_accuracy'] > 55:
            print(f"✅ Directional Accuracy {metrics['direction_accuracy']:.1f}%: Good trend prediction ability")
        else:
            print(f"⚠️ Directional Accuracy {metrics['direction_accuracy']:.1f}%: Limited trend prediction ability")

        if abs(metrics['max_drawdown']) < 0.1:
            print("✅ Maximum Drawdown < 10%: Good risk control")
        else:
            print(f"⚠️ Maximum Drawdown {abs(metrics['max_drawdown']):.1%}: High drawdown risk")

    def compare_strategies(self, strategy1_metrics: Dict[str, Any],
                          strategy2_metrics: Dict[str, Any],
                          strategy1_name: str = "Strategy 1",
                          strategy2_name: str = "Strategy 2") -> pd.DataFrame:
        """
        Compare two strategies side by side.

        Args:
            strategy1_metrics: Metrics for first strategy
            strategy2_metrics: Metrics for second strategy
            strategy1_name: Name of first strategy
            strategy2_name: Name of second strategy

        Returns:
            DataFrame with comparison
        """
        comparison_data = {
            'Metric': [
                'MSE', 'RMSE', 'MAE', 'MAPE (%)', 'Directional Accuracy (%)',
                'Sharpe Ratio', 'Sortino Ratio', 'Calmar Ratio', 'Total Return (%)',
                'Max Drawdown (%)', 'Volatility (%)'
            ],
            strategy1_name: [
                strategy1_metrics['mse'],
                strategy1_metrics['rmse'],
                strategy1_metrics['mae'],
                strategy1_metrics['mape'],
                strategy1_metrics['direction_accuracy'],
                strategy1_metrics['sharpe_ratio'],
                strategy1_metrics['sortino_ratio'],
                strategy1_metrics['calmar_ratio'],
                strategy1_metrics['total_return_percent'],
                strategy1_metrics['max_drawdown'] * 100,
                strategy1_metrics['volatility'] * 100
            ],
            strategy2_name: [
                strategy2_metrics['mse'],
                strategy2_metrics['rmse'],
                strategy2_metrics['mae'],
                strategy2_metrics['mape'],
                strategy2_metrics['direction_accuracy'],
                strategy2_metrics['sharpe_ratio'],
                strategy2_metrics['sortino_ratio'],
                strategy2_metrics['calmar_ratio'],
                strategy2_metrics['total_return_percent'],
                strategy2_metrics['max_drawdown'] * 100,
                strategy2_metrics['volatility'] * 100
            ]
        }

        comparison_df = pd.DataFrame(comparison_data)
        return comparison_df


if __name__ == "__main__":
    # Test financial metrics calculation
    metrics_calculator = FinancialMetrics(risk_free_rate=0.03)

    # Generate test data
    np.random.seed(42)
    actual_prices = 100 + np.cumsum(np.random.randn(100) * 2)
    predicted_prices = actual_prices + np.random.randn(100) * 1

    # Calculate evaluation metrics
    metrics = metrics_calculator.evaluate_predictions(actual_prices, predicted_prices)

    # Print report
    metrics_calculator.print_evaluation_report(metrics)