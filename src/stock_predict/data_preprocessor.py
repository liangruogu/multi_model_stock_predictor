"""
Data Preprocessor Module

Handles loading, cleaning, and preprocessing of stock market data for LSTM model training.
"""

import pandas as pd
import numpy as np
import sys
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import Dataset, DataLoader
from typing import Tuple, Optional
from loguru import logger
from copy import deepcopy as dc

# 配置日志 - 暂时关闭所有日志
logger.remove()
# logger.add(
#     sys.stderr,
#     format="<level>{message}</level>",
#     level="INFO"
# )


class StockDataset(Dataset):
    """PyTorch Dataset for stock price sequences."""

    def __init__(self, data: np.ndarray, sequence_length: int = 30):
        """
        Initialize the dataset.

        Args:
            data: Preprocessed stock data
            sequence_length: Length of input sequences
        """
        self.data = data
        self.sequence_length = sequence_length

    def __len__(self) -> int:
        return len(self.data) - self.sequence_length

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get a sample from the dataset.

        Args:
            idx: Index of the sample

        Returns:
            Tuple of input sequence and target value
        """
        x = self.data[idx:idx + self.sequence_length]
        y = self.data[idx + self.sequence_length]

        # 转换为张量
        x_tensor = torch.FloatTensor(x)
        y_tensor = torch.FloatTensor([y])

        # 记录关键样本信息（只在第一个样本）
        if idx == 0:
            logger.info(f"📊 样本0: 输入序列价格范围 ${x_tensor.min():.2f}-${x_tensor.max():.2f}, 目标价格 ${y_tensor.item():.2f}")

        return x_tensor, y_tensor


class BlogStyleDataPreprocessor:
    """博客风格的数据预处理器 - 使用滞后特征和MinMaxScaler"""

    def __init__(self, n_steps: int = 60):
        """
        Initialize the blog-style data preprocessor.

        Args:
            n_steps: Number of lookback steps (like blog's lookback)
        """
        self.n_steps = n_steps
        self.scaler = MinMaxScaler(feature_range=(-1, 1))  # 博客使用(-1,1)
        self.data: Optional[pd.DataFrame] = None

    def prepare_dataframe_for_lstm(self, df: pd.DataFrame, n_steps: int) -> pd.DataFrame:
        """
        处理数据集，使其适用于LSTM模型 - 博客方法
        """
        df = dc(df)

        # 确保有date列
        if 'trade_date' in df.columns:
            df['date'] = pd.to_datetime(df['trade_date'])
        elif 'date' not in df.columns:
            df['date'] = pd.to_datetime(df.index)

        df.set_index('date', inplace=True)

        # 创建滞后特征
        for i in range(1, n_steps+1):
            df[f'close(t-{i})'] = df['close'].shift(i)

        df.dropna(inplace=True)
        return df

    def get_dataset(self, file_path: str, lookback: int = None, split_ratio: float = 0.9):
        """
        归一化数据、划分训练集和测试集 - 博客方法
        """
        if lookback is None:
            lookback = self.n_steps

        data = pd.read_csv(file_path)

        # 只使用date和close列
        if 'trade_date' in data.columns:
            data = data[['trade_date','close']]
            data.rename(columns={'trade_date': 'date'}, inplace=True)
        else:
            data = data[['date','close']]

        shifted_df = self.prepare_dataframe_for_lstm(data, lookback)

        # 使用MinMaxScaler归一化到(-1,1)
        shifted_np = self.scaler.fit_transform(shifted_df)

        X = shifted_np[:, 1:]  # 滞后特征
        y = shifted_np[:, 0]   # 当前值

        # 反转时间顺序，像博客一样
        X = dc(np.flip(X, axis=1))

        # 划分训练集和测试集
        split_index = int(len(X) * split_ratio)

        X_train = X[:split_index]
        X_test = X[split_index:]

        y_train = y[:split_index]
        y_test = y[split_index:]

        # 重塑为LSTM输入格式
        X_train = X_train.reshape((-1, lookback, 1))
        X_test = X_test.reshape((-1, lookback, 1))

        y_train = y_train.reshape((-1, 1))
        y_test = y_test.reshape((-1, 1))

        # 转换为PyTorch张量
        X_train = torch.tensor(X_train).float()
        y_train = torch.tensor(y_train).float()
        X_test = torch.tensor(X_test).float()
        y_test = torch.tensor(y_test).float()

        return self.scaler, X_train, X_test, y_train, y_test

    def inverse_transform(self, values: np.ndarray) -> np.ndarray:
        """
        反归一化 - 将归一化值转换回原始价格
        """
        # 创建一个完整形状的数组进行反归一化
        dummies = np.zeros((len(values), self.n_steps + 1))
        dummies[:, 0] = values.flatten()

        # 反归一化
        inversed = self.scaler.inverse_transform(dummies)
        return inversed[:, 0].reshape(-1, 1)

    def get_latest_sequence(self) -> np.ndarray:
        """
        获取最新的序列用于预测 - 博客风格
        """
        # 重新读取最新数据
        data = pd.read_csv('data/000001_daily_qfq_8y.csv')
        if 'trade_date' in data.columns:
            data = data[['trade_date','close']]
            data.rename(columns={'trade_date': 'date'}, inplace=True)
        else:
            data = data[['date','close']]

        # 准备数据
        shifted_df = self.prepare_dataframe_for_lstm(data, self.n_steps)
        shifted_np = self.scaler.transform(shifted_df)

        # 获取最新的序列（反转时间顺序）
        X = shifted_np[:, 1:]
        X = np.flip(X, axis=1)

        # 返回最后一个序列
        return X[-1:].reshape(self.n_steps, 1)


class DataPreprocessor:
    """原始数据预处理器 - 保留兼容性"""

    def __init__(self, sequence_length: int = 30):
        """
        Initialize the data preprocessor.

        Args:
            sequence_length: Length of input sequences for LSTM
        """
        self.sequence_length = sequence_length
        self.scaler = MinMaxScaler(feature_range=(0, 1))
        self.data: Optional[pd.DataFrame] = None
        self.train_data: Optional[np.ndarray] = None
        self.test_data: Optional[np.ndarray] = None

    def load_data(self, file_path: str) -> pd.DataFrame:
        """
        Load stock data from CSV file.

        Args:
            file_path: Path to the CSV file containing stock data

        Returns:
            Loaded and preprocessed DataFrame
        """
        self.data = pd.read_csv(file_path)

        # Convert date format and set as index
        self.data['trade_date'] = pd.to_datetime(self.data['trade_date'])
        self.data.set_index('trade_date', inplace=True)

        # Sort by date
        self.data = self.data.sort_index()

        # Select feature columns
        features = ['open', 'close', 'high', 'low', 'volume']
        self.data = self.data[features]

        # Handle missing values
        self.data = self.data.ffill().bfill()

        print(f"Data loaded successfully: {len(self.data)} records")
        print(f"Date range: {self.data.index[0]} to {self.data.index[-1]}")

        return self.data

    def preprocess_data(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        数据预处理 - 使用归一化处理，增强价格变化的敏感性

        Returns:
            Tuple of (train_data, test_data)
        """
        if self.data is None:
            raise ValueError("Data not loaded. Call load_data() first.")

        # 使用收盘价
        close_prices = self.data['close'].values

        # 存储原始价格信息，用于反归一化
        self.original_prices = close_prices
        self.price_mean = np.mean(close_prices)  # 价格均值
        self.price_std = np.std(close_prices)    # 价格标准差
        self.min_price = np.min(close_prices)    # 最小价格
        self.max_price = np.max(close_prices)    # 最大价格

        # 处理可能的异常值
        close_prices = np.nan_to_num(close_prices, nan=0.0, posinf=0.0, neginf=0.0)

        print(f"📊 原始价格统计:")
        print(f"价格范围: ${self.min_price:.2f} - ${self.max_price:.2f}")
        print(f"平均价格: ${self.price_mean:.2f}")
        print(f"价格标准差: ${self.price_std:.2f}")
        print(f"数据点数量: {len(close_prices)}")

        # 关键改进: 使用Z-score归一化，增强小变化的敏感性
        normalized_prices = (close_prices - self.price_mean) / self.price_std

        print(f"\n📈 归一化后统计:")
        print(f"归一化范围: [{normalized_prices.min():.3f}, {normalized_prices.max():.3f}]")
        print(f"归一化均值: {normalized_prices.mean():.6f} (应该接近0)")
        print(f"归一化标准差: {normalized_prices.std():.6f} (应该接近1)")

        # 转换为二维数组格式
        price_data = normalized_prices.reshape(-1, 1)

        # Split training and testing sets
        train_size = len(price_data) - 120  # Last 120 days for testing

        train_data = price_data[:train_size]
        test_data = price_data[train_size - self.sequence_length:]

        self.train_data = train_data
        self.test_data = test_data

        print(f"\n训练集大小: {len(train_data)}")
        print(f"测试集大小: {len(test_data)}")
        print(f"训练集归一化范围: [{train_data.min():.3f}, {train_data.max():.3f}]")
        print(f"测试集归一化范围: [{test_data.min():.3f}, {test_data.max():.3f}]")

        return train_data, test_data

    def create_datasets(self, batch_size: int = 32) -> Tuple[DataLoader, DataLoader, DataLoader]:
        """
        Create PyTorch datasets and data loaders.

        Args:
            batch_size: Batch size for training

        Returns:
            Tuple of (train_loader, val_loader, test_loader)
        """
        if self.train_data is None or self.test_data is None:
            raise ValueError("Data not preprocessed. Call preprocess_data() first.")

        # Create training and validation sets
        train_data, val_data = train_test_split(
            self.train_data,
            test_size=0.2,
            shuffle=False
        )

        train_dataset = StockDataset(train_data, self.sequence_length)
        val_dataset = StockDataset(val_data, self.sequence_length)
        test_dataset = StockDataset(self.test_data, self.sequence_length)

        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False
        )
        test_loader = DataLoader(
            test_dataset,
            batch_size=1,
            shuffle=False
        )

        return train_loader, val_loader, test_loader

    def inverse_transform(self, values: np.ndarray) -> np.ndarray:
        """
        反归一化：将归一化的预测值转换回原始价格

        Args:
            values: Normalized predicted values

        Returns:
            Original price values
        """
        # 反归一化公式: original = normalized * std + mean
        original_values = values * self.price_std + self.price_mean
        return original_values.reshape(-1, 1)

    def convert_returns_to_prices(self, predicted_prices: np.ndarray, start_price: float = None) -> np.ndarray:
        """
        直接返回预测的价格，因为已经预测了绝对价格

        Args:
            predicted_prices: Predicted absolute prices (already in dollars)
            start_price: Not used, kept for compatibility

        Returns:
            Predicted absolute prices (same as input)
        """
        return predicted_prices.flatten()

    def get_latest_sequence(self) -> np.ndarray:
        """
        Get the latest sequence of returns for prediction.

        Returns:
            Latest sequence of returns
        """
        if self.train_data is None:
            raise ValueError("Data not preprocessed. Call preprocess_data() first.")

        return self.train_data[-self.sequence_length:]


if __name__ == "__main__":
    # Test data preprocessing
    preprocessor = DataPreprocessor(sequence_length=30)

    # Load data
    data = preprocessor.load_data("../../data/000001_daily_qfq_8y.csv")

    # Preprocess data
    train_data, test_data = preprocessor.preprocess_data()

    # Create data loaders
    train_loader, val_loader, test_loader = preprocessor.create_datasets(batch_size=32)

    print(f"Training batches: {len(train_loader)}")
    print(f"Validation batches: {len(val_loader)}")
    print(f"Testing batches: {len(test_loader)}")