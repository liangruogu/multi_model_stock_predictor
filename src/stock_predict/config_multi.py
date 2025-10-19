"""
Configuration Loader for Multi-Model Support

Handles loading and validation of configuration files for different model types.
"""

import os
import yaml
from typing import Dict, Any, Optional
from pathlib import Path


class MultiModelConfigLoader:
    """Configuration loader for multi-model stock prediction system."""

    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize configuration loader.

        Args:
            config_path: Optional path to configuration file
        """
        self.config_path = config_path or self._find_default_config()
        self.config = self._load_config()

    def _find_default_config(self) -> str:
        """Find default configuration file."""
        possible_paths = [
            'config_multi.yaml',
            'config.yaml',
            os.path.join(os.path.dirname(__file__), '../../../config_multi.yaml'),
            os.path.join(os.path.dirname(__file__), '../../../config.yaml')
        ]

        for path in possible_paths:
            if os.path.exists(path):
                return path

        raise FileNotFoundError("No configuration file found. Please create config_multi.yaml or config.yaml")

    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from YAML file."""
        try:
            with open(self.config_path, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)
            print(f"✅ 配置文件加载成功: {self.config_path}")
            return config
        except Exception as e:
            print(f"❌ 配置文件加载失败: {e}")
            return self._get_default_config()

    def _get_default_config(self) -> Dict[str, Any]:
        """Get default configuration."""
        return {
            'model': {
                'type': 'LSTM'
            },
            'training': {
                'sequence_length': 60,
                'batch_size': 32,
                'learning_rate': 0.004,
                'epochs': 100,
                'early_stopping_patience': 20,
                'random_seed': 42
            },
            'data': {
                'file': 'data/000001_daily_qfq_8y.csv',
                'use_blog_style': True,
                'split_ratio': 0.9
            },
            'prediction': {
                'days': 20,
                'risk_free_rate': 0.03
            },
            'output': {
                'dir': 'output',
                'save_visualizations': True,
                'save_model': True,
                'save_config': True
            }
        }

    def get_flattened_config(self) -> Dict[str, Any]:
        """
        Get flattened configuration for backward compatibility.

        Returns:
            Flattened configuration dictionary
        """
        flattened = {}

        # Model configuration
        model = self.config.get('model', {})
        model_type = model.get('type', 'LSTM')

        flattened['model_type'] = model_type

        if model_type == 'LSTM':
            lstm_config = model.get('lstm', {})
            flattened.update({
                'hidden_size1': lstm_config.get('hidden_size1', 50),
                'hidden_size2': lstm_config.get('hidden_size2', 64),
                'fc1_size': lstm_config.get('fc1_size', 32),
                'fc2_size': lstm_config.get('fc2_size', 16),
                'dropout': lstm_config.get('dropout', 0.0)
            })
        elif model_type == 'Transformer':
            transformer_config = model.get('transformer', {})
            flattened.update({
                'd_model': transformer_config.get('d_model', 128),
                'nhead': transformer_config.get('nhead', 8),
                'num_encoder_layers': transformer_config.get('num_encoder_layers', 6),
                'dim_feedforward': transformer_config.get('dim_feedforward', 512),
                'transformer_dropout': transformer_config.get('dropout', 0.1)
            })

        # Training configuration
        training = self.config.get('training', {})
        flattened.update({
            'sequence_length': training.get('sequence_length', 60),
            'batch_size': training.get('batch_size', 32),
            'learning_rate': training.get('learning_rate', 0.004),
            'epochs': training.get('epochs', 100),
            'early_stopping_patience': training.get('early_stopping_patience', 20),
            'random_seed': training.get('random_seed', 42)
        })

        # Data configuration
        data = self.config.get('data', {})
        flattened.update({
            'data_file': data.get('file', 'data/000001_daily_qfq_8y.csv'),
            'use_blog_style': data.get('use_blog_style', True),
            'split_ratio': data.get('split_ratio', 0.9)
        })

        # Prediction configuration
        prediction = self.config.get('prediction', {})
        flattened.update({
            'prediction_days': prediction.get('days', 20),
            'risk_free_rate': prediction.get('risk_free_rate', 0.03)
        })

        # Output configuration
        output = self.config.get('output', {})
        flattened.update({
            'output_dir': output.get('dir', 'output')
        })

        return flattened

    def get_model_config(self) -> Dict[str, Any]:
        """
        Get model-specific configuration.

        Returns:
            Model configuration dictionary
        """
        model = self.config.get('model', {})
        model_type = model.get('type', 'LSTM')

        if model_type == 'LSTM':
            return model.get('lstm', {})
        elif model_type == 'Transformer':
            return model.get('transformer', {})
        else:
            raise ValueError(f"Unknown model type: {model_type}")

    def validate_config(self) -> bool:
        """
        Validate configuration.

        Returns:
            True if configuration is valid
        """
        model_type = self.config.get('model', {}).get('type', 'LSTM')

        if model_type not in ['LSTM', 'Transformer']:
            print(f"❌ 无效的模型类型: {model_type}")
            return False

        # Validate model-specific parameters
        if model_type == 'LSTM':
            lstm_config = self.config.get('model', {}).get('lstm', {})
            required_params = ['hidden_size1', 'hidden_size2', 'fc1_size', 'fc2_size']
            for param in required_params:
                if param not in lstm_config:
                    print(f"❌ LSTM 缺少必需参数: {param}")
                    return False

        elif model_type == 'Transformer':
            transformer_config = self.config.get('model', {}).get('transformer', {})
            required_params = ['d_model', 'nhead', 'num_encoder_layers', 'dim_feedforward']
            for param in required_params:
                if param not in transformer_config:
                    print(f"❌ Transformer 缺少必需参数: {param}")
                    return False

            # Validate transformer-specific constraints
            d_model = transformer_config.get('d_model', 128)
            nhead = transformer_config.get('nhead', 8)
            if d_model % nhead != 0:
                print(f"❌ Transformer d_model ({d_model}) 必须能被 nhead ({nhead}) 整除")
                return False

        print(f"✅ 配置验证通过")
        return True

    def save_config(self, path: str):
        """Save configuration to file."""
        with open(path, 'w', encoding='utf-8') as f:
            yaml.dump(self.config, f, default_flow_style=False, allow_unicode=True)
        print(f"✅ 配置已保存到: {path}")


# Global instance for easy access
_config_loader = None


def get_config_loader(config_path: Optional[str] = None) -> MultiModelConfigLoader:
    """Get configuration loader instance."""
    global _config_loader
    if _config_loader is None or config_path is not None:
        _config_loader = MultiModelConfigLoader(config_path)
    return _config_loader


def get_flat_config(config_path: Optional[str] = None) -> Dict[str, Any]:
    """Get flattened configuration for backward compatibility."""
    return get_config_loader(config_path).get_flattened_config()


def get_model_config(config_path: Optional[str] = None) -> Dict[str, Any]:
    """Get model-specific configuration."""
    return get_config_loader(config_path).get_model_config()


if __name__ == "__main__":
    # Test configuration loading
    loader = get_config_loader()
    print("Configuration loaded successfully!")
    print("Available models:", ['LSTM', 'Transformer'])
    print("Flattened config keys:", list(loader.get_flattened_config().keys()))