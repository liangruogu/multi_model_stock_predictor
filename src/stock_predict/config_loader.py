"""
配置文件加载器 (Configuration Loader)

支持YAML配置文件的加载和验证。
"""

import yaml
import os
from typing import Dict, Any, Optional
from pathlib import Path


class ConfigLoader:
    """配置文件加载器"""

    def __init__(self, config_path: Optional[str] = None):
        """
        初始化配置加载器

        Args:
            config_path: 配置文件路径，默认为项目根目录下的config.yaml
        """
        if config_path is None:
            # 获取项目根目录
            current_dir = Path(__file__).parent.parent.parent
            config_path = current_dir / "config.yaml"

        self.config_path = Path(config_path)
        self.config = {}

    def load_config(self) -> Dict[str, Any]:
        """
        加载YAML配置文件

        Returns:
            配置字典
        """
        if not self.config_path.exists():
            raise FileNotFoundError(f"配置文件不存在: {self.config_path}")

        try:
            with open(self.config_path, 'r', encoding='utf-8') as f:
                self.config = yaml.safe_load(f)

            # 验证配置文件
            self._validate_config()

            print(f"✅ 配置文件加载成功: {self.config_path}")
            return self.config

        except yaml.YAMLError as e:
            raise ValueError(f"配置文件格式错误: {e}")
        except Exception as e:
            raise RuntimeError(f"加载配置文件失败: {e}")

    def _validate_config(self):
        """验证配置文件的必要字段"""
        required_sections = ['model', 'training', 'data']

        for section in required_sections:
            if section not in self.config:
                raise ValueError(f"配置文件缺少必要部分: {section}")

        # 验证模型配置
        model_required = ['input_size', 'hidden_size1', 'hidden_size2', 'fc1_size', 'fc2_size']
        for field in model_required:
            if field not in self.config['model']:
                raise ValueError(f"模型配置缺少必要字段: model.{field}")

        # 验证训练配置
        training_required = ['sequence_length', 'batch_size', 'learning_rate', 'epochs']
        for field in training_required:
            if field not in self.config['training']:
                raise ValueError(f"训练配置缺少必要字段: training.{field}")

        # 验证数据配置
        if 'file' not in self.config['data']:
            raise ValueError("数据配置缺少文件路径: data.file")

    def get_flattened_config(self) -> Dict[str, Any]:
        """
        将嵌套的配置字典扁平化，便于与现有代码兼容

        Returns:
            扁平化的配置字典
        """
        if not self.config:
            self.load_config()

        flattened = {}

        # 模型配置
        model = self.config['model']
        flattened.update({
            'input_size': model['input_size'],
            'hidden_size1': model['hidden_size1'],
            'hidden_size2': model['hidden_size2'],
            'fc1_size': model['fc1_size'],
            'fc2_size': model['fc2_size'],
            'dropout': model['dropout'],
            'num_layers': 2,  # 双层LSTM
        })

        # 训练配置
        training = self.config['training']
        flattened.update({
            'sequence_length': training['sequence_length'],
            'batch_size': training['batch_size'],
            'learning_rate': training['learning_rate'],
            'epochs': training['epochs'],
            'early_stopping_patience': training.get('early_stopping_patience', 20),
            'random_seed': training.get('random_seed', 42),
        })

        # 数据配置
        data = self.config['data']
        flattened.update({
            'data_file': data['file'],
            'use_blog_style': data['use_blog_style'],
            'split_ratio': data['split_ratio'],
        })

        # 预测配置
        prediction = self.config['prediction']
        flattened.update({
            'prediction_days': prediction['days'],
        })

        # 金融配置
        financial = self.config.get('financial', {})
        flattened.update({
            'risk_free_rate': financial.get('risk_free_rate', 0.03),
        })

        # 输出配置
        output = self.config.get('output', {})
        flattened.update({
            'output_dir': output.get('directory', 'output'),
        })

        return flattened

    def save_config(self, config_path: Optional[str] = None):
        """
        保存当前配置到文件

        Args:
            config_path: 保存路径，默认为当前配置文件路径
        """
        save_path = config_path or self.config_path

        try:
            with open(save_path, 'w', encoding='utf-8') as f:
                yaml.dump(self.config, f, default_flow_style=False, allow_unicode=True)

            print(f"✅ 配置文件已保存: {save_path}")

        except Exception as e:
            raise RuntimeError(f"保存配置文件失败: {e}")

    def update_config(self, updates: Dict[str, Any]):
        """
        更新配置

        Args:
            updates: 更新的配置字典
        """
        self._deep_update(self.config, updates)

    def _deep_update(self, base_dict: Dict, update_dict: Dict):
        """深度更新字典"""
        for key, value in update_dict.items():
            if key in base_dict and isinstance(base_dict[key], dict) and isinstance(value, dict):
                self._deep_update(base_dict[key], value)
            else:
                base_dict[key] = value

    def print_config(self):
        """打印当前配置"""
        if not self.config:
            print("⚠️  配置未加载")
            return

        print("📋 当前配置:")
        print("=" * 50)
        yaml.dump(self.config, stream=None, default_flow_style=False)
        print("=" * 50)


def load_config(config_path: Optional[str] = None) -> Dict[str, Any]:
    """
    便捷函数：加载配置文件

    Args:
        config_path: 配置文件路径

    Returns:
        配置字典
    """
    loader = ConfigLoader(config_path)
    return loader.load_config()


def get_flat_config(config_path: Optional[str] = None) -> Dict[str, Any]:
    """
    便捷函数：获取扁平化配置

    Args:
        config_path: 配置文件路径

    Returns:
        扁平化的配置字典
    """
    loader = ConfigLoader(config_path)
    return loader.get_flattened_config()


if __name__ == "__main__":
    # 测试配置加载器
    try:
        loader = ConfigLoader()
        config = loader.load_config()
        loader.print_config()

        # 测试扁平化配置
        flat_config = loader.get_flattened_config()
        print(f"\n📊 扁平化配置包含 {len(flat_config)} 个参数")

    except Exception as e:
        print(f"❌ 配置加载失败: {e}")