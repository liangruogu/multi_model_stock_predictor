# 多模型股票预测系统

一个基于深度学习的综合性股票价格预测和量化分析系统，支持LSTM和Transformer架构，为股票价格预测和金融指标评估提供准确解决方案。

## 📋 项目概述

本系统实现了最先进的多模型股票预测系统，具备以下核心功能：

- **多模型架构**：支持LSTM和Transformer神经网络
- **数据预处理**：先进的数据加载、清洗和标准化，支持博客风格MinMaxScaler
- **智能训练**：完整的工作流程，包含早停、学习率调度和模型验证
- **金融指标**：专业的金融评估，包括夏普比率、索提诺比率、卡尔马比率等
- **丰富可视化**：训练历史、预测结果和模型分析的综合图表
- **未来预测**：多种预测策略（递归、直接、混合）以实现稳健预测

## 📊 数据说明

### 支持的数据文件
- `data/000001_daily_qfq_8y.csv` - 平安银行（000001）8年日线数据
- `data/000063_daily_qfq_8y.csv` - 中兴通讯（000063）8年日线数据
- `data/600031_daily_qfq_8y.csv` - 三一重工（600031）8年日线数据
- `data/600519_daily_qfq_8y.csv` - 贵州茅台（600519）8年日线数据
- `data/601857_daily_qfq_8y.csv` - 中国石油（601857）8年日线数据

### 数据结构
- `trade_date`：交易日期
- `open`：开盘价
- `close`：收盘价（主要预测目标）
- `high`：最高价
- `low`：最低价
- `volume`：成交量

### 数据覆盖
- **历史周期**：2017-10-13 至 2025-09-10
- **预测范围**：最新数据之后的20个交易日
- **更新频率**：日线交易数据

## 🚀 快速开始

### 环境要求
- Python 3.13+
- PyTorch 2.0+
- 推荐使用CUDA加速GPU训练

### 安装依赖
```bash
# 使用uv包管理器（推荐）
uv add pandas numpy scikit-learn torch matplotlib seaborn scipy loguru pyyaml

# 或使用pip
pip install pandas numpy scikit-learn torch matplotlib seaborn scipy loguru pyyaml
```

### 运行系统

#### 1. 查看可用模型
```bash
uv run python main.py --list-models
```

#### 2. 完整流程（推荐）
```bash
# LSTM模型（默认）
uv run python main.py --mode full --model LSTM

# Transformer模型
uv run python main.py --mode full --model Transformer
```

#### 3. 仅训练
```bash
uv run python main.py --mode train --model LSTM
uv run python main.py --mode train --model Transformer
```

#### 4. 仅预测（需要已训练模型）
```bash
uv run python main.py --mode predict --model LSTM
uv run python main.py --mode predict --model Transformer
```

#### 5. 自定义数据和配置
```bash
uv run python main.py --model LSTM --data data/600519_daily_qfq_8y.csv --config config.yaml
```

## 📁 项目架构

```
LSTM_predict/
├── src/                           # 源代码目录
│   └── stock_predict/             # 主包
│       ├── __init__.py           # 包初始化
│       ├── core.py               # 多模型协调器
│       ├── model_registry.py     # 模型工厂和注册器
│       ├── base_models.py        # 抽象基类
│       ├── lstm_model.py         # LSTM模型和训练器
│       ├── transformer_model.py  # Transformer模型和训练器
│       ├── data_preprocessor.py  # 高级数据预处理
│       ├── prediction.py         # 多策略预测
│       ├── evaluation.py         # 金融指标评估
│       ├── visualization.py      # 丰富可视化套件
│       ├── config_loader.py      # YAML配置管理
│       └── config_multi.py       # 多模型配置
├── data/                         # 数据文件目录
│   ├── 000001_daily_qfq_8y.csv  # 示例股票数据
│   └── [其他股票文件]
├── config.yaml                   # YAML配置文件
├── main.py                      # CLI入口点
├── pyproject.toml               # Python包配置
└── README.md                    # 项目文档（英文）
└── README_CN.md                 # 项目文档（中文）
```

## 🎯 模型架构

### LSTM网络结构
- **输入层**：序列长度60天，特征维度1（收盘价）
- **架构**：双层LSTM配合三层全连接层
  - LSTM1：50个隐藏单元
  - LSTM2：64个隐藏单元
  - FC1：32单元，FC2：16单元，输出：1单元
- **正则化**：可配置dropout（默认：0.0）
- **推荐学习率**：0.004

### Transformer网络结构
- **输入层**：序列长度60天，特征维度1
- **架构**：标准仅编码器Transformer
  - 模型维度：128（可配置）
  - 注意力头数：8
  - 编码器层数：6
  - 前馈网络维度：512
- **位置编码**：序列位置的正弦编码
- **正则化**：Dropout（默认：0.1）
- **推荐学习率**：0.001配合自适应调度

## ⚙️ 配置系统

### YAML配置（`config.yaml`）
```yaml
# 模型选择和参数
model:
  type: "LSTM"  # 或 "Transformer"
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

# 训练参数
training:
  sequence_length: 60
  batch_size: 32
  learning_rate: 0.0002
  epochs: 100
  early_stopping_patience: 30

# 数据配置
data:
  file: "data/000001_daily_qfq_8y.csv"
  use_blog_style: true
  split_ratio: 0.9

# 预测设置
prediction:
  days: 20
  risk_free_rate: 0.03
```

### 输出组织
系统为每次运行创建带时间戳的输出目录：
```
output/{股票代码}_{模型类型}_{时间戳}/
├── {模型类型}_stock_model.pth      # 训练好的模型
├── {模型类型}_predictions.csv      # 未来预测
├── config.json                       # 运行配置
├── training_history.json             # 训练指标
├── {模型类型}_report_training_history.png
├── {模型类型}_report_price_trend_with_test.png
├── training_fit.png
└── training_set_comparison.png
```

## 📈 金融指标与评估

### 预测准确性指标
- **MSE**：均方误差
- **RMSE**：均方根误差
- **MAE**：平均绝对误差
- **MAPE**：平均绝对百分比误差
- **方向准确率**：价格方向预测准确率

### 风险收益分析
- **夏普比率**：风险调整收益表现
- **索提诺比率**：下行风险调整收益
- **卡尔马比率**：年化收益/最大回撤
- **最大回撤**：峰值到谷底下跌
- **年化波动率**：收益标准差

## 🔮 预测策略

### 多种预测方法
1. **递归**：传统逐步预测，存在误差累积
2. **直接**：始终从原始序列预测，无误差累积
3. **混合**：两种方法的加权组合（推荐）

### 使用示例
```python
from src.stock_predict import MultiModelStockPredictor

# LSTM混合预测
predictor = MultiModelStockPredictor({
    'model_type': 'LSTM',
    'data_file': 'data/600519_daily_qfq_8y.csv',
    'learning_rate': 0.004
})
predictor.run_full_pipeline()

# Transformer自定义配置
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

## 🎨 高级特性

### 模型注册系统
- 动态模型注册和创建
- 可扩展架构，便于添加新模型
- 工厂模式的模型实例化

### 智能学习率调度
- **LSTM**：固定学习率（可配置）
- **Transformer**：自适应`ReduceLROnPlateau`，10个epoch耐心值
- 基于验证性能的自动学习率调整

### 综合可视化
- 训练历史曲线（训练/验证损失）
- 增强的价格趋势与测试数据对比
- 训练拟合分析和位移检测
- 历史+测试+未来预测整合

### 时间戳输出
- 每次运行创建唯一时间戳目录
- 保存配置的完整可重现性
- 组织化的工件管理

## 🔧 性能优化

### 模型特定调优
- **LSTM**：较高学习率效果良好（0.004）
- **Transformer**：受益于自适应调度和较低初始率
- **内存管理**：梯度裁剪和高效数据加载

### 数据处理效率
- 博客风格MinMaxScaler配合滞后特征
- 优化的序列生成
- 智能训练/验证/测试分割

## ⚠️ 重要免责声明

1. **教育目的**：本系统仅供研究和教育使用，不构成投资建议
2. **市场不确定性**：股票预测存在固有不确定性和风险
3. **数据质量**：结果取决于高质量、完整的历史数据
4. **模型局限性**：没有模型能完美预测市场走势
5. **风险管理**：实际交易中务必使用适当的风险管理

## 🔍 故障排除指南

### 常见问题
1. **内存不足**：减少`batch_size`或使用CPU训练
2. **收敛缓慢**：调整学习率或模型架构
3. **预测不佳**：尝试不同模型类型或预测策略
4. **数据加载失败**：验证文件路径和CSV格式合规性

### 性能技巧
- 使用GPU加速（如果可用）
- 尝试不同序列长度
- 使用混合预测提高稳定性
- 监控训练损失优化早停设置

## 🏗️ 添加新模型

系统使用**模型注册模式**，便于添加新模型类型：

### 📋 注册流程

1. **创建模型文件**：在`src/stock_predict/`中创建新文件（如`gru_model.py`）
2. **实现必需类**：
   - 继承`BaseStockModel`的模型类
   - 继承`BaseModelTrainer`的训练器类
3. **注册到注册器**：在`model_registry.py`中添加导入和注册
4. **更新包导出**：如需要，添加到`__init__.py`

### 🔧 必需方法

**模型类必须实现：**
- `forward(self, x: torch.Tensor) -> torch.Tensor`
- `get_model_params(self) -> Dict[str, Any]`
- `from_params(cls, params: Dict[str, Any]) -> BaseStockModel`

**训练器类必须实现：**
- `train(self, train_loader, val_loader, epochs, early_stopping_patience)`
- `predict(self, data_loader) -> Tuple[torch.Tensor, torch.Tensor]`
- `save_model(self, path, train_losses, val_losses, model_params, config)`
- `load_model(cls, path, device)`

### 🚀 注册步骤

1. **导入你的类**到`model_registry.py`
2. **在`register_all_models()`函数中添加注册**：
   ```python
   ModelRegistry.register_model(
       name="YourModel",
       model_class=YourModelClass,
       trainer_class=YourTrainerClass
   )
   ```
3. **更新包导出**（可选）到`__init__.py`

### 🎯 注册后使用

新模型立即可用：
```bash
# 列出模型（包含你的）
uv run python main.py --list-models

# 使用你的模型
uv run python main.py --model YourModel --mode full
```

### ✅ 优势

- **零配置更改**：模型与现有CLI完美配合
- **自动集成**：完整流水线支持（训练、预测、评估）
- **一致接口**：与现有模型相同的方法
- **可扩展架构**：便于添加更多模型

## 📝 开发历史

- **2025-10-19**：重大多模型架构升级
  - 实现Transformer模型支持
  - 添加模型注册和工厂模式
  - 增强YAML配置系统
  - 改进可视化和报告
  - 优化两个模型的学习率调度

- **2025-10-16**：完整系统重构
  - 重构为专业Python包布局
  - 添加综合金融指标评估
  - 实现博客风格数据预处理
  - 创建时间戳输出管理

## 📄 许可证

本项目采用MIT许可证 - 详见LICENSE文件。

## 🤝 贡献

欢迎贡献！请随时提交Pull Request或开启Issue讨论。

### 开发环境设置
```bash
# 克隆仓库
git clone <repository-url>
cd LSTM_predict

# 安装开发依赖
uv add --dev pytest black flake8 mypy

# 运行测试
pytest

# 代码格式化
black src/ main.py

# 类型检查
mypy src/
```

---

**❤️ 为量化金融研究和教育而构建**