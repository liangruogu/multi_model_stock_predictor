"""
Multi-Model Stock Prediction System Main Entry Point

Supports both LSTM and Transformer models for stock price prediction.
"""

import argparse
import sys
import os

# Add src directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from src.stock_predict import MultiModelStockPredictor, get_available_models
from src.stock_predict.config_multi import get_flat_config


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description='Multi-Model Stock Price Prediction System')
    parser.add_argument('--mode', choices=['train', 'predict', 'full'], default='full',
                       help='Run mode: train (training only), predict (prediction only), full (complete pipeline)')
    parser.add_argument('--model', choices=get_available_models(), default='LSTM',
                       help=f'Model type: {", ".join(get_available_models())}')
    parser.add_argument('--config', type=str, help='Configuration file path', default="config.yaml")
    parser.add_argument('--data', type=str, help='Data file path')
    parser.add_argument('--model-path', type=str, help='Pre-trained model file path')
    parser.add_argument('--output-dir', type=str, default='output', help='Output directory')
    parser.add_argument('--list-models', action='store_true', help='List available models and exit')

    args = parser.parse_args()

    # List available models if requested
    if args.list_models:
        print("🤖 Available Models:")
        for model in get_available_models():
            print(f"   - {model}")
        return

    print("=" * 80)
    print("🚀 Multi-Model Stock Prediction System")
    print("=" * 80)
    print(f"   Available Models: {', '.join(get_available_models())}")
    print(f"   Selected Model: {args.model}")
    print(f"   Run Mode: {args.mode}")
    print("=" * 80)

    # Load configuration
    try:
        config = get_flat_config(args.config)
        print("✅ 默认YAML配置文件已加载")
    except Exception as e:
        print(f"⚠️  配置加载失败，使用默认配置: {e}")
        config = {}

    # Override configuration with command line arguments
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
    try:
        if args.mode == 'train':
            predictor.run_training_only()
        elif args.mode == 'predict':
            predictor.run_prediction_only()
        else:  # full
            predictor.run_full_pipeline()
    except Exception as e:
        print(f"❌ 执行失败: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()