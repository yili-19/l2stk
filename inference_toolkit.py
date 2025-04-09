# toolkit.py

import importlib
import sys
import os
from models.custom_model import CustomModel


class InferenceToolkit:
    def __init__(self, model_config, strategy_config=None):
        """
        初始化InferenceToolkit，传入模型配置和策略配置
        :param model_config: 模型配置
        :param strategy_config: 推理策略配置
        """
        self.model = self._load_model(model_config)
        self.strategy = None

        # 如果有策略配置，则加载策略
        if strategy_config:
            self.strategy = self._load_strategy(strategy_config)
    
    def _load_model(self, model_config):
        """根据配置加载模型"""
        model_type = model_config.get('type', 'custom')  # 默认是自定义模型
        model_module = model_config.get('module', 'default')  # 默认模型

        try:
            # 如果是自定义模型，允许传入自定义的推理方法
            if model_type == 'custom':
                infer_method = model_config.get('infer_method')  # 自定义的推理方法
                api_key = model_config.get('api_key')
                model_name = model_config.get('model_name')
                model = CustomModel(api_key,model_name)
            else:
                # 动态导入预设模型模块
                model_module_path = f'{model_module}.{model_type}_model' # import models.pth
                model_class = getattr(importlib.import_module(model_module_path))
                model = model_class(**model_config.get('params', {}))

            return model
        except (ModuleNotFoundError, AttributeError) as e:
            print(f"Error: Could not load model '{model_type}''. {e}")
            return None

    def _load_strategy(self, strategy_config):
        """根据策略配置加载策略"""
        strategy_type = strategy_config.get('strategy', 'default')
        strategy_module = strategy_config.get('module', 'strategy')

        try:
            strategy_class_name = f'{strategy_type.capitalize()}Strategy'
            strategy_module_path = f'{strategy_module}.{strategy_type}_strategy' # import strategy.pth
            strategy_class = getattr(importlib.import_module(strategy_module_path), strategy_class_name)
            return strategy_class(**strategy_config.get(f'{strategy_type}_params', {}))
        except (ModuleNotFoundError, AttributeError) as e:
            print(f"Error: Could not load strategy '{strategy_class_name}' from module '{strategy_module_path}'. {e}")
            return None
    
    def run_inference(self, input_data):
        """根据配置的策略进行推理"""
        if self.strategy:
            input_data = self.strategy.apply(input_data, self.model)
        return self.model.infer(input_data)