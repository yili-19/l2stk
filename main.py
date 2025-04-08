# main.py
from inference_toolkit import InferenceToolkit
from models.custom_model import CustomModel

# 自定义推理方法
def custom_infer_method(input_data):
    return f"Custom Infer Result for: {input_data}"

# 自定义模型配置
model_config = {
    'type': 'custom',
    'infer_method': custom_infer_method, 
    'module': 'models',
}

# 策略配置
strategy_config = {
    'strategy': 'pruning', 
    'module': 'strategies', 
    'pruning_params': {
        'threshold': 0.5,
    }
}

toolkit = InferenceToolkit(model_config=model_config, strategy_config=strategy_config)

# 用户输入数据
input_data = "What is the capital of France?"

# 运行推理并获得结果
output = toolkit.run_inference(input_data)

# 输出推理结果
print("Inference Output:", output)