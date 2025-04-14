# main.py
from OpenL2S import OpenL2S
from models.custom_model import CustomModel
import argparse

# 自定义推理方法
def custom_infer_method(input_data):
    return f"Custom Infer Result for: {input_data}"

def main(args):
    # 自定义模型配置
    model_config = {
        'type': 'custom',
        'infer_method': custom_infer_method, 
        'module': 'models',
        'api_key': args.api_key,
        'model_name':args.model_name
    }

    # 策略配置
    strategy_config = {
        'strategy': args.strategy, 
        'module': 'strategy',
        'params': {
            'threshold': 0.5,
        }
    }

    toolkit = OpenL2S(model_config=model_config, strategy_config=strategy_config)

    # 用户输入数据
    input_data = "A robe takes 2 bolts of blue fiber and half that much white fiber.  How many bolts in total does it take?"
    input_data2 = "There are a set of bricks. The blue brick C is on top of the brick A . For the brick F, the color is blue. The blue brick A is on top of the brick F . The white brick D is on top of the brick C . The blue brick B is on top of the brick E . The blue brick E is on top of the brick D . Now we have to get a specific brick. The bricks must now be grabbed from top to bottom, and if the lower brick is to be grabbed, the upper brick must be removed first. How to get brick F?"

    # 运行推理并获得结果
    output = toolkit.run(input_data2)

    # 输出推理结果
    print("Inference Output:", output)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--api_key", type=str)
    parser.add_argument("--model_name", type=str)
    parser.add_argument("--strategy", type=str, choices=['pruning', 'tale', 'cos'], help="Select strategy to apply.")
    args = parser.parse_args()
    main(args) 