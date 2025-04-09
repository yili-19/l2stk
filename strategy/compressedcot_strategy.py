# strategies/cod_strategy.py
# ref compressed-cot: https://github.com/Compressed-CoT/compressed-cot/tree/main
import time
from utils import Config, Example, compose_request, load_json_config_from_path, format_example

class CompressedCotStrategy:
    def __init__(self, shot=0, prompt_strategy="NoCoT", config_path="./configs/compressed-cot/cot_prompts.json"):
        self.shot = shot
        self.prompt_strategy = prompt_strategy
        self.config_path = config_path
    
    def apply(self, input_data, model):
        config = load_json_config_from_path(self.config_path)

        prompt_instruction = config.get(self.prompt_strategy, "")
        if not prompt_instruction:
            print(f"[Warning] Prompt strategy '{self.prompt_strategy}' not found. Using empty prompt.")

        processed_input = format_example(question=input_data, cot_content=prompt_instruction)
        
        return processed_input