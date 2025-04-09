# strategies/cod_strategy.py
# ref chain-of-draft: https://github.com/sileix/chain-of-draft
import time
from utils import Config, Example, compose_request, load_yaml_config_from_path

class ChainOfDraftStrategy:
    def __init__(self, shot=0, prompt_strategy="baseline", config_path=None):
        self.shot = shot
        self.prompt_strategy = prompt_strategy
        self.config_path = config_path
    
    def apply(self, input_data, model):
        config = load_yaml_config_from_path(self.config_path)
        processed_input = compose_request(config, self.shot, input_data)
        return processed_input