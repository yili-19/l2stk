from core.base_processor import BaseProcessor

import sys
import os
from strategies.concise_reasoning.src.reasoning_generation import generate

class DefaultProcessor(BaseProcessor):
    def __init__(self, config):
        print('processor init start')
        print(config)
        self.config = config
        print('processor init end')

    def process(self, inputs):
        print("Processing start")
        generate(inputs, self.config)
        print("Processing end")
        return inputs  # 示例逻辑