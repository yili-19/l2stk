from core.base_processor import BaseProcessor
from datasets import Dataset
import sys
import os
from strategies.concise_reasoning.src.reasoning_generation import generate

class DpoProcessor(BaseProcessor):
    def __init__(self, config):
        print('processor init start')
        print(config)
        self.config = config
        print('processor init end')

    def process(self, inputs):
        print("Processing start")
        data = inputs
        cleaned_data = [{
            "prompt": item['prompt'],
            "chosen": item['chosen'],
            "rejected": item['rejected']
        } for item in data]

        def format_example(example):
            return {"prompt": f"{example['prompt']}",
                    "chosen": f"{example['chosen']}",
                    "rejected": f"{example['rejected']}"}

        train_dataset = Dataset.from_list([format_example(d) for d in cleaned_data])
        print("Processing end")
        return train_dataset