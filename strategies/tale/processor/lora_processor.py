from core.base_processor import BaseProcessor
from datasets import Dataset
import sys
import os
from strategies.concise_reasoning.src.reasoning_generation import generate

class LoraProcessor(BaseProcessor):
    def __init__(self, config):
        print('processor init start')
        print(config)
        self.config = config
        print('processor init end')

    def process(self, inputs):
        print("Processing start")
        data = inputs
        cleaned_data = [{
            "question": item['question'],
            "prediction": item['prediction_budget']  
        } for item in data]

        def format_example(example):
            return {"input_text": f"{example['question']}",
                    "target_text": f"{example['prediction']}"}

        train_dataset = Dataset.from_list([format_example(d) for d in cleaned_data])
        print("Processing end")
        return train_dataset