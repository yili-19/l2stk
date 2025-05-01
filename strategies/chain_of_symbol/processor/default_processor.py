from core.base_processor import BaseProcessor
import sys
import os

class DefaultProcessor(BaseProcessor):
    def __init__(self, config):
        print('processor init start')
        print(config)
        self.config = config
        print('processor init end')

    def process(self, inputs):
        print("Processing start")
        prompt = open('long2short/strategies/chain_of_symbol/src/cos_demo_shuffle_both_1.txt').read()
        pruned_data = 'Answer the question using the following format in these given examples:'+ '\n\n' +prompt+ '\n\n' + 'Question:' + '\n' + inputs + '\nAnswer:\n'
        print("Processing end")
        return pruned_data  # 示例逻辑