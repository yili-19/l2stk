from core.base_processor import BaseProcessor
from transformers import AutoTokenizer
import sys
import os
from src.utils import load_aoa_dataset,load_multiplication_dataset
class DefaultProcessor(BaseProcessor):
    def __init__(self, config):
        print('processor init start')
        print(config)
        self.config = config
        print('processor init end')

    def process(self, inputs): 
        args = self.config
        tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name, padding_side="left")
        tokenizer.pad_token = tokenizer.unk_token
        path_dict = {
        'train': args.train_dataset,
        'test': args.test_dataset
        }
        print("Processing start")
        if '/aoa/' in args.train_dataset:
            raw_datasets = load_aoa_dataset(path_dict, tokenizer, args.mode)
        elif '/ma/' in args.train_dataset or '/dr/' in args.train_dataset or '/gsm/' in args.train_dataset:
            path_dict.pop('test')
            raw_datasets = load_multiplication_dataset(path_dict, tokenizer, args.mode)
        else:
            raise ValueError(f"Unknown dataset: {args.train_dataset}")

        if args.data_num > 0:
            raw_datasets['train'] = raw_datasets['train'].select(range(args.data_num))
        train_dataset = raw_datasets["train"]
        print("Processing end")
        return train_dataset