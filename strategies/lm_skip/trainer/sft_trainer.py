# flake8: noqa
import os
import json
import logging
from dataclasses import dataclass, field, asdict
from typing import Optional, Dict
from datetime import datetime
from transformers import Trainer, TrainingArguments, AutoTokenizer
from datasets import Dataset
import pandas as pd
from tqdm import tqdm
from trl import (
    ModelConfig,
    RichProgressCallback,
    SFTTrainer,
    get_peft_config,
    get_quantization_config,
    get_kbit_device_map,
    DataCollatorForCompletionOnlyLM,
    set_seed
)
from core.base_trainer import BaseTrainer
from src.utils import load_data_from_file, load_txt, is_main_process

IGNORE_INDEX = -100

@dataclass
class SftTrainingConfig:
    # Dataset configuration
    dataset: str
    model_name: str
    output_dir: str
    mode: str
    data_path: Dict[str, str]
    dataset_type: str = 'aoa'  # 'aoa' or 'multiplication'

    # Training hyperparameters
    batch_size: int = 16
    grad_steps: int = 1
    learning_rate: float = 1e-5
    num_train_epochs: float = 1.0
    max_steps: int = -1
    seed: int = 0

    # Logging and saving
    logging_steps: int = 10
    save_strategy: str = "epoch"
    save_steps: int = 500
    save_total_limit: int = 3
    report_to: Optional[str] = None

    # Hardware configuration
    bf16: bool = False
    gradient_checkpointing: bool = False
    use_peft: bool = False
    lora_r: Optional[int] = None
    lora_alpha: Optional[int] = None

    # Quantization
    quantization_bit: Optional[int] = None

    # Other configurations
    use_flash_attention: bool = False
    push_to_hub: bool = False
    no_log: bool = False

    def __post_init__(self):
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir, exist_ok=True)

    @classmethod
    def from_json(cls, json_path: str):
        with open(json_path, 'r') as f:
            cfg = json.load(f)
        return cls(**cfg)

    def to_dict(self):
        return asdict(self)


class SftTrainer(BaseTrainer):
    def __init__(self, config):
        self.config = config

    def _formatting_function(self, example):
        text = f"{example['input']}{example['output']}"
        return {'text': text}

    def tokenize_function(self, example):
        inputs = example['text']
        model_inputs = self.tokenizer(
            inputs, truncation=True, max_length=2048, padding="max_length", add_special_tokens=True
        )
        input_ids = model_inputs['input_ids']
        labels = input_ids.copy()
        model_inputs["labels"] = labels
        return model_inputs

    def train(self, model, dataset):
        self.model = model.model
        self.tokenizer = model.tokenizer

        dataset_bundle = self.load_dataset(self.tokenizer)
        train_dataset = dataset_bundle['train']

        train_dataset = train_dataset.map(self._formatting_function, remove_columns=['input', 'output'])
        train_dataset = train_dataset.map(self.tokenize_function, remove_columns=['text'])

        train_dataset = dataset
        output_dir = f"{self.config.output_dir}/ckpts"
        logging_dir = f"{self.config.output_dir}/logs" if not self.config.no_log else None

        training_args = TrainingArguments(
            output_dir=output_dir,
            remove_unused_columns=False,
            logging_dir=logging_dir,
            num_train_epochs=self.config.num_train_epochs,
            lr_scheduler_type="constant",
            per_device_train_batch_size=self.config.batch_size,
            gradient_accumulation_steps=self.config.grad_steps,
            learning_rate=self.config.learning_rate,
            save_strategy=self.config.save_strategy,
            logging_strategy="steps" if not self.config.no_log else "no",
            save_steps=self.config.save_steps if self.config.save_strategy == "steps" else None,
            logging_steps=self.config.logging_steps,
            save_total_limit=self.config.save_total_limit,
            seed=self.config.seed,
            bf16=self.config.bf16,
            gradient_checkpointing=self.config.gradient_checkpointing,
            report_to=self.config.report_to if self.config.report_to else [],
            max_steps=self.config.max_steps,
            push_to_hub=self.config.push_to_hub
        )

        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            tokenizer=self.tokenizer,
        )

        trainer.train()
