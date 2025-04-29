from core.base_trainer import BaseTrainer
import os
import json
import logging
from transformers import Trainer, TrainingArguments
from peft import get_peft_model, LoraConfig, TaskType
from typing import Optional
from dataclasses import dataclass, asdict, field
from datetime import datetime

@dataclass
class LoraTrainingConfig:
    # Dataset and model configuration
    dataset: str
    model_name: str
    output_dir: str
    type: str
    data_path: str = './data'

    # LoRA-specific configuration
    r: int = 8
    lora_alpha: int = 32
    target_modules: list = field(default_factory=lambda: ["q_proj", "v_proj"])
    
    # Training hyperparameters
    batch_size: int = 16
    grad_steps: int = 8
    learning_rate: float = 1e-4
    seed: int = 0

    # Training process configurations
    num_train_epochs: float = 3.0
    save_strategy: str = "steps"
    save_steps: int = 1
    logging_steps: int = 1
    save_total_limit: int = 100
    lr_scheduler_type: str = "constant"

    # Hardware and logging configurations
    fp16: bool = True
    no_log: bool = False

    # Generation configs
    max_new_tokens: int = 512

    # Flash Attention specific configurations
    use_flash_attention: bool = False

    # Whether to use the raw output directory or the organized one
    use_raw_output_dir: bool = False

    # Whether to save model after training
    save: bool = True

    def __post_init__(self):
        if not self.target_modules:
            self.target_modules = ["q_proj", "v_proj"]
        if not self.use_raw_output_dir:
            timestamp = datetime.now().strftime("%Y-%m-%d/%H-%M-%S")
            self.output_dir = os.path.join(
                self.output_dir,
                self.dataset,
                self.model_name,
                self.type,
                f"{self.data_path.split('/')[-1]}",
                timestamp
            )
        if self.use_flash_attention and not self.fp16:
            self.fp16 = True

    @classmethod
    def from_json(cls, json_path: str):
        with open(json_path, 'r') as f:
            cfg = json.load(f)
        return cls(**cfg)

    @classmethod
    def from_dict(cls, cfg: dict):
        return cls(**cfg)

    def to_dict(self):
        return asdict(self)

class LoraTrainer(BaseTrainer):
    def __init__(self, config):
        self.config = config

    def tokenize_data(self, dataset, tokenizer):
        def tokenize_function(examples):
            input_texts = examples["input_text"]
            output_texts = examples["target_text"]
            full_texts = [inp + "\n" + out for inp, out in zip(input_texts, output_texts)]
            tokenized = tokenizer(full_texts, padding="max_length", truncation=True, max_length=2048)
            tokenized["labels"] = tokenized["input_ids"].copy()
            return tokenized
        tokenized_dataset = dataset.map(tokenize_function, batched=True)
        return tokenized_dataset
    
    def train(self, model, dataset):
        self.model = model.model
        self.tokenizer = model.tokenizer

        lora_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=self.config.r,
            lora_alpha=self.config.lora_alpha,
            target_modules=self.config.target_modules
        )
        self.model = get_peft_model(self.model, lora_config)

        self.train_dataset = self.tokenize_data(dataset,self.tokenizer)

        output_dir = os.path.join(self.config.output_dir, 'ckpts')
        logging_dir = os.path.join(self.config.output_dir, 'logs') if not self.config.no_log else None
        os.makedirs(logging_dir, exist_ok=True) if logging_dir else None

        training_args = TrainingArguments(
            output_dir=output_dir,
            remove_unused_columns=False,
            logging_dir=logging_dir,
            num_train_epochs=self.config.num_train_epochs,
            lr_scheduler_type=self.config.lr_scheduler_type,
            per_device_train_batch_size=self.config.batch_size,
            gradient_accumulation_steps=self.config.grad_steps,
            learning_rate=self.config.learning_rate,
            save_strategy=self.config.save_strategy,
            logging_strategy="steps" if not self.config.no_log else "no",
            save_steps=self.config.save_steps if self.config.save_strategy == "steps" else None,
            logging_steps=self.config.logging_steps,
            save_total_limit=self.config.save_total_limit,
            seed=self.config.seed,
            fp16=self.config.fp16,
            report_to='tensorboard' if not self.config.no_log else [],
        )

        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=self.train_dataset,
            processing_class=self.tokenizer,
        )

        trainer.train()

