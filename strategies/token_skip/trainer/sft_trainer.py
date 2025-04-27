from core.base_trainer import BaseTrainer
import os
from datetime import datetime
from dataclasses import dataclass
from transformers import TrainingArguments, Trainer
from transformers.trainer_utils import set_seed
import json
from dataclasses import dataclass, field, asdict
from typing import Optional

# from  strategies.concise_reasoning.src.dataset import GSM8kDatasetLoader, MATHDatasetLoader
# from  strategies.concise_reasoning.src.model import DataCollatorForSupervisedDataset
# IGNORE_INDEX = -100

@dataclass
class SftTrainingConfig:
    # Dataset configuration
    dataset: str
    model_name: str
    output_dir: str
    type: str
    data_path: str = './data'
    percentage: float = 100
    target_total: Optional[int] = None

    # Training hyperparameters
    batch_size: int = 16
    grad_steps: int = 1
    learning_rate: float = 1e-5
    seed: int = 0

    # Training process configurations
    num_train_epochs: float = 1.0
    save_strategy: str = "epoch"
    save_steps: float = 1.0
    logging_steps: int = 16
    save_total_limit: int = 3
    lr_scheduler_type: str = "constant"

    # Hardware and logging configurations
    bf16: bool = False
    no_log: bool = False

    # Generation configs
    max_new_tokens: int = 512

    # Flash Attention specific configurations
    use_flash_attention: bool = False

    # Whether to use the raw output directory or the organized one
    use_raw_output_dir: bool = False

    def __post_init__(self):
        from datetime import datetime
        import os
        if not self.use_raw_output_dir:
            timestamp = datetime.now().strftime("%Y-%m-%d/%H-%M-%S")
            self.output_dir = os.path.join(
                self.output_dir,
                self.dataset,
                self.model_name,
                self.type,
                f"{self.percentage}",
                timestamp
            )
        if self.use_flash_attention and not self.bf16:
            self.bf16 = True

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

class SftTrainer(BaseTrainer):
    def __init__(self, config):
        self.config = config

    def _format_direct_prompt(self, example):
        if "qwen" in self.config.model_name.lower():
            return {"prompt": example['input']}
        return {"prompt": self.tokenizer.bos_token + example['input']}


    def tokenize_function(self, example):
        if "qwen" in self.config.model_name.lower():
            input_text = example['input']
        else:
            input_text = self.tokenizer.bos_token + example['input']

        input_tokenized = self.tokenizer(input_text, add_special_tokens=False)
        input_length = len(input_tokenized['input_ids'])

        full_text = input_text + ' ' + example['label'] + self.tokenizer.eos_token
        model_inputs = self.tokenizer(full_text, add_special_tokens=False)
        labels = model_inputs["input_ids"].copy()
        labels[:input_length] = [IGNORE_INDEX] * input_length

        model_inputs['labels'] = labels
        model_inputs['prompt'] = input_text
        return model_inputs

    def train(self, model, dataset):
        self.model = model.model
        self.tokenizer = model.tokenizer
        if self.config.dataset == 'gsm8k':
            dataset_loader = GSM8kDatasetLoader()
        elif self.config.dataset == 'math':
            dataset_loader = MATHDatasetLoader()
        else:
            raise ValueError(f"Unsupported dataset: {self.config.dataset}")
        
        self.train_dataset = dataset_loader.load_from_dict(
            dataset['train'], dataset['test'],[]
        )['train']

        # self.train_dataset = self.train_dataset.remove_columns(['length', 'gold'])
        self.train_dataset = self.train_dataset.map(
            self.tokenize_function,
            remove_columns=['input', 'label']
        )

        output_dir = f"{self.config.output_dir}/ckpts"
        logging_dir = f"{self.config.output_dir}/logs" if not self.config.no_log else None

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
            bf16=self.config.bf16,
            report_to='tensorboard'
        )

        data_collator = DataCollatorForSupervisedDataset(tokenizer=self.tokenizer)

        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=self.train_dataset,
            data_collator=data_collator,
            tokenizer=self.tokenizer,
        )

        trainer.train()