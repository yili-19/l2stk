from core.base_trainer import BaseTrainer
import os
import json
from transformers import Trainer, TrainingArguments
from trl import DPOConfig, DPOTrainer
from peft import get_peft_model, LoraConfig, TaskType
from typing import Optional
from dataclasses import dataclass, asdict, field

@dataclass
class DpoTrainingConfig:
    # Dataset configuration
    dataset: str
    model_name: str
    output_dir: str
    type: str
    data_path: str = './data'
    percentage: float = 100
    target_total: Optional[int] = None
    
    # LoRA-specific configuration
    r: int = 8
    lora_alpha: int = 32
    target_modules: list = field(default_factory=lambda: ["q_proj", "v_proj"])
    
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


class DpoTrainer(BaseTrainer):
    def __init__(self, config):
        super().__init__(config=config, model=None, tokenizer=None)
        self.config = config

    def train(self, model, dataset):
        self.model = model
        self.tokenizer = model.tokenizer

        lora_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=self.config.r,
            lora_alpha=self.config.lora_alpha,
            target_modules=self.config.target_modules
        )
        self.model = get_peft_model(self.model, lora_config)

        # Tokenize dataset
        self.train_dataset = dataset

        output_dir = os.path.join(self.config.output_dir, 'ckpts')
        logging_dir = os.path.join(self.config.output_dir, 'logs') if not self.config.no_log else None
        os.makedirs(logging_dir, exist_ok=True) if logging_dir else None

        training_args = DPOConfig(
            output_dir=output_dir,
            log_level="info",
            logging_dir=logging_dir,
            per_device_train_batch_size=self.config.batch_size,
            gradient_accumulation_steps=self.config.grad_steps,
            num_train_epochs=self.config.num_train_epochs,
            learning_rate=self.config.learning_rate,
            logging_steps=self.config.logging_steps,
            save_strategy="steps",
            save_steps=1,
            save_total_limit=self.config.save_total_limit,
            weight_decay=0.001,
            max_length=self.config.max_new_tokens,
            seed=self.config.seed,
            logging_first_step=True,
            beta=0.5,
            max_grad_norm=5.0
        )

        trainer = DPOTrainer(
            model=self.model,
            args=training_args,
            processing_class=self.tokenizer,
            ref_model=None,
            train_dataset=self.train_dataset,
            eval_dataset=None,
        )

        trainer.train()
