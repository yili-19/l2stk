from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer
import logging
logger = logging.getLogger(__name__)

def load_model(self, model_path, lora_path=None):
        """
        Load and prepare the model for training or inference.

        Args:
            model_path: Path to the base model.
            lora_path: Optional path to LoRA weights.

        Returns:
            tuple: (model, tokenizer) where:
                model: Loaded and prepared model.
                tokenizer: Configured tokenizer.
        """
        tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
        if model_path in ['Qwen2.5-14B', 'Qwen2.5-7B-Instruct-1M']:
            tokenizer.padding_side = "left"
            tokenizer.pad_token = tokenizer.eos_token if tokenizer.pad_token is None else tokenizer.pad_token
        base_model = AutoModelForCausalLM.from_pretrained(model_path, local_files_only=True)
        base_model.eval()
        logger.info(f"Load base model from {model_path}")
        logger.info(f"Tokenizer max length: {tokenizer.model_max_length}")
        logger.info(f"Model config max length: {base_model.config.max_position_embeddings}")

        if lora_path is None:
            return base_model, tokenizer
        lora_model = PeftModel.from_pretrained(base_model, lora_path)
        logger.info(f"Load LoRA model from {lora_path} Successfully!")
        merged_model = lora_model.merge_and_unload()
        assert not isinstance(merged_model, PeftModel), "merge_and_unload failed"
        merged_model.half()
        merged_model.eval()
        return merged_model, tokenizer
