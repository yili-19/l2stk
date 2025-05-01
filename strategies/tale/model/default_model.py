from transformers import AutoModelForCausalLM, AutoTokenizer
from strategies.tale.src.load_model import load_model
from core.base_model import BaseModel

class DefaultModel(BaseModel):
    def __init__(self, args):
        """
        Initialize the model and tokenizer based on the configuration.

        Args:
            config: Configuration object that contains model and LoRA paths, and other settings.
        """
        print("start load model")
        if args.strategy == "lora":
            self.model, self.tokenizer = load_model(args.model_path, args.lora_path)
            
        elif args.strategy == "dpo":
            self.tokenizer = AutoTokenizer.from_pretrained(args.model_path, local_files_only=True)
            self.model = AutoModelForCausalLM.from_pretrained(args.model_path, local_files_only=True)
        
        print("end load model")

    

    def generate(self, inputs):
        """Generate outputs based on inputs"""
        
        pass
