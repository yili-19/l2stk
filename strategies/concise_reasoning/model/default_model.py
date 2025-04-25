from core.base_model import BaseModel
from strategies.concise_reasoning.src.model import load_model_with_flash_attention, load_model
class DefaultModel(BaseModel):
    def __init__(self, config):
        print("start load model")
        if config.use_flash_attention:
            self.model, self.tokenizer = load_model_with_flash_attention(
                config.model_path,
                "auto",
                "train"
            )
        else:
            self.model, self.tokenizer = load_model(
                config.model_path,
                "auto",
                "train"
            )
        print("end load model")

        def generate(self, inputs):
            """Generate outputs based on inputs"""
        pass
    
