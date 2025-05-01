from transformers import AutoModelForCausalLM, AutoTokenizer
from strategies.tale.src.load_model import load_model
from core.base_model import BaseModel
import logging
import time
import requests
import random
logger = logging.getLogger(__name__)
class DefaultModel(BaseModel):
    def __init__(self, args):
        """
        Initialize the model API config.

        Args:
            args: Object with attributes like api_key, model_path (used as model_name), strategy, etc.
        """
        self.api_key = args.api_key
        self.model_name = args.model_path

        print(f"Initialized OpenAI API model: {self.model_name}")

    def generate(self, inputs):
        """
        Generate outputs based on inputs using OpenAI API.

        Args:
            inputs (str): The prompt string to send to the model.

        Returns:
            str: The generated response.
        """
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}"
        }

        data = {
            "model": self.model_name,
            "messages": [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": inputs}
            ],
            "max_tokens": 512,
            "seed": 1024,
        }

        try:
            response = requests.post("https://api.openai.com/v1/chat/completions",
                                     headers=headers, json=data)
            if response.status_code == 200:
                result = response.json()
                return result['choices'][0]['message']['content']
            else:
                logger.info(f"Error: {response.status_code} - {response.text}")
                time.sleep(random.uniform(1, 3))
                return None
        except Exception as e:
            logger.error(f"Exception during OpenAI request: {e}")
            return None
