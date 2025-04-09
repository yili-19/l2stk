import requests
import logging
import time
import random
logger = logging.getLogger(__name__)

class CustomModel:
    def __init__(self, api_key, model_name="gpt-4o-mini", **kwargs):
        self.api_key = api_key
        self.model_name = model_name

    def infer(self, input_data):

        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}"
        }

        data = {
            "model": self.model_name,
            "messages": [
                {"role": "system",
                 "content": "You are a helpful assistant."
                 },
                {"role": "user",
                 "content": input_data
                 }
            ],
            "max_tokens": 512,
            "seed": 1024,
        }

        response = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=data)
        if response.status_code == 200:
            responses = response.json()
            answers = []
            answers.append(responses['choices'][0]['message']['content'])
            return answers[0]
        else:
            logger.info(f"Error: {response.status_code} - {response.text}")
            time.sleep(random.uniform(1, 3))

# class CustomModel:
#     def __init__(self, infer_method):
#         self.infer_method = infer_method

#     def infer(self, input_data):
#         # 使用用户自定义的推理方法
#         return self.infer_method(input_data)