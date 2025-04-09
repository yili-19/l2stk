import re
from typing import List, Literal, Union

import json
import yaml
from pydantic import BaseModel


class Example(BaseModel):
    question: str
    answer: str


class Config(BaseModel):
    system_prompt: str
    format: str
    fewshot: List[Example]


def load_config(task: Literal["gsm8k", "date"], config: Literal["baseline", "cot", "cod"]) -> Config:
    with open(f"./configs/{task}_{config}.yaml") as f:
        return Config.model_validate(yaml.safe_load(f))

def load_yaml_config_from_path(config_path) -> Config:
    with open(config_path) as f:
        return Config.model_validate(yaml.safe_load(f))

def load_json_config_from_path(config_path) -> Config:
    with open(config_path, "r") as file:
        cot_prompts = json.load(file)
    return cot_prompts

def compose_request(config: Config, shot: int, question: str) -> str:
    request = config.system_prompt + "\n"
    if shot is None:
        shot = len(config.fewshot)
    if shot != 0:
        fewshot = [config.format.format(question=ex.question, answer=ex.answer) for ex in config.fewshot[:shot]]
        request += "\n".join(fewshot) + "\n"
    request += config.format.format(question=question, answer="")
    return request


def nth_percentile(values: list[float], percentile: float) -> float:
    values = sorted(values)
    index = min(round(percentile * len(values)), len(values)) - 1
    return values[index]


def average(values: list[float]) -> float:
    return sum(values) / len(values)


def trimmed_average(values: list[float], percentile: float) -> float:
    values = sorted(values)
    count = round(len(values) * percentile)
    trimmed = values[count : len(values) - count]
    return average(trimmed)


def extract_number_from_string(s: str) -> Union[int, float]:
    match = re.search(r"\d{1,3}(?:,\d{3})*(?:\.\d+)?", s)
    if match:
        number_str = match.group().replace(",", "")  # Remove commas
        return float(number_str) if "." in number_str else int(number_str)
    return None

def format_example(question, cot_content='Think step by step before answering.'):

    example = '''Answer the following question. {}
    
Question: {}

The last line of your response should be of the following format: 'Answer: ($NUMBER)' (without quotes) where NUMBER is your final answer.
'''.format(cot_content, question)
    
    return example