# strategies/sot_strategy.py
# ref SoT: https://github.com/SimonAytes/SoT
import time
import json
import os
import copy
import torch
from loguru import logger

from pathlib import Path
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
from utils import Config, Example, compose_request, load_json_config_from_path, format_example

# Warning for when an image is passed, but the multimodal format was not specified
MULTIMODAL_MISALIGNMENT = (
    "Image data was passed, but the format is set to `llm`. "
    "Resulting context will not include image data. "
    "Please change format to `vlm` to include image data."
)

NO_IMAGE = (
    "Format was specified as `vlm` but no image data was passed."
    "Resulting multimodal context will not include image data. "
)

def default_path():
    """Return the root path of the toolkit project."""
    return Path(__file__).parent.parent / "configs" / "sot"

class SoTStrategy:
    def __init__(self, ):
        self.sot = SoT()

    def apply(self, input_data, model):
        paradigm = self.sot.classify_question(input_data)
        system_prompt = self.sot.get_system_prompt(paradigm)

        processed_input = self.sot.get_initialized_context(
            paradigm=paradigm, 
            question=input_data, 
            format="llm",
            include_system_prompt=True
        )
        
        return processed_input

class SoT:
    def __init__(self):
        # Set base config path
        self.__CONFIG_BASE = default_path()

        # Load the model from HF
        self.__MODEL_PATH = "saytes/SoT_DistilBERT"
        self.model = DistilBertForSequenceClassification.from_pretrained(self.__MODEL_PATH)
        self.tokenizer = DistilBertTokenizer.from_pretrained(self.__MODEL_PATH)

        # Load the label mapping
        self.__LABEL_MAPPING_PATH = self.__CONFIG_BASE / "label_mapping.json"
        self.__LABEL_MAPPING = json.load(open(self.__LABEL_MAPPING_PATH))

        # Prompt and exemplar base paths
        self.__PROMPT_PATH_BASE = self.__CONFIG_BASE / "prompts"
        self.__CONTEXT_PATH_BASE = self.__CONFIG_BASE / "exemplars.json"

        self.__PROMPT_FILENAMES = {
            "chunked_symbolism": "ChunkedSymbolism_SystemPrompt.md",
            "expert_lexicons": "ExpertLexicons_SystemPrompt.md",
            "conceptual_chaining": "ConceptualChaining_SystemPrompt.md",
        }

        self.PROMPT_CACHE = {}
        self.CONTEXT_CACHE = {}

        self.__preload_contexts()
        self.__LANGUAGE_CODES = list(self.CONTEXT_CACHE.keys())
        self.__preload_prompts()
    
    def __preload_prompts(self):
        """
        Loads all available system prompts into memory at startup.
        """

        for lang in self.__LANGUAGE_CODES:
            self.PROMPT_CACHE[lang] = {}
            for paradigm, filename in self.__PROMPT_FILENAMES.items():
                file_path = os.path.join(self.__PROMPT_PATH_BASE, lang, f"{lang}_{filename}")
                if os.path.exists(file_path):
                    with open(file_path, "r", encoding="utf-8") as file:
                        self.PROMPT_CACHE[lang][paradigm] = file.read()

    def __preload_contexts(self):
        """
        Loads all available contexts into memory at startup.
        """
        with open(self.__CONTEXT_PATH_BASE, "r") as f:
            self.CONTEXT_CACHE = json.load(f)
    
    def available_languages(self):
        """
        Lists all currently supported languages.
        """
        return self.__LANGUAGE_CODES

    def available_paradigms(self):
        """
        Returns list of all currently supported paradigms.
        """
        return list(self.__PROMPT_FILENAMES.keys())

    def get_system_prompt(self, paradigm="chunked_symbolism", language_code="EN"):
        """
        Retrieves the preloaded system prompt based on the given paradigm and language code.
        
        :param paradigm: The type of prompt (e.g., "chunked_symbolism", "expert_lexicons", "conceptual_chaining").
        :param language_code: The language code (e.g., "EN" for English, "KR" for Korean, etc.).
        :return: The content of the corresponding prompt file or None if not found.
        """
        assert paradigm in self.available_paradigms(), f"`{paradigm}` is not a recognized paradigm!"
        assert language_code in self.available_languages(), f"`{language_code}` is not a compatible language!"
        
        return copy.deepcopy(self.PROMPT_CACHE[language_code][paradigm])
    
        
    def get_initialized_context(self, paradigm, question=None, image_data=None, language_code="EN", include_system_prompt=True, format="llm"):
        """
        Retrieves the preloaded conversation context for the given paradigm and language.
        Dynamically inserts the user's question and system prompt.

        :param paradigm: The reasoning paradigm ("conceptual_chaining", "chunked_symbolism", "expert_lexicons", "cot").
        :param question: The user's question to be added to the context. If `None` or empty, it will not be added.
        :param image_data: The image associated with the user's question. Required `format="vlm"`.
        :param language_code: The language code (e.g., "KR" for Korean).
        :param include_system_prompt: Whether to add the system prompt to the context. Not available in raw format.
        :param format: The format to return. Accepted values are: `llm`, `raw`, or `vlm`.
        :return: The full initialized conversation list.
        """

        assert paradigm in self.available_paradigms(), f"`{paradigm}` is not a recognized paradigm!"
        assert language_code in self.available_languages(), f"`{language_code}` is not a compatible language!"

        if format.lower() == "llm":
            # Warn for multimodal misalignment
            if image_data:
                logger.warning(MULTIMODAL_MISALIGNMENT)
            
            exemplars = self.CONTEXT_CACHE[language_code][paradigm]
            if include_system_prompt:
                context = [{"role": "system", "content": self.get_system_prompt(paradigm=paradigm, language_code=language_code)}]
            else:
                context = []

            for ex in exemplars:
                context += [
                    {"role": "user", "content": ex['question']},
                    {"role": "assistant", "content": ex['answer']},
                ]
            
            # Add user question, if it exists
            if question and question != "":
                context += [{"role": "user", "content": question}]

            return context
        
        elif format.lower() == "vlm":
            # Warn for missing image
            if image_data is None:
                logger.warning(NO_IMAGE)
            
            exemplars = self.CONTEXT_CACHE[language_code][paradigm]
            if include_system_prompt:
                context = [{"role": "system", "content": [{"type": "text", "text": self.get_system_prompt(paradigm=paradigm, language_code=language_code)}]}]
            else:
                context = []

            for ex in exemplars:
                context += [
                    {"role": "user", "content": [{"type": "text", "text": ex['question']}]},
                    {"role": "assistant", "content": [{"type": "text", "text": ex['answer']}]},
                ]
            
            # Add user question, if it exists
            if question and question != "":
                context = [{"role": "user", "content": [{"type": "text", "text": question}, {"type": "image", "image": image_data}]}]
            return context
        
        else:  # Default case, return raw format
            return copy.deepcopy(self.CONTEXT_CACHE[language_code][paradigm])
    
    def classify_question(self, question):
        """
        Uses the pretrained DistilBERT model to classify the paradigm of a question.
        """

        inputs = self.tokenizer(question, return_tensors="pt", truncation=True, padding=True)
        outputs = self.model(**inputs)
        predicted_class = torch.argmax(outputs.logits, dim=1).item()
        
        # Reverse mapping to get the paradigm name
        label_mapping_reverse = {v: k for k, v in self.__LABEL_MAPPING.items()}
        return label_mapping_reverse[predicted_class]