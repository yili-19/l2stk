from abc import ABC, abstractmethod

class BaseTrainer(ABC):
    def __init__(self, config, model, processor):
        self.config = config
        self.model = model
        self.processor = processor

    @abstractmethod
    def train(self, dataset):
        """Fine-tune model on the given dataset"""
        pass