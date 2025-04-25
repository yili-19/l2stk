from abc import ABC, abstractmethod

class BaseModel(ABC):
    def __init__(self, config):
        self.config = config

    # @abstractmethod
    # def load(self):
    #     """Load or initialize model"""
    #     pass

    @abstractmethod
    def generate(self, inputs):
        """Generate outputs based on inputs"""
        pass