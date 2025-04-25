from abc import ABC, abstractmethod
class BaseDataset(ABC):
    def __init__(self, config):
        self.config = config

    @abstractmethod
    def get_data(self):
        """
        return as: 
        [
            {"input": ..., "label": ...},
            ...
        ]
        """
        pass