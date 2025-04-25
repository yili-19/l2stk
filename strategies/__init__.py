from core.base_processor import BaseProcessor
from core.base_model import BaseModel
from core.base_trainer import BaseTrainer
from utils.load_module import dynamic_load_module


def processor(config) -> BaseProcessor:
    return dynamic_load_module('processor', config, default_cls="PruneProcessor", base_path="strategies")

def model(config) -> BaseModel:
    return dynamic_load_module('model', config, default_cls="PruneModel", base_path=f"strategies")

def trainer(config) -> BaseTrainer:
    return dynamic_load_module('trainer', config, default_cls="PruneTrainer", base_path="strategies")