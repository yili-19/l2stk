import strategies
from dataset.reformat import load_dataset_as_dict
from types import SimpleNamespace

from omegaconf import OmegaConf
    
import json
config = OmegaConf.load("/home/liyi/code/l2stk/configs/concise_reasoning.json")

data = load_dataset_as_dict(config.dataset)
processor = strategies.processor(config.processor) 
model = strategies.model(config.model)          
trainer = strategies.trainer(config.trainer)      

processed = processor.process(data)    
trainer.train(model, processed)           