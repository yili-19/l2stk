# base_strategy.py
import importlib

class DefaultStrategy:
    def __init__(self, config):
        self.config = config
        self._modules = {}

    def get(self, key):
        if key in self._modules:
            return self._modules[key]
        
        if key == "processor":
            processor_type = self.config.get("processor", {}).get("type", "DefaultProcessor")
            return self.load_module("processors", processor_type, self.config["processor"].get("config", {}))
        
        if key == "model":
            model_type = self.config.get("model", {}).get("type", "DefaultModel")
            return self.load_module("models", model_type, self.config["model"].get("config", {}))
        
        if key == "trainer":
            trainer_type = self.config.get("trainer", {}).get("type", "DefaultTrainer")
            return self.load_module("trainer", trainer_type, self.config["trainer"].get("config", {}))
        
        if key == "evaluator":
            evaluator_type = self.config.get("evaluator", {}).get("type", "DefaultEvaluator")
            return self.load_module("evaluation", evaluator_type, self.config["evaluator"].get("config", {}))

        # Fallback to default modules
        return ValueError(f"Unknown key: {key}")
    
    def load_module(self, module_folder, module_type, module_config):
        try:
            # 动态加载模块：根据 module_folder 和 module_type 构建模块路径
            module_path = f"{module_folder}.{module_type.lower()}"
            module = importlib.import_module(module_path)
            # 假设每个模块都有一个与模块名称同名的类
            class_name = getattr(module, module_type)
            return class_name(module_config)
        except (ImportError, AttributeError) as e:
            raise ValueError(f"Failed to load {module_type} from {module_folder}: {str(e)}")

    @property
    def processor(self):
        return self.get("processor")

    @property
    def model(self):
        return self.get("model")

    @property
    def trainer(self):
        return self.get("trainer")

    @property
    def evaluator(self):
        return self.get("evaluator")