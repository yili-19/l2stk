import importlib
import re

def camel_to_snake(name: str) -> str:
    return re.sub(r'(?<!^)(?=[A-Z])', '_', name).lower()

def snake_to_camel(name: str) -> str:
    return ''.join(word.title() for word in name.split('_'))

def dynamic_load_module(module_type: str, config, default_cls: str, base_path: str):
    module_name = config.name
    module_config = config.config
    strategy_type = config.strategy

    full_path = f"{base_path}.{camel_to_snake(strategy_type)}.{module_type}.{camel_to_snake(module_name)}"

    try:
        module = importlib.import_module(full_path)
        cls = getattr(module, module_name)
        return cls(module_config)
    except Exception as e: 
        raise ImportError(f"Failed to load '{module_name}' from {full_path}") from e