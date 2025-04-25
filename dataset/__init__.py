import importlib

def dataset(config):
    dataset_type = config["type"]  # 例如 "HFDataset"
    dataset_config = config["config"]

    module_name = dataset_type.lower().replace("dataset", "")
    class_name = dataset_type

    module = importlib.import_module(f".{module_name}", __name__)
    cls = getattr(module, class_name)
    return cls(dataset_config)