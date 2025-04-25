from datasets import load_dataset
from omegaconf import OmegaConf
def load_dataset_as_dict(config):
    print("start load dataset")
    data_splits = {"train": [], "val": [], "test": []}
    if config.name == 'gsm8k': 
        if config.config.use_local == False:
            hf_dataset = load_dataset("gsm8k", "main") 
            data_splits["train"] = [{"question": x["question"], "answer": x["answer"]} for x in hf_dataset["train"]]
            data_splits["test"] = [{"question": x["question"], "answer": x["answer"]} for x in hf_dataset["test"]]
            data_splits["val"] = []
        else:
            from datasets import load_dataset
            print(config.config.data_files)
            hf_dataset = load_dataset("json", data_files=OmegaConf.to_container(config.config.data_files, resolve=True))
            data_splits["train"] = [{"question": x["question"], "answer": x["answer"]} for x in hf_dataset["train"]][:5]
            data_splits["test"] = [{"question": x["question"], "answer": x["answer"]} for x in hf_dataset["test"]][:5]
            data_splits["val"] = []

    
    if config.name == 'math': 
        if config.config.use_local == False:
            hf_dataset = load_dataset("math") 
            data_splits["train"] = [{"question": x["problem"], "answer": x["solution"]} for x in hf_dataset["train"]]
            data_splits["test"] = [{"question": x["problem"], "answer": x["solution"]} for x in hf_dataset["test"]]
            data_splits["val"] = []
        else:
            from datasets import load_dataset
            print(config.config.data_files)
            hf_dataset = load_dataset("json", data_files=OmegaConf.to_container(config.config.data_files, resolve=True))
            data_splits["train"] = [{"question": x["problem"], "answer": x["solution"]} for x in hf_dataset["train"]][:5]
            data_splits["test"] = [{"question": x["problem"], "answer": x["solution"]} for x in hf_dataset["test"]][:5]
            data_splits["val"] = []
    print("end load dataset")
    return data_splits
    

    


    