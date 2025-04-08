from datasets import load_dataset

def load_gsm8k(split='test'):
    dataset = load_dataset("openai/gsm8k", "main")
    return dataset[split]

def extract_answer_label(example):
    return example['answer'].split('####')[1].strip()