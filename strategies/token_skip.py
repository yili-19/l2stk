# strategies/token_skip.py
# ref TokenSkip: https://github.com/hemingkx/TokenSkip
import time
from utils import Config, Example, compose_request, load_yaml_config_from_path
from peft import PeftModel
from strategies.token_skip.eval.utils import generate_completions

class TokenSkipStrategy:
    def __init__(self, model_type="llama3", compression_ratio=0.5, adapter_path=None, output_type="text_only"):
        self.model_type = model_type
        self.compression_ratio = compression_ratio
        self.adapter_path = adapter_path
        self.output_type = output_type

    def apply(self, input_data, model):
        if self.model_type not in ["qwen", "llama3"]:
            raise ValueError("Unsupported model type. Only 'qwen' is supported.")

        
        prompt = ""
        for mess in input_data['messages']:
            if mess['role'] == 'user':
                if self.model_type == 'llama3':
                    if self.compression_ratio < 1.0:
                        prompt += (
                            f"{tokenizer.bos_token}<|start_header_id|>user<|end_header_id|>\n\n"
                            f"Please reason step by step, and put your final answer within \\boxed{{}}.\n"
                            f"{mess['content']}\n"
                            f"{tokenizer.eos_token}{self.compression_ratio}{tokenizer.eos_token}{tokenizer.eos_token}"
                            "<|start_header_id|>assistant<|end_header_id|>\n\n"
                        )
                    else:
                        prompt += (
                            f"{tokenizer.bos_token}<|start_header_id|>user<|end_header_id|>\n\n"
                            f"Please reason step by step, and put your final answer within \\boxed{{}}.\n"
                            f"{mess['content']}\n"
                            f"{tokenizer.eos_token}<|start_header_id|>assistant<|end_header_id|>\n\n"
                        )
                elif self.model_type == 'qwen':
                    if self.compression_ratio < 1.0:
                        prompt += (
                            "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n"
                            "<|im_start|>user\nPlease reason step by step, and put your final answer within \\boxed{}.\n"
                            f"{mess['content']}<|eot_id|>{self.compression_ratio}<|eot_id|><|im_end|>\n<|im_start|>assistant\n"
                        )
                    else:
                        prompt += (
                            "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n"
                            "<|im_start|>user\nPlease reason step by step, and put your final answer within \\boxed{}.\n"
                            f"{mess['content']}<|im_end|>\n<|im_start|>assistant\n"
                        )
                else:
                    raise NotImplementedError()
            elif mess['role'] == 'assistant':
                prompt += mess['content'].rstrip()
        prompt = prompt.lstrip()

        input_data['prompt'] = prompt

        if self.adapter_path:
            model = PeftModel.from_pretrained(model, self.adapter_path)
            model = model.merge_and_unload()


        # set padding side to left for batch generation
        tokenizer.padding_side = "left"
        # set pad token to eos token if pad token is not set (as is the case for llama models)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            tokenizer.pad_token_id = tokenizer.eos_token_id

        stop_id_sequences = []
        if tokenizer.eos_token_id is not None:
            stop_id_sequences = [[tokenizer.eos_token_id]]

        torch.cuda.synchronize()
        start_time = time()
        outputs, finish_completion = generate_completions(
            model=model,
            tokenizer=tokenizer,
            prompts=prompts,
            max_new_tokens=args.max_new_tokens,
            do_sample=False,
            temperature=args.temperature,
            top_p=1.0,
            batch_size=args.eval_batch_size,
            stop_id_sequences=stop_id_sequences if stop_id_sequences else None,
            end_of_generation_id_sequence=[tokenizer.eos_token_id] if tokenizer.eos_token_id is not None else None
        )
        torch.cuda.synchronize()
        total_time = time() - start_time

        model_outputs = outputs
        cot_lengths = []
        for model_completion in model_outputs:
            cot = model_completion.split('\n\nThe final answer is:')[0]
            cot_length = tokenizer(cot, return_tensors="pt")['input_ids'].shape[1]
            cot_lengths.append(cot_length)

        predictions = [eval(answer_extraction_fn)(item['messages'][-2]['content'], output, task='cot') for item, output in tqdm(zip(test_data, model_outputs), desc="extract answer", total=len(model_outputs))]
        assert len(model_outputs) > 0, f"{len(model_outputs)}"

        results = []
        for example, output, pred, cot_length in zip(test_data, model_outputs, predictions, cot_lengths):
            item = deepcopy(example)
            item.update({
                'model_output': output,
                'prediction': pred,
                'cot_length': cot_length,
            })
            results.append(item)

        if self.output_type == "text_only":
            processed_input = [item['model_output'] for item in results]
        elif self.output_type == "json":
            processed_input = [item for item in results]
        else:
            raise ValueError("Invalid output type. Choose 'text_only' or 'json'.")
        return processed_input