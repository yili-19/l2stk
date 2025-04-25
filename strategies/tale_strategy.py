import time
import tiktoken
import logging
from langchain.prompts import PromptTemplate

import re
from math_verify import parse, verify

logger = logging.getLogger(__name__)

class utils:
    def extract_question(question):
        new_question = question.replace(
            "以下是一道关于数学的单项选择题，请你一步一步推理，并在最后用“所以答案为选项X”给出答案，其中“X”为选项A，B，C，D中你认为正确的选项。下面是你要回答的问题\n", ''
        ).replace(
            "Here is a multiple-choice question about mathematics. Please reason through it step by step, and at the end, "
            "provide your answer option with 'Therefore, the correct answer is option X', Where 'X' is the correct option "
            "you think from A，B，C，D. Here is the question you need to answer:\n", ''
        ).replace(
            "请严格按照如下格式回答：[[选项]]，例如：选项: [[A]]。\n让我们一步一步思考：\n", ''
        ).replace(
            "Please Give the response by strictly following this format: [[choice]],"
            "for example: Choice: [[A]].\nLet's think step by step:\n", ''
        ).replace(
            "Please answer the following question directly and give the answer directly without any reasoning process. "
            "Please strictLy follow the format: [[choice]],for example: Choice: [[A]].\n", ''
        )
        return new_question

    def add_budget(question, budget):
        new_question = question \
            .replace("Let's think step by step:\n", f"Let's think step by step and use less than {budget} tokens:\n") \
            .replace("让我们一步一步思考：\n", f"让我们一步一步思考并使用少于 {budget} tokens:\n")
        return new_question

    def token_measure(text):
        tokenizer = tiktoken.encoding_for_model("gpt-4")
        return len(tokenizer.encode(text))

    def extract_number(text):
        pattern = r"\[\[(\d+)\]\]"
        match = re.search(pattern, text)
        if match:
            result = match.group(1)
            return result
        else:
            return -1

class AccEvaluator:
    """
    A class for evaluating the accuracy of predictions against ground truth in various formats.
    """

    def __init__(self, dataset=None):
        """
        Initialize the AccEvaluator with an optional dataset.
        
        Args:
            dataset: Optional dataset to evaluate. If None, must be set later.
        """
        self.dataset = dataset

    def accuracy(self):
        """
        Calculate the overall accuracy across the entire dataset.
        
        Returns:
            float: The accuracy score as a ratio of correct predictions to total samples
        """
        acc_num = 0
        for sample in self.dataset:
            acc_num += self.evaluate_sample(sample)
        return acc_num / len(self.dataset)

    @staticmethod
    def find_answer(text):
        """
        Extract multiple choice answer (A, B, C, or D) from text response.
        
        Args:
            text: The text response to analyze
            
        Returns:
            str: The extracted answer choice (A, B, C, D) or 'None' if not found

        """
        text = text.strip()
        last_newline_index = text.rfind('\n')
        prediction = text[last_newline_index + 1:]
        if len(prediction) < 5:
            search_texts = [
                'the correct answer is',
                '答案为选项'
            ]
            for search_text in search_texts:
                index = text.find(search_text)
                if index != -1:
                    prediction = text[index:]
                    break

        pattern = re.compile(r'[ABCD]')
        matches = pattern.findall(prediction)
        if matches:

            answer = ''.join(matches)[-1]
        else:
            answer = 'None'

        return answer

    @staticmethod
    def extract_predicted_answer(text):
        """
        Extract numerical or text answer from a response.
        
        Args:
            text: The text response to analyze
            
        Returns:
            str or None: The extracted answer or None if no valid answer found

        """
        pattern = r"\[\[(.*?)\]\]"

        match = re.findall(pattern, text)

        if match:
            return match[-1]

        regex_pattern = "(-?[$0-9.,]{2,})|(-?[0-9]+)"
        regexes_to_ignore = [
            ",",
            "\\$",
            "(?s).*#### ",
            "\\.$"
        ]
        match = re.findall(regex_pattern, text)
        if match:
            match = match[-1]
            if isinstance(match, tuple):
                match = [m for m in match if m][0]
            text = match.strip()
            for regex in regexes_to_ignore:
                text = re.sub(regex, "", text)
            return text
        else:
            return None

    # @staticmethod
    def evaluate_sample(self, sample, cloze=True):
        """
        Evaluate a single sample against its ground truth.
        
        Args:
            sample: Dictionary containing 'ground truth' and 'prediction' keys
            cloze: Boolean indicating if this is a cloze-style question (True) or 
                  multiple choice (False)
            
        Returns:
            bool: True if the prediction matches ground truth, False otherwise
            
        """
        gt = sample['ground truth']
        pred = sample['prediction']
        if cloze:
            return (gt == self.extract_predicted_answer(pred)) or (f"[[{gt}]]" in pred) \
                   or verify(parse(gt), parse(self.extract_predicted_answer(pred)))
        else:
            if f'[[{gt}]]' in pred:
                return True
            choice = self.find_answer(sample['prediction'])
            return choice == gt

class TaleStrategy:
    def __init__(self, threshold=0.5, output_path='./temp/default.jsonl', data_name='GSM8K'):
        self.threshold = threshold
        self.output_path = output_path
        self.data_name = data_name
        # self.evaluator = AccEvaluator()
        self.context = self._create_zero_shot_context()
        self.prompt_template = PromptTemplate(
            input_variables=["context", "question"],
            template="{context}\n\nBelow is the question:\n\nQuestion: \"{question}\"\n"
        )
        self.results = []
        self.acc_num = 0.0

    def _create_zero_shot_context(self):
        return """Task: Analyze the given question and estimate the minimum number of tokens required to generate a complete and accurate response. Please Give the response by strictly following this format: [[budget]],for example: Budget: [[12]]."""
    

    def apply(self, input_data, model):
        if isinstance(input_data, dict):
            raw_prompt = input_data['round'][0]['prompt']+"Let's think step by step:\n"
            gt = input_data['gold']
        else:
            raw_prompt = input_data+"Let's think step by step:\n"
            gt = None

        question = utils.extract_question(raw_prompt)
        question = raw_prompt
        format_prompt = self.prompt_template.format(
            context=self.context,
            question=question
        )

        answer = model.infer(format_prompt)
        budget_pred = int(utils.extract_number(answer))
        pruned_data = utils.add_budget(raw_prompt, budget_pred)

        return pruned_data
