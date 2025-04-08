# strategies/pruning_strategy.py

class PruningStrategy:
    def __init__(self, threshold=0.5):
        self.threshold = threshold

    def apply(self, input_data, model):
        pruned_data = input_data 
        return pruned_data