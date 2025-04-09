class CosStrategy:
    def __init__(self, threshold=0.5):
        self.threshold = threshold

    def apply(self, input_data, model):
        prompt = open('toolkit/datasets/cos_demo/cos_demo_shuffle_both_1.txt').read()
        pruned_data = 'Answer the question using the following format in these given examples:'+ '\n\n' +prompt+ '\n\n' + 'Question:' + '\n' + input_data + '\nAnswer:\n'
        return pruned_data