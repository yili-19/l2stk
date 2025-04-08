# models/custom_model.py

class CustomModel:
    def __init__(self, infer_method):
        self.infer_method = infer_method

    def infer(self, input_data):
        # 使用用户自定义的推理方法
        return self.infer_method(input_data)