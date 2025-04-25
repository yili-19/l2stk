from transformers import AutoModel, AutoTokenizer
import torch
import torch.nn as nn

class DefaultModel(nn.Module):
    def __init__(self, config, strategy):
        super(DefaultModel, self).__init__()
        self.model_type = config.get("model_type", "transformer")  # 模型类型，默认为transformer
        self.model_name_or_path = config.get("model_name_or_path", "bert-base-uncased")  # 用户提供的预训练模型路径
        self.strategy = strategy  # 获取策略

        self.modified_structure = self.strategy.get('modify_structure', False)  # 根据策略判断是否修改模型结构
        self.load_model()

    def load_model(self):
        # 使用AutoModel加载通用的大模型
        if self.modified_structure:
            # 如果需要修改模型结构
            self.model = self.create_modified_model()
        else:
            # 否则直接加载预训练模型
            self.model = AutoModel.from_pretrained(self.model_name_or_path)

    def create_modified_model(self):
        # 加载基础模型
        model = AutoModel.from_pretrained(self.model_name_or_path)
        
        # 假设你需要修改模型的某些部分，比如加上一个分类头或修改现有层
        # 这里是一个简单的修改：我们将添加一个线性层
        model.classifier = nn.Sequential(
            nn.Linear(model.config.hidden_size, model.config.hidden_size),
            nn.ReLU(),
            nn.Linear(model.config.hidden_size, 2)  # 假设是二分类任务
        )
        return model

    def forward(self, input_ids, attention_mask=None):
        return self.model(input_ids=input_ids, attention_mask=attention_mask)