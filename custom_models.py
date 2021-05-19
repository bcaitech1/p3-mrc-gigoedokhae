import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss
from transformers.modeling_outputs import QuestionAnsweringModelOutput

from transformers import XLMRobertaConfig, XLMRobertaTokenizerFast, XLMRobertaForQuestionAnswering


class CustomXLMRoberta(XLMRobertaForQuestionAnswering):
    """
    1. transfer learning 1: qa_outputs 레이어와 roberta encoder 첫번째 레이어 작은 lr 학습 (1, 2 uncomment)
    2. transfer learning 2: qa_outputs 레이어만 학습 (1 uncomment, 2 comment)
    3. fine-tuning: 전체 작은 lr 학습 (1, 2 comment)
    """
    def __init__(self, config):
        super().__init__(config)

        # 1. freeze parameters of XLMRoberta (if transfer learning 1, 2)
        # for param in self.roberta.parameters():
        #     param.requires_grad = False

        # 2. unfreeze parameters of the first layer of encoder (if transfer learning 1)
        # for name, param in self.roberta.encoder.layer[0].named_parameters():
        #     if "LayerNorm" not in name:
        #         param.requires_grad = True

        # 3. qa layer is not frozen; ["qa_outputs.weight", "qa_outputs.bias"]
        # self.qa_outputs = nn.Linear(config.hidden_size, config.num_labels)

if __name__ == "__main__":
    config = XLMRobertaConfig.from_pretrained("xlm-roberta-large")
    tokenizer = XLMRobertaTokenizerFast.from_pretrained("xlm-roberta-large")
    config.is_decoder = True
    model = CustomXLMRoberta.from_pretrained("xlm-roberta-large", config=config)

    question, text = "Who was Jim Henson?", "Jim Henson was a nice puppet"
    inputs = tokenizer(question, text, return_tensors='pt')
    start_positions = torch.tensor([1])
    end_positions = torch.tensor([3])

    outputs = model(**inputs, start_positions=start_positions, end_positions=end_positions)
    loss = outputs.loss
    start_scores = outputs.start_logits
    end_scores = outputs.end_logits
