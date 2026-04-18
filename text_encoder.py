import torch
import torch.nn as nn
from transformers import BertModel


class TextEncoder(nn.Module):
    def __init__(self, pretrained_model="bert-base-uncased", freeze=True):
        super(TextEncoder, self).__init__()

        self.bert = BertModel.from_pretrained(pretrained_model)

        if freeze:
            for param in self.bert.parameters():
                param.requires_grad = False

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask
        )

        # CLS token representation
        cls_embedding = outputs.last_hidden_state[:, 0, :]  # (batch, 768)

        return cls_embedding