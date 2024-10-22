from transformers import BertModel, BertConfig, BertPreTrainedModel
import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss, MSELoss

from model.OutRobertaClassificationHead import OutRobertaClassificationHead  # Giữ nguyên tên lớp cũ

class OutRobertaForSequenceClassification(BertPreTrainedModel):  # Giữ nguyên tên lớp cũ
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.bert = BertModel(config)
        self.classifier = nn.Linear(config.hidden_size + external_feature_size, self.num_labels)  # Thêm external feature

    def forward(self, input_ids=None, attention_mask=None, token_type_ids=None,
                position_ids=None, head_mask=None, inputs_embeds=None,
                labels=None, externalFeature=None):  # Nhận thêm externalFeature

        outputs = self.bert(input_ids, attention_mask=attention_mask,
                            token_type_ids=token_type_ids,
                            position_ids=position_ids, head_mask=head_mask)

        sequence_output = outputs[0]
        combined_output = torch.cat((sequence_output, externalFeature), dim=-1)  # Kết hợp với externalFeature
        logits = self.classifier(combined_output)

        outputs = (logits,) + outputs[2:]

        if labels is not None:
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            outputs = (loss,) + outputs

        return outputs
