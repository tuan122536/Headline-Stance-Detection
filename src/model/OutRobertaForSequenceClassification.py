from transformers import BertModel, BertConfig, BertPreTrainedModel
import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss, MSELoss

from model.OutRobertaClassificationHead import OutRobertaClassificationHead  # Giữ nguyên tên lớp cũ

class OutRobertaForSequenceClassification(BertPreTrainedModel):  # Giữ nguyên tên lớp cũ
    config_class = BertConfig
    base_model_prefix = "bert"

    def __init__(self, config, weight=None):
        super(OutRobertaForSequenceClassification, self).__init__(config)
        
        self.num_labels = config.num_labels  # Số nhãn của phân loại
        self.bert = BertModel(config)  # Khởi tạo mô hình BERT
        self.classifier = OutRobertaClassificationHead(config)  # Giữ nguyên lớp phân loại cũ
        self.weight = weight  # Trọng số cho hàm mất mát (nếu có)

    def forward(self, input_ids=None, attention_mask=None, token_type_ids=None,
                position_ids=None, head_mask=None, inputs_embeds=None,
                labels=None, externalFeature=None):
        outputs = self.bert(input_ids, attention_mask=attention_mask,
                            token_type_ids=token_type_ids,
                            position_ids=position_ids, head_mask=head_mask)

        sequence_output = outputs[0]  # Lấy đầu ra cuối cùng từ BERT
        logits = self.classifier(sequence_output, externalFeature=externalFeature)  # Lấy logits từ lớp phân loại

        outputs = (logits,) + outputs[2:]  # Kết hợp logits với các đầu ra khác

        if labels is not None:
            if self.num_labels == 1:
                loss_fct = MSELoss()  # Sử dụng MSE nếu chỉ có 1 nhãn
                loss = loss_fct(logits.view(-1), labels.view(-1))
            else:
                loss_fct = CrossEntropyLoss(weight=self.weight)  # Sử dụng CrossEntropy với trọng số (nếu có)
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            outputs = (loss,) + outputs  # Thêm loss vào đầu ra

        return outputs  # Trả về (loss), logits, (hidden_states), (attentions)
