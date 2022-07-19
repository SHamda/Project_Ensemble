import torch
import torch.nn as nn
import transformers
from transformers import AdamW, get_linear_schedule_with_warmup
import tez
from sklearn import metrics


class TextModel(tez.Model):
    def __init__(self, num_classes, num_train_steps):
        super().__init__()
        self.bert = transformers.BertModel.from_pretrained(
            "aubmindlab/bert-base-arabertv02", return_dict=False
        )
        self.bert_drop = nn.Dropout(0.3)
        self.out = nn.Linear(768, num_classes)
        self.num_train_steps = num_train_steps

    def fetch_optimizer(self):
        opt = AdamW(self.parameters(), lr=1e-4)
        return opt

    def fetech_scheduler(self):
        sch = get_linear_schedule_with_warmup(
            self.optimizer, num_warmup_steps=0, num_training_steps=self.num_train_steps
        )
        return sch

    def loss(self, outputs, targets):
        return nn.BCEWithLogitsLoss()(outputs, targets.view(-1, 1))

    def monitor_metrics(self, outputs, targets):
        outputs = torch.sigmoid(outputs).cpu().detach().numpy() >= 0.5
        targets = targets.cpu().detach().numpy()
        return {"accuracy": metrics.accuracy_score(targets, outputs)}

    def forward(self, ids, mask, token_type_ids, targets=None):
        _, x = self.bert(ids, attention_mask=mask, token_type_ids=token_type_ids)
        x = self.bert_drop(x)
        x = self.out(x)
        if targets is not None:
            loss = self.loss(x, targets)
            met = self.monitor_metrics(x, targets)
            return x, loss, met
        return x, 0, {}
