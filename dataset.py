import torch
from torch.utils.data import Dataset


class CustomDataset(Dataset):
    def __init__(self, data, targets, tokenizer, max_len=64):
        self.data = data
        self.targets = targets
        self.tokenizer = tokenizer
        self.max_len = max_len

    def len(self):
        return len(self.data)

    def __getitem__(self, idx):
        text = self.data[idx]
        target = self.targets[idx]
        inputs = self.tokenizer.encode_plus(
            text,
            None,
            add_special_tokens=True,
            max_length=self.max_len,
            padding="max_length",
            truncation=True,
        )
        resp = {
            "ids": torch.Tensor(inputs["input_ids"], dtype=torch.long),
            "mask": torch.Tensor(inputs["attention_mask"], dtype=torch.long),
            "token_type_ids": torch.Tensor(inputs["token_type_ids"], dtype=torch.long),
            "target": torch.Tensor(target, dtype=torch.float),
        }
        return resp
