import torch
from torch.utils.data import Dataset
import os
import csv


class CsvDataset(Dataset):
    def __init__(self, csv_path, tokenizer, max_len):
        self.tokenizer = tokenizer
        self.max_len = max_len

        data = None
        with open(csv_path) as csvfile:
            reader = csv.reader(csvfile)
            next(reader) # skip header row
            data = list(reader)

        self.encodings = [self.tokenizer.encode_plus(
          row[1],
          add_special_tokens=True,
          max_length=self.max_len,
        #   return_token_type_ids=True,
          truncation=True,
          padding='max_length',
        #   return_attention_mask=True,
          return_tensors='pt',
        ) for row in data]  
         
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        # ['id', 'comment_text', 'toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']
        row = self.data[idx]
        target = [int(l) for l in row[2:]]
        encoding = self.encodings[idx]
        # 'input_ids': (encoding['input_ids']).flatten()
        # 'attention_mask': (encoding['attention_mask']).flatten()
        # 'token_type_ids': (encoding['token_type_ids']).flatten()
        return {'input_ids': (encoding['input_ids']).flatten()}, torch.tensor(target, dtype=torch.float)
