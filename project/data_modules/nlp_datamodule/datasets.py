import torch
from torch.utils.data import Dataset
import os
import csv


class CsvDataset(Dataset):
    def __init__(self, csv_path, tokenizer, max_len):
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.data = None

        with open(csv_path) as csvfile:
            self.data = list(csv.reader(csvfile))
         
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        # ['id', 'comment_text', 'toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']
        row = self.data[idx]
        text = str(row[1])
        target = row[2:]
        
        encoding = self.tokenizer.encode_plus(
          text,
          add_special_tokens=True,
          max_length=self.max_len,
        #   return_token_type_ids=True,
          truncation=True,
          padding='max_length',
        #   return_attention_mask=True,
          return_tensors='pt',
        )    
        # 'input_ids': (encoding['input_ids']).flatten()
        # 'attention_mask': (encoding['attention_mask']).flatten()
        # 'token_type_ids': (encoding['token_type_ids']).flatten()
        print('==============\n', encoding['input_ids'])
        return encoding, torch.tensor(target, dtype=torch.long)
