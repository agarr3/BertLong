'''
Created on 12-Sep-2020

@author: ragarwal
'''
import torch
from torch.utils.data import Dataset
from BertLongFrmrConfig import tokenizer_long

# class CustomDatasetLong(Dataset):
# 
#     def __init__(self, dataframe, tokenizer, max_len):
#         self.tokenizer = tokenizer
#         self.data = dataframe
#         self.text = dataframe.text
#         self.targets = self.data.labelvec
#         self.max_len = max_len
# 
#     def __len__(self):
#         return len(self.text)
# 
#     def __getitem__(self, index):
#         text = str(self.text[index])
#         text = " ".join(text.split()[0:self.max_len])
#         
#         input_ids = torch.tensor(self.tokenizer.encode(text)).unsqueeze(0)
#         #input_ids = torch.tensor(self.tokenizer.encode(text))
#         attention_mask = torch.ones(input_ids.shape, dtype=torch.long, device=input_ids.device)
#         
#         global_attention_mask = torch.zeros(input_ids.shape, dtype=torch.long, device=input_ids.device)
#         global_attention_mask[:, [1, 4, 21,]] = 1 
# 
#        
#        
#        
#       
#         #token_type_ids = inputs["token_type_ids"]
#         
#         #return torch.tensor(ids, dtype=torch.long), torch.tensor(mask, dtype=torch.long), torch.tensor(self.targets[index], dtype=torch.float)
#         return input_ids, attention_mask, global_attention_mask


class CustomDatasetLong(Dataset):

    def __init__(self, dataframe, tokenizer, max_len):
        self.tokenizer = tokenizer
        self.data = dataframe
        self.text = dataframe.text
        self.targets = self.data.labelvec
        self.max_len = max_len

    def __len__(self):
        return len(self.text)

    def __getitem__(self, index):
        text = str(self.text[index])
        text = " ".join(text.split()[0:self.max_len])

        inputs = self.tokenizer.encode_plus(
            text,
            None,
            add_special_tokens=True,
            max_length=self.max_len,
            pad_to_max_length=True,
            return_attention_mask=True,
            #return_tensors='pt',
            truncation=True
        )
        ids = inputs['input_ids']
        mask = inputs['attention_mask']
        global_mask = torch.zeros(torch.tensor(ids, dtype=torch.long).shape)
        global_mask[[self.tokenizer.convert_ids_to_tokens(ids).index('<s>')]] = 1
        #token_type_ids = inputs["token_type_ids"]
        
        #return torch.tensor(ids, dtype=torch.long), torch.tensor(mask, dtype=torch.long), torch.tensor(self.targets[index], dtype=torch.float)
        return torch.tensor(ids, dtype=torch.long), torch.tensor(mask, dtype=torch.long), global_mask


class CustomDatasetBERT(Dataset):

    def __init__(self, dataframe, tokenizer, max_len):
        self.tokenizer = tokenizer
        self.data = dataframe
        self.text = dataframe.text
        self.targets = self.data.labelvec
        self.max_len = max_len

    def __len__(self):
        return len(self.text)

    def __getitem__(self, index):
        text = str(self.text[index])
        text = " ".join(text.split()[0:self.max_len])

        inputs = self.tokenizer.encode_plus(
            text,
            None,
            add_special_tokens=True,
            max_length=self.max_len,
            pad_to_max_length=True,
            return_attention_mask=True,
            #return_tensors='pt',
            truncation=True
        )
        ids = inputs['input_ids']
        mask = inputs['attention_mask']
        #token_type_ids = inputs["token_type_ids"]
        
        #return torch.tensor(ids, dtype=torch.long), torch.tensor(mask, dtype=torch.long), torch.tensor(self.targets[index], dtype=torch.float)
        return torch.tensor(ids, dtype=torch.long), torch.tensor(mask, dtype=torch.long), torch.tensor(self.targets[index], dtype=torch.float)
