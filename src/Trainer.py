'''
Created on 12-Sep-2020

@author: ragarwal
'''

import pandas as pd
from sklearn.preprocessing import LabelBinarizer

from BertLongFrmrConfig import TRAIN_BATCH_SIZE, VALID_BATCH_SIZE,\
    EPOCHS, LEARNING_RATE, TRAINING_DATA, MAX_LEN_LONG, tokenizer_long,\
    MAX_LEN_BERT, tokenizer_bert
from torch.utils.data.dataloader import DataLoader
from torch import cuda
from preprocessing import processFileNamesWithFinanceDictAndClean
import torch
from torch.utils.data.sampler import SequentialSampler
import gc
from torch.hub import tqdm
from BertLongModel import BertLongFrmrEnsembleClassifier
from DataSet import CustomDatasetLong, CustomDatasetBERT


device = 'cuda' if cuda.is_available() else 'cpu'

if(True):
    data =pd.read_pickle(TRAINING_DATA)
    data['fileName'] = data['path'].apply(lambda x: x.split("/")[-1])
    data['fileName'] = data['fileName'].apply(lambda x: processFileNamesWithFinanceDictAndClean(x))
    
    data['label'] = data['label'].replace("operation", 'operations')

    data = data.sample(frac=1).reset_index(drop=True)

    lb = LabelBinarizer()
    
    data['labelvec'] = lb.fit_transform(data['label']).tolist()
    
    import joblib
    joblib.dump(lb, "labelEncoder.sav")
    
    contentData = data[['clean_content_v1', 'labelvec']]
    fileData = data[['fileName', 'labelvec']]

    contentData = contentData.rename(columns = {"clean_content_v1":"text"})
    fileData = fileData.rename(columns = {"fileName":"text"})
    
    contentData.to_pickle("fullContentData.pkl")
    fileData.to_pickle("Long_fileData.pkl")
else:
    contentData = pd.read_pickle("fullContentData.pkl")
    fileData = pd.read_pickle("Long_fileData.pkl")
    

print("hi")

train_size = 0.8




content_training_set = CustomDatasetLong(contentData, tokenizer_long, MAX_LEN_LONG)
file_training_set = CustomDatasetBERT(fileData, tokenizer_bert, MAX_LEN_BERT)

train_params = {'batch_size': TRAIN_BATCH_SIZE,
                'shuffle': True,
                'num_workers': 0
                }

content_training_loader = DataLoader(content_training_set,  
                            batch_size=TRAIN_BATCH_SIZE, 
                            sampler=SequentialSampler(content_training_set))
file_training_loader = DataLoader(file_training_set,
                            batch_size=TRAIN_BATCH_SIZE, 
                            sampler=SequentialSampler(file_training_set))

model = BertLongFrmrEnsembleClassifier(len(contentData.iloc[0]['labelvec']))
#PATH = "EnsembleModel_Bert_Long_v1.pt"
#model.load_state_dict(torch.load(PATH, map_location=device))
model.to(device)

#Further training

#PATH = "EnsembleModel_Bert_Long.pt"
#model.load_state_dict(torch.load(PATH, map_location=device))

def loss_fn(outputs, targets):
    return torch.nn.BCEWithLogitsLoss()(outputs, targets)


no_decay = ["bias", "LayerNorm.weight"]
optimizer_grouped_parameters = [{
  "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
  }]

optimizer = torch.optim.AdamW(params =  optimizer_grouped_parameters, lr=LEARNING_RATE)

def train(epoch):
    
    for step, combined_batch in tqdm(enumerate(zip(content_training_loader, file_training_loader))):
        
        contentBatch, fileBatch = combined_batch
        model.train()
        
#         content_ids = contentBatch['ids'].to(device, dtype = torch.long)
#         content_mask = contentBatch['mask'].to(device, dtype = torch.long)
#         content_token_type_ids = contentBatch['token_type_ids'].to(device, dtype = torch.long)
#          
#         file_ids = fileBatch['ids'].to(device, dtype = torch.long)
#         file_mask = fileBatch['mask'].to(device, dtype = torch.long)
#         file_oken_type_ids = fileBatch['token_type_ids'].to(device, dtype = torch.long)
        
        
        batch_1 = tuple(t.to(device) for t in contentBatch)
        batch_2 = tuple(t.to(device) for t in fileBatch)
        
        targets = batch_2[2]
#         if(batch_1[0].shape[0] ==1):
#             inputs = {
#             "input_ids": [batch_1[0][0], batch_2[0][0]],
#             "attention_mask": [batch_1[1][0], batch_2[1][0]]
#             }
#         else:
#             inputs = {
#                 "input_ids": [torch.squeeze(batch_1[0]), torch.squeeze(batch_2[0])],
#                 "attention_mask": [torch.squeeze(batch_1[1]), torch.squeeze(batch_2[1])]
#             }

        inputs = {
        "input_ids": [batch_1[0], batch_2[0]],
        "attention_mask": [batch_1[1], batch_2[1]],
        "global_attention_mask": batch_1[2]
        }

        outputs = model(**inputs)
        
        optimizer.zero_grad()

        loss = loss_fn(outputs, targets)
        print('Epoch: {}, Loss:  {}'.format(epoch,loss.item() ))
        if step%5==0:
            print('Epoch: {}, Loss:  {}'.format(epoch,loss.item() ))
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        del outputs, inputs, batch_1, batch_2
        gc.collect()
        
        
for epoch in range(EPOCHS):
    train(epoch)
    
PATH = "EnsembleModel_Bert_Long_v3.pt"  
torch.save(model.state_dict(), PATH)

