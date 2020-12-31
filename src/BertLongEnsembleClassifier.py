'''
Created on 14-Sep-2020

@author: ragarwal
'''
import joblib
import torch
from torch import cuda
from transformers.tokenization_bert import BertTokenizer
from preprocessing import transformSingle,\
    processFileNamesWithFinanceDictAndClean
from BertLongModel import BertLongFrmrEnsembleClassifier
from BertLongFrmrConfig import MAX_LEN_LONG, MAX_LEN_BERT, tokenizer_long,\
    tokenizer_bert


device = 'cuda' if cuda.is_available() else 'cpu'

class BertLongEnsembleClassifierModel(object):
    '''
    classdocs
    '''


    def __init__(self):
        '''
        Constructor
        '''
        self.lb = joblib.load( "labelEncoder.sav")
        self.model = BertLongFrmrEnsembleClassifier(len(self.lb.classes_))
        PATH = "EnsembleModel_Bert_Long_v3.pt"
        self.model.load_state_dict(torch.load(PATH, map_location=device))
        self.model.eval()
        
        #self.model.to(device)
#         model_name = 'bert-base-uncased'
#         self.tokenizer = BertTokenizer.from_pretrained(model_name)
#         self.max_len = 512
        
    def predict(self, document, documentName, processed=False):
        
        self.model.eval()
        if(processed==False):
            document = transformSingle(document)
            documentName = processFileNamesWithFinanceDictAndClean(documentName)
        
        document = " ".join(document.split(" ")[0:MAX_LEN_LONG])
        documentName = " ".join(documentName.split(" ")[0:MAX_LEN_BERT])
        
        
        with torch.no_grad():
            inputs = tokenizer_long.encode_plus(
                document,
                None,
                add_special_tokens=True,
                max_length=MAX_LEN_LONG,
                pad_to_max_length=True,
                return_attention_mask=True,
                #return_tensors='pt',
                truncation=True
            )
            ids = inputs['input_ids']
            mask = inputs['attention_mask']

            global_mask = torch.zeros(torch.tensor(ids, dtype=torch.long).shape)
            global_mask[[tokenizer_long.convert_ids_to_tokens(ids).index('<s>')]] = 1
            
            ids = [ids]
            mask = [mask]
            
            contentIds = torch.tensor(ids, dtype=torch.long)
            contentMask = torch.tensor(mask, dtype=torch.long)
            
            
            
            inputs = tokenizer_bert.encode_plus(
                documentName,
                None,
                add_special_tokens=True,
                max_length=MAX_LEN_BERT,
                pad_to_max_length=True,
                return_attention_mask=True,
                #return_tensors='pt',
                truncation=True
            )
            ids = inputs['input_ids']
            mask = inputs['attention_mask']
            
            
            ids = [ids]
            mask = [mask]
            
            fileNameIds = torch.tensor(ids, dtype=torch.long)
            fileNametMask = torch.tensor(mask, dtype=torch.long)
            
            inputs = {
            "input_ids": [contentIds, fileNameIds],
            "attention_mask": [contentMask, fileNametMask],
            "global_attention_mask": global_mask
            }
            
            outputs = self.model(**inputs)
            prediction = torch.sigmoid(outputs)
        
        return self.lb.inverse_transform(prediction.detach().numpy())
        
        