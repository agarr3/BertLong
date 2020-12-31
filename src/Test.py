'''
Created on 23-Sep-2020

@author: ragarwal
'''

# import torch
# from transformers import LongformerForMaskedLM, LongformerTokenizer
# 
# model = LongformerForMaskedLM.from_pretrained('allenai/longformer-base-4096', return_dict=True)
# tokenizer = LongformerTokenizer.from_pretrained('allenai/longformer-base-4096')
# 
# SAMPLE_TEXT = "Hello World"  # long input document
# encoding = tokenizer(SAMPLE_TEXT, return_tensors="pt")  # batch of size 1
# 
# print(encoding["input_ids"])
# 
# all_tokens = tokenizer.convert_ids_to_tokens(encoding["input_ids"][0].tolist())
# 
# print(all_tokens)
# 
# print(encoding["attention_mask"])

# import pandas as pd
#
# data = pd.read_pickle("/Users/ragarwal/Work/Intralinks/ContentOrganization/Data/rubic_training_data_JUL15_2020.pickle")
#
# print(data.label.value_counts())
import pandas

from Longformer import LongFormerContentClassifier

BASE_DIR = "longformer_content"
longModel = LongFormerContentClassifier(BASE_DIR, mode="train")
longModel.avg_valid_losses = [0.7, 0.8]
longModel.avg_train_losses = [0.3, 0.4]
longModel.visualizeTraining()