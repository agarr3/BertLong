'''
Created on 18-Sep-2020

@author: ragarwal
'''
from transformers.tokenization_longformer import LongformerTokenizer

'''
Created on 12-Sep-2020

@author: ragarwal
'''
from transformers.tokenization_bert import BertTokenizer
# from transformers import LongformerTokenizer, LongformerForSequenceClassification
 
#model_name = 'roberta-large'
#tokenizer = RobertaTokenizer.from_pretrained(model_name)
 
model_bert_name = 'bert-base-uncased'
tokenizer_bert = BertTokenizer.from_pretrained(model_bert_name)
 
model_long_name = 'allenai/longformer-base-4096'
tokenizer_long = LongformerTokenizer.from_pretrained(model_long_name)
 
MAX_LEN_BERT = 512
MAX_LEN_LONG = 1024
TRAIN_BATCH_SIZE = 2
VALID_BATCH_SIZE = 1
EPOCHS = 3
LEARNING_RATE = 1e-05
 
 
 
 
#TRAINING_DATA = "/Users/ragarwal/Work/Intralinks/ContentOrganization/Data/rubic_training_data_JUL15_2020.pickle"
 
TRAINING_DATA = "/home/ec2-user/rajat/rubic_training_data_JUL15_2020.pickle"
