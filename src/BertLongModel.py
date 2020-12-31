'''
Created on 18-Sep-2020
 
@author: ragarwal
'''
import transformers
import torch
from BertLongFrmrConfig import model_bert_name, model_long_name
from transformers.modeling_longformer import LongformerModel
 
class BertLongFrmrEnsembleClassifier(torch.nn.Module):
    '''
    classdocs
    '''
 
 
    def __init__(self, num_classes):
        super(BertLongFrmrEnsembleClassifier, self).__init__()
     
        self.bert_mode = transformers.BertModel.from_pretrained(model_bert_name)
        self.nameDropout =  torch.nn.Dropout(0.3)
        self.longfomer_model = LongformerModel.from_pretrained(model_long_name, return_dict=True, output_hidden_states=True)
        self.dense = torch.nn.Linear(2 * 768, 2 * 768)
        self.droout =  torch.nn.Dropout(0.1)
        self.cls = torch.nn.Linear(2 * 768, num_classes)
 
    def forward(
          self,
          input_ids=None,
          attention_mask=None,
          global_attention_mask= None,
          token_type_ids=None,
          position_ids=None,
          head_mask=None,
          inputs_embeds=None,
          class_label=None,
    ):
 
        #outputs = []
        input_ids_1 = input_ids[1]
        attention_mask_1 = attention_mask[1]
        bert_output = self.bert_mode(input_ids_1, attention_mask=attention_mask_1)[1]
        bert_dropped_output = self.nameDropout(bert_output)
     
        input_ids_2 = input_ids[0]
        attention_mask_2 = attention_mask[0]
        long_output = self.longfomer_model(input_ids_2,attention_mask=attention_mask_2, global_attention_mask= global_attention_mask)
        long_output_hidden_state = long_output[1]
     
        # just get the [CLS] embeddings
        last_hidden_states = torch.cat([bert_dropped_output, long_output_hidden_state], dim=1)
         
         
#         last_hidden_states = torch.cat([self.roberta_model_1(input_ids[0], attention_mask=attention_mask[0])[1], 
#                                         self.roberta_model_2(input_ids[1],attention_mask=attention_mask[1])[1]], dim=1)
         
        dense_op = self.dense(last_hidden_states)
        droppedOutOP = self.droout(dense_op)
        logits = self.cls(droppedOutOP)
     
        # crossentropyloss: https://pytorch.org/docs/stable/nn.html#crossentropyloss
        if class_label is not None:
            loss_fct = torch.nn.CrossEntropyLoss(ignore_index=-1)
            next_sentence_loss = loss_fct(logits.view(-1, 2), class_label.view(-1))
            return next_sentence_loss, logits
        else:
            return logits
