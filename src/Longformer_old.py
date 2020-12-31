'''
Created on 03-Oct-2020

@author: ragarwal
'''
import gc
import os

from sklearn.preprocessing import LabelBinarizer
from transformers import LongformerTokenizer, LongformerConfig
from transformers.modeling_longformer import LongformerModel

import joblib
import torch
from torch import cuda

from preprocessing import transformSingle
import pandas as pd
from torch.utils.data import Dataset
from torch.utils.data.dataloader import DataLoader
from torch.utils.data.sampler import SequentialSampler
from torch.hub import tqdm


class LongformerContentModelConfig:
    defaultConfig = LongformerConfig()
    model_name = 'allenai/longformer-base-4096'
    labelEncoderFileName = 'labelEncoder_longfrmr_content.sav'
    savedModelFileName = 'Longformer_Content_Model.pt'
    tokenizer = LongformerTokenizer.from_pretrained(model_name)
    MAX_LEN = 1024 #max= 4096
    TRAIN_BATCH_SIZE = 3
    VALID_BATCH_SIZE = 1
    EPOCHS = 3
    LEARNING_RATE = 1e-05


class LongFormerContentModel(torch.nn.Module):
    '''
    classdocs
    '''

    def __init__(self, num_classes, configuration=None):
        super(LongFormerContentModel, self).__init__()
        if configuration:
            self.configuration = configuration
        else:
            self.configuration = LongformerContentModelConfig()
        self.longfomer_model = LongformerModel.from_pretrained(self.configuration.model_name, return_dict=True,
                                                               output_hidden_states=True)
        self.dense = torch.nn.Linear(768, 768)
        self.dropout = torch.nn.Dropout(0.3)
        self.classifier = torch.nn.Linear(768, num_classes)

    def forward(
            self,
            input_ids=None,
            attention_mask=None,
            global_attention_mask=None,
            token_type_ids=None,
            position_ids=None,
            head_mask=None,
            inputs_embeds=None,
            class_label=None,
    ):

        long_output = self.longfomer_model(input_ids, attention_mask=attention_mask,
                                           global_attention_mask=global_attention_mask)
        last_hidden_states = long_output[1]

        dense_op = self.dense(last_hidden_states)
        droppedOutOP = self.dropout(dense_op)
        logits = self.classifier(droppedOutOP)

        # crossentropyloss: https://pytorch.org/docs/stable/nn.html#crossentropyloss
        if class_label is not None:
            loss_fct = torch.nn.CrossEntropyLoss(ignore_index=-1)
            next_sentence_loss = loss_fct(logits.view(-1, 2), class_label.view(-1))
            return next_sentence_loss, logits
        else:
            return logits


class LongFormerContentClassifier(object):
    '''
    classdocs
    '''

    def __init__(self, base_dir, configuration=None, mode="eval"):
        '''
        Constructor
        '''
        if configuration:
            self.configuration = configuration
        else:
            self.configuration = LongformerContentModelConfig()
        self.BASE_DIR = base_dir
        self.labelEncoderPath = os.path.join(self.BASE_DIR, self.configuration.labelEncoderFileName)
        self.modelPath = os.path.join(self.BASE_DIR, self.configuration.savedModelFileName)
        self.device = 'cuda' if cuda.is_available() else 'cpu'

        self.tokenizer = self.configuration.tokenizer
        if mode == "eval":
            self.lb = joblib.load(self.labelEncoderPath)
            self.model = LongFormerContentModel(len(self.lb.classes_), self.configuration)
            PATH = self.modelPath
            self.model.load_state_dict(torch.load(PATH, map_location=self.device))
            self.model.eval()
        elif mode == "train":
            pass
        else:
            raise Exception("illegal mode")

    def predict(self, document, processed=False):

        self.model.eval()
        if not processed:
            document = transformSingle(document)

        document = " ".join(document.split(" ")[0:self.configuration.MAX_LEN])

        with torch.no_grad():
            inputs = self.tokenizer.encode_plus(
                document,
                None,
                add_special_tokens=True,
                max_length=self.configuration.MAX_LEN,
                pad_to_max_length=True,
                return_attention_mask=True,
                # return_tensors='pt',
                truncation=True
            )
            ids = inputs['input_ids']
            mask = inputs['attention_mask']

            global_mask = torch.zeros(torch.tensor(ids, dtype=torch.long).shape)
            global_mask[[self.tokenizer.convert_ids_to_tokens(ids).index('<s>')]] = 1

            ids = [ids]
            mask = [mask]


            contentIds = torch.tensor(ids, dtype=torch.long)
            contentMask = torch.tensor(mask, dtype=torch.long)

            inputs = {
                "input_ids": contentIds,
                "attention_mask": contentMask,
                "global_attention_mask": global_mask
            }

            outputs = self.model(**inputs)
            prediction = torch.sigmoid(outputs)

        return self.lb.inverse_transform(prediction.detach().numpy())

    def loss_fn(self, outputs, targets):
        return torch.nn.BCEWithLogitsLoss()(outputs, targets)

    def run_training(self, epoch, model, training_data_loader, optimizer):

        for step, contentBatch in tqdm(enumerate(training_data_loader)):

            model.train()
            batch_1 = tuple(t.to(self.device) for t in contentBatch)

            targets = batch_1[3]

            inputs = {
                "input_ids": batch_1[0],
                "attention_mask": batch_1[1],
                "global_attention_mask": batch_1[2]
            }

            outputs = model(**inputs)

            optimizer.zero_grad()

            loss = self.loss_fn(outputs, targets)
            print('Epoch: {}, Loss:  {}'.format(epoch, loss.item()))
            if step % 5 == 0:
                print('Epoch: {}, Loss:  {}'.format(epoch, loss.item()))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            del outputs, inputs, batch_1
            gc.collect()

    def train(self, training_data, previousSavedModel= None):

        lb = LabelBinarizer()
        training_data['labelvec'] = lb.fit_transform(training_data['label']).tolist()
        training_data = training_data[['text', 'labelvec']]
        joblib.dump(lb, self.labelEncoderPath)

        content_training_set = CustomDataset(training_data, self.tokenizer, self.configuration.MAX_LEN)
        train_params = {'batch_size': self.configuration.TRAIN_BATCH_SIZE,
                        'shuffle': True,
                        'num_workers': 0
                        }

        content_training_loader = DataLoader(content_training_set,
                                             batch_size=self.configuration.TRAIN_BATCH_SIZE,
                                             sampler=SequentialSampler(content_training_set))

        model = LongFormerContentModel(len(contentData.iloc[0]['labelvec']))
        if previousSavedModel:
            PATH = previousSavedModel
            model.load_state_dict(torch.load(PATH, map_location=self.device))
        model.to(self.device)

        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [{
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
        }]

        optimizer = torch.optim.AdamW(params=optimizer_grouped_parameters, lr=self.configuration.LEARNING_RATE)

        for epoch in range(self.configuration.EPOCHS):
            self.run_training(epoch, model, content_training_loader, optimizer)

        torch.save(model.state_dict(), self.modelPath)


class CustomDataset(Dataset):

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
            truncation=True
        )
        ids = inputs['input_ids']
        mask = inputs['attention_mask']
        global_mask = torch.zeros(torch.tensor(ids, dtype=torch.long).shape)
        global_mask[[self.tokenizer.convert_ids_to_tokens(ids).index('<s>')]] = 1
        # token_type_ids = inputs["token_type_ids"]

        # return torch.tensor(ids, dtype=torch.long), torch.tensor(mask, dtype=torch.long), torch.tensor(self.targets[index], dtype=torch.float)
        return torch.tensor(ids, dtype=torch.long), torch.tensor(mask, dtype=torch.long), global_mask, torch.tensor(self.targets[index], dtype=torch.float)


if __name__ == '__main__':
    createData = False
    BASE_DIR = "longformer_content"
    if not os.path.exists(BASE_DIR):
        os.mkdir(BASE_DIR)
    if createData:
        #dataPath = "/home/ec2-user/rajat/rubic_training_data_JUL15_2020.pickle"
        dataPath = "/Users/ragarwal/Work/Intralinks/ContentOrganization/Data/rubic_training_data_JUL15_2020.pickle"
        data = pd.read_pickle(dataPath)
        data['label'] = data['label'].replace("operation", 'operations')
        data = data.sample(frac=1).reset_index(drop=True)
        contentData = data[['clean_content_v1', 'label']]
        contentData = contentData.rename(columns={"clean_content_v1": "text"})
        contentData.to_pickle(os.path.join(BASE_DIR, "fullContentData.pkl"))
    else:
        contentData = pd.read_pickle(os.path.join(BASE_DIR, "fullContentData.pkl"))

    longModel = LongFormerContentClassifier(BASE_DIR, mode="train")
    longModel.train(contentData)
