'''
Created on 03-Oct-2020

@author: ragarwal
'''
import gc
import math
import os

from sklearn.preprocessing import LabelBinarizer
from transformers import LongformerTokenizer, LongformerConfig, get_linear_schedule_with_warmup
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

from pytorchtools import EarlyStoppingAndCheckPointer, ModelCheckPointer
import numpy as np
import matplotlib.pyplot as plt



class LongformerContentModelConfig:
    defaultConfig = LongformerConfig()
    model_name = 'allenai/longformer-base-4096'
    labelEncoderFileName = 'labelEncoder_longfrmr_content.sav'
    savedModelFileName = 'Longformer_Content_Model_v1.pt'
    tokenizer = LongformerTokenizer.from_pretrained(model_name)
    MAX_LEN = 1024  # max= 4096
    TRAIN_BATCH_SIZE = 3
    ACCUMULATION_STEPS = 9
    VALID_BATCH_SIZE = 2
    EPOCHS = 5
    LEARNING_RATE = 1e-06
    WEIGHT_DECAY = 4e-3
    PATIENCE = 2
    WARM_UP_RATIO = 0.06
    max_grad_norm = 1.0


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

    def __init__(self, base_dir, modelOverRideForEval = None,  configuration=None, mode="eval"):
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
            if modelOverRideForEval:
                self.model.load_state_dict(torch.load(modelOverRideForEval, map_location=self.device))
                self.model.eval()
            else:
                modelCheckpointer = ModelCheckPointer()
                modelCheckpointer.loadBestModel(self.BASE_DIR, self.model, self.device)
                self.model.eval()
        elif mode == "train":
            # to track the training loss as the model trains
            self.train_losses = []
            # to track the validation loss as the model trains
            self.valid_losses = []
            # to track the average training loss per epoch as the model trains
            self.avg_train_losses = []
            # to track the average validation loss per epoch as the model trains
            self.avg_valid_losses = []
            self.accuracyBoolList = []
            self.accuracyList = []
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

    def run_evaluation(self, model, validation_data_loader):
        model.eval()
        for step, contentBatch in tqdm(enumerate(validation_data_loader), desc="running evaluation"):
            batch_1 = tuple(t.to(self.device) for t in contentBatch)
            targets = batch_1[3]
            inputs = {
                "input_ids": batch_1[0],
                "attention_mask": batch_1[1],
                "global_attention_mask": batch_1[2]
            }
            outputs = model(**inputs)
            loss = self.loss_fn(outputs, targets)
            self.valid_losses.append(loss.item())
            _, predicted = torch.max(outputs, 1)
            _, trueClass = torch.max(targets, 1)
            booleans = (predicted == trueClass).squeeze()
            self.accuracyBoolList.extend([boolean.item() for boolean in booleans])

    def run_training(self, epoch, model, training_data_loader, optimizer, scheduler):

        model.train()
        model.zero_grad()
        for step, contentBatch in tqdm(enumerate(training_data_loader), desc="running training for epoch {}".format(epoch)):
            batch_1 = tuple(t.to(self.device) for t in contentBatch)
            targets = batch_1[3]
            inputs = {
                "input_ids": batch_1[0],
                "attention_mask": batch_1[1],
                "global_attention_mask": batch_1[2]
            }
            outputs = model(**inputs)
            loss = self.loss_fn(outputs, targets)
            self.train_losses.append(loss.item())
            loss = loss / self.configuration.ACCUMULATION_STEPS
            loss.backward()

            if (step + 1) % self.configuration.ACCUMULATION_STEPS == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), self.configuration.max_grad_norm)
                optimizer.step()  # Now we can do an optimizer step
                scheduler.step()
                optimizer.zero_grad()
                model.zero_grad()
                print('Epoch: {}, step: {},  Loss:  {}'.format(epoch, step, loss.item()))


            del outputs, inputs, batch_1
            gc.collect()

    def train(self, training_data, validationData, trainDataSize, trainFromScratch=True):

        lb = LabelBinarizer()
        training_data['labelvec'] = lb.fit_transform(training_data['label']).tolist()
        training_data = training_data[['text', 'labelvec']]
        joblib.dump(lb, self.labelEncoderPath)

        validationData['labelvec'] = lb.transform(validationData['label']).tolist()
        validationData = validationData[['text', 'labelvec']]

        content_training_set = CustomDataset(training_data, self.tokenizer, self.configuration.MAX_LEN)
        content_validation_set = CustomDataset(validationData, self.tokenizer, self.configuration.MAX_LEN)

        content_training_loader = DataLoader(content_training_set,
                                             batch_size=self.configuration.TRAIN_BATCH_SIZE,
                                             sampler=SequentialSampler(content_training_set), drop_last=False)

        content_validation_loader = DataLoader(content_validation_set,
                                             batch_size=self.configuration.VALID_BATCH_SIZE,
                                             sampler=SequentialSampler(content_validation_set), drop_last=False)

        model = LongFormerContentModel(len(contentData.iloc[0]['labelvec']))

        # no_decay = ["bias", "LayerNorm.weight"]
        # optimizer_grouped_parameters = [{
        #     "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
        # }]

        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
                "weight_decay": self.configuration.WEIGHT_DECAY,
            },
            {
                "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
            },
        ]

        t_total = len(content_training_loader) // self.configuration.ACCUMULATION_STEPS * self.configuration.EPOCHS
        warmup_steps = math.ceil(t_total * self.configuration.WARM_UP_RATIO)
        optimizer = torch.optim.AdamW(params=optimizer_grouped_parameters, lr=self.configuration.LEARNING_RATE)
        scheduler = get_linear_schedule_with_warmup(
            optimizer, num_warmup_steps=warmup_steps, num_training_steps=t_total
        )

        if not trainFromScratch:
            # PATH = previousSavedModel
            modelCheckpointer = ModelCheckPointer()
            modelCheckpointer.load_checkpoint(self.BASE_DIR, model, self.device, optimizer, scheduler)
            # model.load_state_dict(torch.load(PATH, map_location=self.device))
        model.to(self.device)

        early_stopping = EarlyStoppingAndCheckPointer(patience=self.configuration.PATIENCE, verbose=True, basedir=self.BASE_DIR)

        for epoch in range(self.configuration.EPOCHS):
            self.run_training(epoch, model, content_training_loader, optimizer, scheduler)
            self.run_evaluation(model, content_validation_loader)

            train_loss = np.average(self.train_losses)
            valid_loss = np.average(self.valid_losses)
            accuracy = sum(self.accuracyBoolList)/len(self.accuracyBoolList) * 100
            self.train_losses = []
            self.valid_losses = []
            self.accuracyBoolList = []
            self.avg_train_losses.append(train_loss)
            self.avg_valid_losses.append(valid_loss)
            self.accuracyList.append(accuracy)
            print('Epoch: {},  Total Train Loss:  {}'.format(epoch, train_loss))
            print('Epoch: {},  Total Validation Loss:  {}'.format(epoch, valid_loss))
            print('Epoch: {},  Total Validation accuracy:  {}'.format(epoch, accuracy))
            early_stopping(valid_loss, model, optimizer, epoch, scheduler)
            self.visualizeTraining()
            self.visualizeValAccuracy()
            if early_stopping.early_stop:
                print("Early stopping")
                self.model = model
                break

        # torch.save(model.state_dict(), self.modelPath)
    def visualizeTraining(self):
        # visualize the loss as the network trained
        fig = plt.figure(figsize=(10, 8))
        plt.plot(range(1, len(self.avg_train_losses) + 1), self.avg_train_losses, label='Training Loss')
        plt.plot(range(1, len(self.avg_valid_losses) + 1), self.avg_valid_losses, label='Validation Loss')

        # find position of lowest validation loss
        minposs = self.avg_valid_losses.index(min(self.avg_valid_losses)) + 1
        plt.axvline(minposs, linestyle='--', color='r', label='Early Stopping Checkpoint')

        plt.xlabel('epochs')
        plt.ylabel('loss')
        plt.ylim(0, 0.5)  # consistent scale
        plt.xlim(0, len(self.avg_train_losses) + 1)  # consistent scale
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.show()
        fig.savefig(os.path.join(self.BASE_DIR , 'loss_plot.png'), bbox_inches='tight')

    def visualizeValAccuracy(self):
        # visualize the loss as the network trained
        fig = plt.figure(figsize=(10, 8))
        plt.plot(range(1, len(self.accuracyList) + 1), self.accuracyList, label='Validation Accuracy')

        # find position of lowest validation loss

        plt.xlabel('epochs')
        plt.ylabel('accuracy')
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.show()
        fig.savefig(os.path.join(self.BASE_DIR , 'accuracy_plot.png'), bbox_inches='tight')


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
        return torch.tensor(ids, dtype=torch.long), torch.tensor(mask, dtype=torch.long), global_mask, torch.tensor(
            self.targets[index], dtype=torch.float)


if __name__ == '__main__':
    runMode = "train"
    #val_data_path = "/Users/ragarwal/eclipse-workspace/PyTEnsembleClassifier/src/doc_category_validation_data.pkl"
    val_data_path = "/home/ec2-user/rajat/doc_category_validation_data.pkl"
    validationDataOriginal = pd.read_pickle(val_data_path)
    BASE_DIR = "longformer_content"
    if(runMode=="train"):
        createData = False
        if not os.path.exists(BASE_DIR):
            os.mkdir(BASE_DIR)
        if createData:
            dataPath = "/home/ec2-user/rajat/rubic_training_data_JUL15_2020.pickle"
            #dataPath = "/Users/ragarwal/Work/Intralinks/ContentOrganization/Data/rubic_training_data_JUL15_2020.pickle"
            data = pd.read_pickle(dataPath)
            data['label'] = data['label'].replace("operation", 'operations')
            data = data.sample(frac=1).reset_index(drop=True)
            contentData = data[['clean_content_v1', 'label']]
            contentData = contentData.rename(columns={"clean_content_v1": "text"})
            contentData.to_pickle(os.path.join(BASE_DIR, "fullContentData.pkl"))
        else:
            contentData = pd.read_pickle(os.path.join(BASE_DIR, "fullContentData.pkl"))

        validationData = validationDataOriginal[['content', 'fileName', 'label']]
        validationData = validationData.rename(columns={"content": "text"})
        validationData = validationData.reset_index(drop=True)
        longModel = LongFormerContentClassifier(BASE_DIR, mode="train")
        longModel.train(contentData, validationData, len(contentData.index), trainFromScratch=True)
        print("After Training - accuracy {}".format(longModel.accuracyList))
        print("After Training - Training loss List {}".format(longModel.avg_train_losses))
        print("After Training - Validation loss List {}".format(longModel.avg_valid_losses))
    else:
        #modelOverRideForEval = "/Users/ragarwal/eclipse-workspace/BertLong/src/longformer_content/Longformer_Content_Model.pt"
        modelOverRideForEval = None
        longModel = LongFormerContentClassifier(BASE_DIR, modelOverRideForEval= modelOverRideForEval, mode="eval")
        debugList = []
        matchCount = 0
        for index, row in tqdm(validationDataOriginal.iterrows()):
            predictedCategory = longModel.predict(row['content'], processed=True)[0]
            debugList.append([row['fileName'], predictedCategory, row['label']])
            if (predictedCategory == row['label']):
                matchCount = matchCount + 1
        debugDF = pd.DataFrame(debugList, columns=['filename', 'predictedCategory', 'actualCategory'])
        debugDF.to_csv("bert_long_debug.csv")
        accuracy = (matchCount / len(debugDF.index)) * 100
        print(accuracy)
