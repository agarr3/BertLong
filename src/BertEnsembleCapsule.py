'''
Created on 03-Oct-2020

@author: ragarwal
'''
import gc
import math
import os

from sklearn.preprocessing import LabelBinarizer
from transformers import LongformerTokenizer, LongformerConfig, get_linear_schedule_with_warmup, BertConfig, \
    BertTokenizer, BertModel
from transformers.modeling_longformer import LongformerModel
from transformers import BertForMaskedLM

import joblib
import torch
from torch import cuda

from capsule.capsuleclassification import DenseCapsule
from preprocessing import transformSingle, processFileNamesWithFinanceDictAndClean
import pandas as pd
from torch.utils.data import Dataset
from torch.utils.data.dataloader import DataLoader
from torch.utils.data.sampler import SequentialSampler
from torch.hub import tqdm

from pytorchtools import EarlyStoppingAndCheckPointer, ModelCheckPointer
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns



class BertEnsembleModelConfig:
    defaultConfig = BertConfig()
    model_name = 'bert-base-uncased'
    labelEncoderFileName = 'labelEncoder_bert_ensemble.sav'
    savedModelFileName = 'Bert_Ensemble_Model_v1.pt'
    tokenizer = BertTokenizer.from_pretrained(model_name)
    MAX_LEN = 512
    MAX_LEN_FILENAME = 20
    TRAIN_BATCH_SIZE = [1,1,1,1,1]
    ACCUMULATION_STEPS = 1
    VALID_BATCH_SIZE = 2
    EPOCHS = 5

    LEARNING_RATE = 1e-03
    LERANING_RATE_DECAY_MANUAL = [1, 0.9, 0.9*0.9, 0.9*0.9*0.9, 0.9*0.9*0.9*0.9]
    LEARNING_RATE_AUTO_DECAY_FLAG = False
    LR_DECAY_MODE = "EPOCH"

    WEIGHT_DECAY = 0.0
    PATIENCE = 3
    WARM_UP_RATIO = 0.06
    WARM_UP_STEPS = 0
    max_grad_norm = None
    LOSS_FN = "BCEWithLogitsLoss"
    #LOSS_FN = "CrossEntropyLoss"
    #BERT_MODEL_OP = "last_hidden"
    BERT_MODEL_OP = "CLS"

    VALID_FNAME_LEN_TH = 18



class BertEnsembletModel(torch.nn.Module):
    '''
    classdocs
    '''

    def __init__(self, num_classes, configuration=None):
        super(BertEnsembletModel, self).__init__()
        if configuration:
            self.configuration = configuration
        else:
            self.configuration = BertEnsembleModelConfig()
        self.bert_model_content = BertModel.from_pretrained(self.configuration.model_name)
        self.bert_model_filename = BertModel.from_pretrained(self.configuration.model_name)

        self.digitcaps = DenseCapsule(in_num_caps=self.configuration.MAX_LEN + self.configuration.MAX_LEN_FILENAME,
                                      in_dim_caps=768,
                                      out_num_caps=num_classes, out_dim_caps=16,
                                      routings=3)

        # self.dense = torch.nn.Linear(2 * 768, 2 * 768)
        # self.dropout = torch.nn.Dropout(0.3)
        # self.classifier = torch.nn.Linear(2 * 768, num_classes)

    def forward(
            self,
            input_ids=None,
            attention_mask=None,
            token_type_ids=None,
            position_ids=None,
            head_mask=None,
            inputs_embeds=None,
            class_label=None,
            exclusion_mask= None
    ):

        content_output = self.bert_model_content(input_ids[0], attention_mask=attention_mask[0])
        filename_output = self.bert_model_filename(input_ids[1], attention_mask=attention_mask[1])

        if exclusion_mask:
            content_output = content_output * exclusion_mask[0]
            filename_output = filename_output * exclusion_mask[1]

        op_bert_ensemble = torch.cat([content_output[0], filename_output[0]], dim=1)

        op_bert_ensemble = utils.squash(op_bert_ensemble)

        classvecs = self.digitcaps(op_bert_ensemble)
        outputs = classvecs.norm(dim=-1)
        return outputs


class BertEnsembleClassifier(object):
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
            self.configuration = BertEnsembleModelConfig()
        self.BASE_DIR = base_dir
        self.labelEncoderPath = os.path.join(self.BASE_DIR, self.configuration.labelEncoderFileName)
        self.modelPath = os.path.join(self.BASE_DIR, self.configuration.savedModelFileName)
        self.device = 'cuda' if cuda.is_available() else 'cpu'

        self.tokenizer = self.configuration.tokenizer
        if mode == "eval":
            self.lb = joblib.load(self.labelEncoderPath)
            self.model = BertEnsembletModel(len(self.lb.classes_), self.configuration)
            if modelOverRideForEval:
                self.model.load_state_dict(torch.load(modelOverRideForEval, map_location=self.device))
                self.model.eval()
            else:
                modelCheckpointer = ModelCheckPointer()
                modelCheckpointer.loadBestModel(self.BASE_DIR, self.model, self.device)
                self.model.eval()
        elif mode == "train":
            self.avg_train_losses = []
            self.avg_valid_losses = []
            self.train_accuracy_list = []
            self.valid_accuracy_list = []
            self.LR = []
        else:
            raise Exception("illegal mode")

    def predict(self, document, fileName, processed=False):

        self.model.eval()
        if not processed:
            document = transformSingle(document)
            fileName = processFileNamesWithFinanceDictAndClean(fileName)

        document = " ".join(document.split(" ")[0:self.configuration.MAX_LEN])
        if len(fileName) > self.configuration.MAX_LEN:
            fileName = " ".join(fileName.split(" ")[0:self.configuration.MAX_LEN_FILENAME])

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

            ids = [ids]
            mask = [mask]

            contentIds = torch.tensor(ids, dtype=torch.long)
            contentMask = torch.tensor(mask, dtype=torch.long)
            contenteExclusionMask = torch.ones((1))

            inputs = self.tokenizer.encode_plus(
                fileName,
                None,
                add_special_tokens=True,
                max_length=self.configuration.MAX_LEN_FILENAME,
                pad_to_max_length=True,
                return_attention_mask=True,
                # return_tensors='pt',
                truncation=True
            )
            ids = inputs['input_ids']
            mask = inputs['attention_mask']

            ids = [ids]
            mask = [mask]

            fileNameIds = torch.tensor(ids, dtype=torch.long)
            fileNameMask = torch.tensor(mask, dtype=torch.long)

            fileNameExclusionMask = torch.ones((1))

            if (len(fileName) < self.configuration.VALID_FNAME_LEN_TH):
                fileNameExclusionMask = torch.zeros((1))

            inputs = {
                "input_ids": [contentIds, fileNameIds],
                "attention_mask": [contentMask, fileNameMask],
                "exclusion_mask": [contenteExclusionMask,fileNameExclusionMask]
            }

            outputs = self.model(**inputs)
            prediction = torch.sigmoid(outputs)

        return self.lb.inverse_transform(prediction.detach().numpy())

    def loss_fn(self, outputs, targets):
        if self.configuration.LOSS_FN == "CrossEntropyLoss":
            torch.nn.CrossEntropyLoss()(outputs, targets)
        elif self.configuration.LOSS_FN == "BCEWithLogitsLoss":
            return torch.nn.BCEWithLogitsLoss()(outputs, targets)

    def caps_loss(self, y_true, y_pred):
        """
        Capsule loss = Margin loss
        :param y_true: true labels, one-hot coding, size=[batch, classes]
        :param y_pred: predicted labels by CapsNet, size=[batch, classes]
        :return: Variable contains a scalar loss value.
        """
        L = y_true * torch.clamp(0.9 - y_pred, min=0.) ** 2 + \
            0.5 * (1 - y_true) * torch.clamp(y_pred - 0.1, min=0.) ** 2
        L_margin = L.sum(dim=1).mean()

        return L_margin

    def run_evaluation(self, model, validation_data_loader_content,validation_data_loader_filename):
        model.eval()
        valid_losses = []
        accuracyBoolList = []
        confusionMatrix = torch.zeros(len(self.lb.classes_), len(self.lb.classes_))
        for step, batch in tqdm(enumerate(zip(validation_data_loader_content, validation_data_loader_filename)), desc="running evaluation"):
            contentBatch, fileNameBatch = batch
            batch_1 = tuple(t.to(self.device) for t in contentBatch)
            batch_2 = tuple(t.to(self.device) for t in fileNameBatch)

            targets = batch_1[2]
            inputs = {
                "input_ids": [batch_1[0], batch_2[0]],
                "attention_mask": [batch_1[1],batch_2[1]],
                "exclusion_mask": [batch_1[3],batch_2[3]]
            }
            outputs = model(**inputs)
            loss = self.caps_loss(outputs, targets)
            valid_losses.append(loss.item())
            _, predicted = torch.max(outputs, 1)
            _, trueClass = torch.max(targets, 1)
            for trueClassLabel, predictedClassLabel in zip(trueClass, predicted):
                confusionMatrix[trueClassLabel, predictedClassLabel] = confusionMatrix[trueClassLabel, predictedClassLabel] +1
            booleans = (predicted == trueClass)
            accuracyBoolList.extend([boolean.item() for boolean in booleans])

            del outputs, inputs, batch_1, batch_2
            gc.collect()
        return valid_losses, accuracyBoolList, confusionMatrix

    def run_training(self, epoch, model, training_data_loader_content, training_data_loader_filename, optimizer, scheduler):

        model.train()
        model.zero_grad()
        train_losses = []
        accuracyBoolList = []
        confusionMatrix = torch.zeros(len(self.lb.classes_), len(self.lb.classes_))

        for step, batch in tqdm(enumerate(zip(training_data_loader_content,training_data_loader_filename)), desc="running training for epoch {}".format(epoch)):
            contentBatch, fileNameBatch = batch

            batch_1 = tuple(t.to(self.device) for t in contentBatch)
            batch_2 = tuple(t.to(self.device) for t in fileNameBatch)

            targets = batch_1[2]
            inputs = {
                "input_ids": [batch_1[0],batch_2[0]],
                "attention_mask": [batch_1[1],batch_2[1]],
                "exclusion_mask": [batch_1[3],batch_2[3]]
            }
            outputs = model(**inputs)
            loss = self.caps_loss(outputs, targets)
            train_losses.append(loss.item())
            if self.configuration.ACCUMULATION_STEPS != 1:
                loss = loss / self.configuration.ACCUMULATION_STEPS
            loss.backward()

            _, predicted = torch.max(outputs, 1)
            _, trueClass = torch.max(targets, 1)
            for trueClassLabel, predictedClassLabel in zip(trueClass, predicted):
                confusionMatrix[trueClassLabel, predictedClassLabel] = confusionMatrix[
                                                                           trueClassLabel, predictedClassLabel] + 1
            booleans = (predicted == trueClass)
            accuracyBoolList.extend([boolean.item() for boolean in booleans])

            if (step + 1) % self.configuration.ACCUMULATION_STEPS == 0:
                if self.configuration.max_grad_norm is not None:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), self.configuration.max_grad_norm)
                optimizer.step()  # Now we can do an optimizer step
                if self.configuration.LR_DECAY_MODE == "BATCH" and self.configuration.LEARNING_RATE_AUTO_DECAY_FLAG:
                    print("Learning rate decay at batch level, reducing LR")
                    scheduler.step()
                    self.LR.append(scheduler.get_lr())
                #optimizer.zero_grad()
                model.zero_grad()
                print('Epoch: {}, step: {},  Loss:  {}'.format(epoch, step, loss.item()))


            del outputs, inputs, batch_1, batch_2
            gc.collect()
        return train_losses, accuracyBoolList, confusionMatrix

    def train(self, training_data, validationData, trainDataSize, trainFromScratch=True):

        self.lb = LabelBinarizer()
        training_data['labelvec'] = self.lb.fit_transform(training_data['label']).tolist()

        training_data_content = training_data[['text', 'labelvec']]
        training_data_filename = training_data[['filename', 'labelvec']]
        training_data_filename = training_data_filename.rename(columns={"filename":"text"})

        joblib.dump(self.lb, self.labelEncoderPath)

        validationData['labelvec'] = self.lb.transform(validationData['label']).tolist()

        validationDataContent = validationData[['text', 'labelvec']]
        validationDataFileName = validationData[['filename', 'labelvec']]
        validationDataFileName = validationDataFileName.rename(columns={"filename": "text"})

        content_training_set = CustomDataset(training_data_content, self.tokenizer, self.configuration.MAX_LEN)
        filename_training_set = CustomDatasetFileName(training_data_filename, self.tokenizer, self.configuration.MAX_LEN_FILENAME, self.configuration.VALID_FNAME_LEN_TH)

        content_validation_set = CustomDataset(validationDataContent, self.tokenizer, self.configuration.MAX_LEN)
        filename_validation_set = CustomDatasetFileName(validationDataFileName, self.tokenizer, self.configuration.MAX_LEN_FILENAME, self.configuration.VALID_FNAME_LEN_TH)

        content_training_loader = DataLoader(content_training_set,
                                             batch_size=self.configuration.TRAIN_BATCH_SIZE[0],
                                             sampler=SequentialSampler(content_training_set), drop_last=False)
        filename_training_loader = DataLoader(filename_training_set,
                                             batch_size=self.configuration.TRAIN_BATCH_SIZE[0],
                                             sampler=SequentialSampler(filename_training_set), drop_last=False)

        content_validation_loader = DataLoader(content_validation_set,
                                             batch_size=self.configuration.VALID_BATCH_SIZE,
                                             sampler=SequentialSampler(content_validation_set), drop_last=False)
        filename_validation_loader = DataLoader(filename_validation_set,
                                               batch_size=self.configuration.VALID_BATCH_SIZE,
                                               sampler=SequentialSampler(filename_validation_set), drop_last=False)

        model = BertEnsembletModel(len(self.lb.classes_))

        # no_decay = ["bias", "LayerNorm.weight"]
        # optimizer_grouped_parameters = [{
        #     "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
        # }]

        no_decay = ["bias", "LayerNorm.weight"]
        if self.configuration.WEIGHT_DECAY == 0.0:
            print("weight decay parameter is 0 so, using no weight decay anywhere")
            optimizer_grouped_parameters = [{
                "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            }]
        else:
            print("weight decay parameter is {} so, using this weight decay anywhere".format(self.configuration.WEIGHT_DECAY))
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

        if self.configuration.LR_DECAY_MODE == "BATCH":
            t_total = len(content_training_loader) // self.configuration.ACCUMULATION_STEPS * self.configuration.EPOCHS
            print("Learning rate decay at batch level, t_total is {}".format(t_total))
        elif self.configuration.LR_DECAY_MODE == "EPOCH":
            t_total = self.configuration.EPOCHS
            print("Learning rate decay at epoch level, t_total is {}".format(t_total))

        if self.configuration.WARM_UP_STEPS == None:
            warmup_steps = math.ceil(t_total * self.configuration.WARM_UP_RATIO)
        else:
            warmup_steps = self.configuration.WARM_UP_STEPS
        optimizer = torch.optim.AdamW(params=optimizer_grouped_parameters, lr=self.configuration.LEARNING_RATE)
        scheduler = get_linear_schedule_with_warmup(
            optimizer, num_warmup_steps=warmup_steps, num_training_steps=t_total
        )

        savedEpoch = 0
        if not trainFromScratch:
            # PATH = previousSavedModel
            modelCheckpointer = ModelCheckPointer()
            savedEpoch = modelCheckpointer.load_checkpoint(self.BASE_DIR, model, self.device, optimizer, scheduler)
            # model.load_state_dict(torch.load(PATH, map_location=self.device))
        model.to(self.device)

        early_stopping = EarlyStoppingAndCheckPointer(patience=self.configuration.PATIENCE, verbose=True, basedir=self.BASE_DIR)

        #self.initialLog(model,content_training_loader,filename_training_loader,content_validation_loader,filename_validation_loader)
        for epoch in range(savedEpoch, self.configuration.EPOCHS):
            print("starting training. The LR is {}".format(scheduler.get_lr()))

            trainBatchSizeForEpoch = self.configuration.TRAIN_BATCH_SIZE[epoch]
            if trainBatchSizeForEpoch < 1:
                trainBatchSizeForEpoch = 1
            content_training_loader = DataLoader(content_training_set,
                                                 batch_size=trainBatchSizeForEpoch,
                                                 sampler=SequentialSampler(content_training_set), drop_last=False)
            filename_training_loader = DataLoader(filename_training_set,
                                                  batch_size=trainBatchSizeForEpoch,
                                                  sampler=SequentialSampler(filename_training_set), drop_last=False)

            if not self.configuration.LEARNING_RATE_AUTO_DECAY_FLAG:
                lrForEpoch = self.configuration.LEARNING_RATE * self.configuration.LERANING_RATE_DECAY_MANUAL[epoch]
                for param_group in optimizer.param_groups:
                    param_group['lr'] = lrForEpoch

            train_losses, accuracyBoolListTrain, confusionMatrixTrain = self.run_training(epoch, model, content_training_loader,filename_training_loader, optimizer, scheduler)
            if self.configuration.LR_DECAY_MODE == "EPOCH" and self.configuration.LEARNING_RATE_AUTO_DECAY_FLAG:
                scheduler.step()
                print("Learning rate decay at epoch level, reducing LR to {}".format(scheduler.get_lr()))
                self.LR.append(scheduler.get_lr())

            valid_losses, accuracyBoolListValid, confusionMatrixValid = self.run_evaluation(model, content_validation_loader,filename_validation_loader)

            train_loss = np.average(train_losses)
            valid_loss = np.average(valid_losses)
            accuracy_train = sum(accuracyBoolListTrain)/len(accuracyBoolListTrain) * 100
            accuracy_valid = sum(accuracyBoolListValid) / len(accuracyBoolListValid) * 100

            self.avg_train_losses.append(train_loss)
            self.avg_valid_losses.append(valid_loss)
            self.train_accuracy_list.append(accuracy_train)
            self.valid_accuracy_list.append(accuracy_valid)

            print('Epoch: {},  Total Train Loss:  {}'.format(epoch+1, train_loss))
            print('Epoch: {},  Total Validation Loss:  {}'.format(epoch+1, valid_loss))
            print('Epoch: {},  Total training accuracy:  {}'.format(epoch+1, accuracy_train))
            print('Epoch: {},  Total Validation accuracy:  {}'.format(epoch+1, accuracy_valid))

            early_stopping(valid_loss, model, optimizer, epoch, scheduler)
            self.visualizeTraining(epoch+1, confusionMatrixTrain)
            self.visualizeValAccuracy(epoch+1, confusionMatrixValid)
            self.visualizeLR(epoch + 1)
            if early_stopping.early_stop:
                print("Early stopping")
                self.model = model
                break

    def initialLog(self, model,
                   content_training_loader,
                   filename_training_loader,
                   content_validation_loader,
                   filename_validation_loader):
        train_losses, accuracyBoolListTrain, confusionMatrixTrain = self.run_evaluation(model,content_training_loader,filename_training_loader)
        valid_losses, accuracyBoolListValid, confusionMatrixValid = self.run_evaluation(model,
                                                                                        content_validation_loader,
                                                                                        filename_validation_loader)

        train_loss = np.average(train_losses)
        valid_loss = np.average(valid_losses)
        accuracy_train = sum(accuracyBoolListTrain) / len(accuracyBoolListTrain) * 100
        accuracy_valid = sum(accuracyBoolListValid) / len(accuracyBoolListValid) * 100

        self.avg_train_losses.append(train_loss)
        self.avg_valid_losses.append(valid_loss)
        self.train_accuracy_list.append(accuracy_train)
        self.valid_accuracy_list.append(accuracy_valid)
        self.visualizeTraining(0, confusionMatrixTrain)
        self.visualizeValAccuracy(0, confusionMatrixValid)

    def visualizeTraining(self, epoch, confusionMatrixTrain):
        # visualize the loss as the network trained
        fig = plt.figure(figsize=(10, 8))
        plt.plot(range(0, len(self.avg_train_losses)), self.avg_train_losses, label='Training Loss')
        plt.plot(range(0, len(self.avg_valid_losses)), self.avg_valid_losses, label='Validation Loss')

        # find position of lowest validation loss
        minposs = self.avg_valid_losses.index(min(self.avg_valid_losses))
        plt.axvline(minposs, linestyle='--', color='r', label='Early Stopping Checkpoint')

        plt.xlabel('epochs')
        plt.ylabel('loss')
        plt.ylim(0, 0.5)  # consistent scale
        plt.xlim(0, len(self.avg_train_losses))  # consistent scale
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.show()
        fig.savefig(os.path.join(self.BASE_DIR , 'loss_plot_{}.png'.format(epoch)), bbox_inches='tight')

        confusionMatrixTrain = confusionMatrixTrain.numpy()
        confusionMatrixTrain = confusionMatrixTrain / confusionMatrixTrain.astype(np.float).sum(axis=1, keepdims=True)
        hmap = sns.heatmap(confusionMatrixTrain, annot=True,
                           fmt='.2', cmap='Blues', annot_kws={"size": 6},xticklabels=self.lb.classes_, yticklabels=self.lb.classes_)
        hmap.set_xticklabels(hmap.get_xmajorticklabels(), fontsize=6)
        hmap.set_yticklabels(hmap.get_ymajorticklabels(), fontsize=6)
        figure = hmap.get_figure()
        figure.savefig(os.path.join(self.BASE_DIR ,'training_confusion_matrix_{}.png'.format(epoch)), dpi=400)

    def visualizeValAccuracy(self, epoch, confusionMatrix):
        # visualize the loss as the network trained
        fig = plt.figure(figsize=(10, 8))
        plt.plot(range(0, len(self.valid_accuracy_list)), self.valid_accuracy_list, label='Validation Accuracy')
        plt.plot(range(0, len(self.train_accuracy_list)), self.train_accuracy_list, label='Training Accuracy')

        # find position of lowest validation loss

        plt.xlabel('epochs')
        plt.ylabel('accuracy')
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.show()
        fig.savefig(os.path.join(self.BASE_DIR , 'accuracy_plot_{}.png'.format(epoch)), bbox_inches='tight')

        confusionMatrix = confusionMatrix.numpy()
        confusionMatrix = confusionMatrix / confusionMatrix.astype(np.float).sum(axis=1, keepdims=True)
        hmap = sns.heatmap(confusionMatrix , annot=True,
                    fmt='.2', cmap='Blues', annot_kws={"size": 6},xticklabels=self.lb.classes_, yticklabels=self.lb.classes_)
        hmap.set_xticklabels(hmap.get_xmajorticklabels(), fontsize=6)
        hmap.set_yticklabels(hmap.get_ymajorticklabels(), fontsize=6)
        figure = hmap.get_figure()
        figure.savefig(os.path.join(self.BASE_DIR ,'validation_confusion_matrix_{}.png'.format(epoch)), dpi=400)

    def visualizeLR(self, epoch):
        # visualize the loss as the network trained
        fig = plt.figure(figsize=(10, 8))
        plt.plot(range(0, len(self.LR)), self.LR, label='Learning rate')

        # find position of lowest validation loss

        if self.configuration.LR_DECAY_MODE == "EPOCH":
            plt.xlabel('epochs')
        else:
            plt.xlabel('step')
        plt.ylabel('learning rate')
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.show()
        fig.savefig(os.path.join(self.BASE_DIR , 'lr_plot_{}.png'.format(epoch)), bbox_inches='tight')


class CustomDatasetFileName(Dataset):

    def __init__(self, dataframe, tokenizer, max_len, valid_len_th):
        self.tokenizer = tokenizer
        self.data = dataframe
        self.text = dataframe.text
        self.targets = self.data.labelvec
        self.max_len = max_len
        self.valid_len_th = valid_len_th

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

        exclusionMask = torch.ones((1))

        if (len(text) < self.valid_len_th):
            exclusionMask = torch.zeros((1))

        return torch.tensor(ids, dtype=torch.long), torch.tensor(mask, dtype=torch.long), torch.tensor(
            self.targets[index], dtype=torch.float), exclusionMask


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

        exclusionMask = torch.ones((1))

        return torch.tensor(ids, dtype=torch.long), torch.tensor(mask, dtype=torch.long), torch.tensor(
            self.targets[index], dtype=torch.float), exclusionMask


if __name__ == '__main__':
    runMode = "train"
    #val_data_path = "/Users/ragarwal/eclipse-workspace/PyTEnsembleClassifier/src/doc_category_validation_data.pkl"
    val_data_path = "/home/ec2-user/rajat/doc_category_validation_data.pkl"
    validationDataOriginal = pd.read_pickle(val_data_path)
    BASE_DIR = "bert_ensemble_capsule_content"
    if(runMode=="train"):
        createData = False
        if not os.path.exists(BASE_DIR):
            os.mkdir(BASE_DIR)
        if createData:
            dataPath = "/home/ec2-user/rajat/rubic_training_data_JUL15_2020.pickle"
            #dataPath = "/Users/ragarwal/Work/Intralinks/ContentOrganization/Data/rubic_training_data_JUL15_2020.pickle"
            data = pd.read_pickle(dataPath)

            data['fileName'] = data['path'].apply(lambda x: x.split("/")[-1])
            data['fileName'] = data['fileName'].apply(lambda x: processFileNamesWithFinanceDictAndClean(x))

            data['label'] = data['label'].replace("operation", 'operations')
            data = data.sample(frac=1).reset_index(drop=True)
            contentFileNameData = data[['content_trim', 'fileName', 'label']]
            contentFileNameData = contentFileNameData.rename(columns={"content_trim": "text","fileName":"filename"})
            contentFileNameData.to_pickle(os.path.join(BASE_DIR, "fullContentFileNameCleanedData.pkl"))
        else:
            contentFileNameData = pd.read_pickle(os.path.join(BASE_DIR, "fullContentFileNameCleanedData.pkl"))

        validationData = validationDataOriginal[['content', 'fileName', 'label']]
        validationData = validationData.rename(columns={"content": "text","fileName":"filename"})
        validationData = validationData.reset_index(drop=True)
        bert_ensemble_model = BertEnsembleClassifier(BASE_DIR, mode="train")
        bert_ensemble_model.train(contentFileNameData, validationData, len(contentFileNameData.index), trainFromScratch=True)
        print("After Training - validation accuracy {}".format(bert_ensemble_model.valid_accuracy_list))
        print("After Training - traininng accuracy {}".format(bert_ensemble_model.train_accuracy_list))
        print("After Training - Training loss List {}".format(bert_ensemble_model.avg_train_losses))
        print("After Training - Validation loss List {}".format(bert_ensemble_model.avg_valid_losses))
    else:
        #modelOverRideForEval = "/Users/ragarwal/eclipse-workspace/BertLong/src/longformer_content/Longformer_Content_Model.pt"
        modelOverRideForEval = None
        bert_ensemble_model = BertEnsembleClassifier(BASE_DIR, modelOverRideForEval= modelOverRideForEval, mode="eval")
        debugList = []
        matchCount = 0
        for index, row in tqdm(validationDataOriginal.iterrows()):
            predictedCategory = bert_ensemble_model.predict(row['content'],row['fileName'], processed=True)[0]
            debugList.append([row['fileName'], predictedCategory, row['label']])
            if (predictedCategory == row['label']):
                matchCount = matchCount + 1
        debugDF = pd.DataFrame(debugList, columns=['filename', 'predictedCategory', 'actualCategory'])
        debugDF.to_csv("bert_emsemble_debug.csv")
        accuracy = (matchCount / len(debugDF.index)) * 100
        print(accuracy)
