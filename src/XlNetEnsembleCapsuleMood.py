'''
Created on 03-Oct-2020

@author: ragarwal
'''
import gc
import math
import os
import random

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer
from transformers import LongformerTokenizer, LongformerConfig, get_linear_schedule_with_warmup, BertConfig, \
    BertTokenizer, BertModel, XLNetConfig, XLNetTokenizer, XLNetModel
from transformers.modeling_longformer import LongformerModel
from transformers import BertForMaskedLM

import joblib
import torch
from torch import cuda

from capsule import utils
from capsule.capsuleclassification import DenseCapsule
from preprocessing import transformSingle, processFileNamesWithFinanceDictAndClean, \
    cleanSingleSentenceWithoutRemovingMeaning
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
    defaultConfig = XLNetConfig()
    #model_name = 'xlnet-base-cased'
    model_name = "xlnet-base-cased"
    labelEncoderFileName = 'labelEncoder_xlnet_mood.sav'
    savedModelFileName = 'Bert_Ensemble_Model_v1.pt'
    tokenizer = XLNetTokenizer.from_pretrained('xlnet-base-cased')
    if model_name == 'xlnet-base-cased':
        MAX_LEN = 1024
        TRAIN_BATCH_SIZE = [2, 2, 2, 1, 1]
        ACCUMULATION_STEPS = 1
    elif model_name == "xlnet-large-cased":
        MAX_LEN = 512
        TRAIN_BATCH_SIZE = [1, 1, 1, 1, 1]
        ACCUMULATION_STEPS = 2
    MAX_LEN_FILENAME = 20
    VALID_BATCH_SIZE = 1
    EPOCHS = 5

    LEARNING_RATE = 1e-05
    LERANING_RATE_DECAY_MANUAL = [1, 0.9, 0.9*0.9*0.9, 0.9*0.9*0.9*0.9, 0.9*0.9*0.9*0.9*0.9]
    LEARNING_RATE_AUTO_DECAY_FLAG = False
    LR_DECAY_MODE = "EPOCH"

    WEIGHT_DECAY = 0.0
    PATIENCE = 3
    WARM_UP_RATIO = 0.06
    WARM_UP_STEPS = 0
    max_grad_norm = None
    #LOSS_FN = "BCEWithLogitsLoss"
    #LOSS_FN = "CrossEntropyLoss"
    LOSS_FN = "MarginLoss"
    #BERT_MODEL_OP = "last_hidden"
    BERT_MODEL_OP = "CLS"

    VALID_FNAME_LEN_TH = 18
    HELD_OUT_VALIDATION = True
    CONDITIONAL_TRAINING = False

    DETERMINISTIC = True



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
        self.bert_model_content = XLNetModel.from_pretrained(self.configuration.model_name, mem_len=1024)

        if(self.configuration.model_name == "xlnet-large-cased"):
            self.digitcaps = DenseCapsule(in_num_caps=self.configuration.MAX_LEN,
                                          in_dim_caps=1024,
                                          out_num_caps=num_classes, out_dim_caps=16,
                                          routings=3)
        elif (self.configuration.model_name == "xlnet-base-cased"):
            self.digitcaps = DenseCapsule(in_num_caps=self.configuration.MAX_LEN,
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

        content_output = self.bert_model_content(input_ids, attention_mask=attention_mask)

        if exclusion_mask is not None and self.configuration.CONDITIONAL_TRAINING:
            content_output = content_output[0] * exclusion_mask[0].unsqueeze(dim=-1)
        else:
            content_output = content_output[0]


        op_bert_ensemble = utils.squash(content_output)

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

        if self.configuration.DETERMINISTIC:
            self.seed_everything()

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

    def predict(self, content, processed=False):

        self.model.eval()
        if not processed:
            content = cleanSingleSentenceWithoutRemovingMeaning(content)

        content = " ".join(content.split(" ")[0:self.configuration.MAX_LEN])

        with torch.no_grad():
            inputs = self.tokenizer.encode_plus(
                content,
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

            inputs = {
                "input_ids": contentIds,
                "attention_mask": contentMask,
                "exclusion_mask": contenteExclusionMask
            }

            outputs = self.model(**inputs)
            prediction = torch.sigmoid(outputs)

        return self.lb.inverse_transform(prediction.detach().numpy())

    def seed_everything(self, seed=1234):
        random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)
        os.environ['PYTHONHASHSEED'] = str(seed)
        torch.backends.cudnn.deterministic = True

    def loss_fn(self, outputs, targets):
        if self.configuration.LOSS_FN == "CrossEntropyLoss":
            torch.nn.CrossEntropyLoss()(outputs, targets)
        elif self.configuration.LOSS_FN == "BCEWithLogitsLoss":
            return torch.nn.BCEWithLogitsLoss()(outputs, targets)
        elif self.configuration.LOSS_FN == "MarginLoss":
            return self.caps_loss(targets, outputs)

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

    def run_evaluation(self, model, validation_data_loader_content):
        model.eval()
        valid_losses = []
        accuracyBoolList = []
        confusionMatrix = torch.zeros(len(self.lb.classes_), len(self.lb.classes_))
        for step, batch in tqdm(enumerate(validation_data_loader_content), desc="running evaluation"):
            contentBatch = batch
            batch_1 = tuple(t.to(self.device) for t in contentBatch)

            targets = batch_1[2]
            inputs = {
                "input_ids": batch_1[0],
                "attention_mask": batch_1[1],
                "exclusion_mask": batch_1[3]
            }
            outputs = model(**inputs)
            loss = self.loss_fn(outputs, targets)
            valid_losses.append(loss.item())
            _, predicted = torch.max(outputs, 1)
            _, trueClass = torch.max(targets, 1)
            for trueClassLabel, predictedClassLabel in zip(trueClass, predicted):
                confusionMatrix[trueClassLabel, predictedClassLabel] = confusionMatrix[trueClassLabel, predictedClassLabel] +1
            booleans = (predicted == trueClass)
            accuracyBoolList.extend([boolean.item() for boolean in booleans])

            del outputs, inputs, batch_1
            gc.collect()
        return valid_losses, accuracyBoolList, confusionMatrix

    def run_training(self, epoch, model, training_data_loader_content, optimizer, scheduler):

        model.train()
        model.zero_grad()
        train_losses = []
        accuracyBoolList = []
        confusionMatrix = torch.zeros(len(self.lb.classes_), len(self.lb.classes_))

        for step, batch in tqdm(enumerate(training_data_loader_content), desc="running training for epoch {}".format(epoch)):
            contentBatch = batch

            batch_1 = tuple(t.to(self.device) for t in contentBatch)

            targets = batch_1[2]
            inputs = {
                "input_ids": batch_1[0],
                "attention_mask": batch_1[1],
                "exclusion_mask": batch_1[3]
            }
            outputs = model(**inputs)
            loss = self.loss_fn(outputs, targets)
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


            del outputs, inputs, batch_1
            gc.collect()
        return train_losses, accuracyBoolList, confusionMatrix

    def train(self, training_data, testing_data, trainDataSize, trainFromScratch=True):

        self.lb = LabelBinarizer()
        training_data['labelvec'] = self.lb.fit_transform(training_data['Mood']).tolist()

        training_data_content = training_data[['lyrics', 'labelvec']]
        training_data_content = training_data_content.rename(columns={"lyrics": "text"})
        joblib.dump(self.lb, self.labelEncoderPath)

        testing_data['labelvec'] = self.lb.transform(testing_data['Mood']).tolist()

        testingDataContent = testing_data[['lyrics', 'labelvec']]
        testingDataContent = testingDataContent.rename(columns={"lyrics": "text"})

        content_training_set = CustomDataset(training_data_content, self.tokenizer, self.configuration.MAX_LEN)
        content_testing_set = CustomDataset(testingDataContent, self.tokenizer, self.configuration.MAX_LEN)


        content_training_loader = DataLoader(content_training_set,
                                             batch_size=self.configuration.TRAIN_BATCH_SIZE[0],
                                             sampler=SequentialSampler(content_training_set), drop_last=False)


        content_testing_loader = DataLoader(content_testing_set,
                                               batch_size=self.configuration.VALID_BATCH_SIZE,
                                               sampler=SequentialSampler(content_testing_set), drop_last=False)


        model = BertEnsembletModel(len(self.lb.classes_))

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

        early_stopping = EarlyStoppingAndCheckPointer(patience=self.configuration.PATIENCE, verbose=True, basedir=self.BASE_DIR, epoch_level_save=False)

        #self.initialLog(model,content_training_loader, content_testing_loader)
        for epoch in range(savedEpoch, self.configuration.EPOCHS):
            print("starting training. The LR is {}".format(scheduler.get_lr()))

            trainBatchSizeForEpoch = self.configuration.TRAIN_BATCH_SIZE[epoch]
            if trainBatchSizeForEpoch < 1:
                trainBatchSizeForEpoch = 1
            content_training_loader = DataLoader(content_training_set,
                                                 batch_size=trainBatchSizeForEpoch,
                                                 sampler=SequentialSampler(content_training_set), drop_last=False)


            if not self.configuration.LEARNING_RATE_AUTO_DECAY_FLAG:
                lrForEpoch = self.configuration.LEARNING_RATE * self.configuration.LERANING_RATE_DECAY_MANUAL[epoch]
                for param_group in optimizer.param_groups:
                    param_group['lr'] = lrForEpoch
                    print("setting LR manually to {}".format(lrForEpoch))

            train_losses, accuracyBoolListTrain, confusionMatrixTrain = self.run_training(epoch, model, content_training_loader, optimizer, scheduler)
            if self.configuration.LR_DECAY_MODE == "EPOCH" and self.configuration.LEARNING_RATE_AUTO_DECAY_FLAG:
                scheduler.step()
                print("Learning rate decay at epoch level, reducing LR to {}".format(scheduler.get_lr()))
                self.LR.append(scheduler.get_lr())

            valid_losses, accuracyBoolListValid, confusionMatrixValid = self.run_evaluation(model, content_testing_loader)

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
                   content_testing_loader):
        train_losses, accuracyBoolListTrain, confusionMatrixTrain = self.run_evaluation(model,content_training_loader)
        valid_losses, accuracyBoolListValid, confusionMatrixValid = self.run_evaluation(model,content_testing_loader)

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

        fig = plt.figure(figsize=(10, 8))
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

        fig = plt.figure(figsize=(10, 8))
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
    BASE_DIR = "bert_mood_capsule_content"
    if(runMode=="train"):
        if not os.path.exists(BASE_DIR):
            os.mkdir(BASE_DIR)

        originalData = pd.read_pickle(os.path.join(BASE_DIR, "ml_balanced_data_augmented.pkl"))

        originalData = originalData.replace(np.nan, '', regex=True)

        originalData['lyrics'] = originalData['lyrics'].apply(lambda x: cleanSingleSentenceWithoutRemovingMeaning(x))

        originalData = originalData[originalData['lyrics'] != ""]

        originalData = originalData.reset_index(drop=True)

        train, test = train_test_split(originalData, test_size=0.1, random_state=0, stratify=originalData[['Mood']])
        train = train.reset_index(drop=True)
        test = test.reset_index(drop=True)
        bert_ensemble_model = BertEnsembleClassifier(BASE_DIR, mode="train")
        bert_ensemble_model.train(train, test, len(train.index), trainFromScratch=True)
        print("After Training - traininng accuracy {}".format(bert_ensemble_model.train_accuracy_list))
        print("After Training - Training loss List {}".format(bert_ensemble_model.avg_train_losses))

    else:
        pass
