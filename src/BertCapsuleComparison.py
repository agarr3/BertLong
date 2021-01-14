import json
import os

import torch
from sklearn.preprocessing import LabelBinarizer
from transformers import BertConfig, BertForSequenceClassification, BertModel, BertTokenizer

from capsule import utils
from capsule.capsuleclassification import DenseCapsule
from torch.utils.data import Dataset
from torch.utils.data.dataloader import DataLoader

from pytorchtools import EarlyStoppingAndCheckPointer

device = "cuda" if torch.has_cuda else "cpu"

import matplotlib.pyplot as plt
import numpy as np

class BaseConfig:
    defaultConfig = BertConfig()
    model_name = 'bert-base-uncased'
    labelEncoderFileName = 'labelEncoder_bert_ensemble.sav'
    MAX_LEN = 512
    tokenizer = BertTokenizer.from_pretrained(model_name)
    TRAIN_BATCH_SIZE = 2
    VALID_BATCH_SIZE = 2
    epoch = 1
    PATIENCE = 5
    IN_LEARNING_RATE = 1e-05
    LR_GAMMA = 0.9

class BertModelConfig(BaseConfig):
    BERT_MODEL_OP = "last_hidden"
    LOSS_FN = "BCEWithLogitsLoss"
    BASE_DIR = "./sts-comparison-vanilla"


class BertCapsuleConfig(BaseConfig):
    LOSS_FN = "MarginLoss"
    BASE_DIR = "./sts-comparison-capsule"

class BertVanillaModel(torch.nn.Module):
    '''
    classdocs
    '''

    def __init__(self, num_classes, configuration=None):
        super(BertVanillaModel, self).__init__()
        if configuration:
            self.configuration = configuration
        else:
            self.configuration = BertModelConfig()

        self.bert_model_content = BertModel.from_pretrained(self.configuration.model_name)
        self.dropout = torch.nn.Dropout(0.1)
        self.classifier = torch.nn.Linear(768, num_classes)

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

        if self.configuration.BERT_MODEL_OP == "last_hidden":
            op_content = content_output[1]
        elif self.configuration.BERT_MODEL_OP == "CLS":
            op_content = content_output[0][:, 0, :]


        x = self.dropout(op_content)
        x = self.classifier(x)

        # crossentropyloss: https://pytorch.org/docs/stable/nn.html#crossentropyloss
        if class_label is not None:
            loss_fct = torch.nn.CrossEntropyLoss(ignore_index=-1)
            next_sentence_loss = loss_fct(x.view(-1, 2), class_label.view(-1))
            return next_sentence_loss, x
        else:
            return x


class BertCapsuleModel(torch.nn.Module):
    '''
    classdocs
    '''

    def __init__(self, num_classes, configuration=None):
        super(BertCapsuleModel, self).__init__()
        if configuration:
            self.configuration = configuration
        else:
            self.configuration = BertCapsuleConfig()

        self.bert_model_content = BertModel.from_pretrained(self.configuration.model_name)
        self.digitcaps = DenseCapsule(in_num_caps=self.configuration.MAX_LEN,
                                      in_dim_caps=768,
                                      out_num_caps=num_classes, out_dim_caps=16,
                                      routings=3)


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
        op_bert_ensemble = utils.squash(content_output)
        classvecs = self.digitcaps(op_bert_ensemble)
        outputs = classvecs.norm(dim=-1)
        return outputs

class CustomDataset(Dataset):

    def __init__(self, dataframe, tokenizer, max_len):
        self.tokenizer = tokenizer
        self.data = dataframe
        self.text = dataframe.sentence
        self.targets = self.data.label
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

        label = self.targets[index]
        if label == 0:
            label = torch.tensor([1, 0])
        else:
            label = torch.tensor([0, 1])


        return torch.tensor(ids, dtype=torch.long), torch.tensor(mask, dtype=torch.long), torch.tensor(
            label, dtype=torch.float)

def caps_loss(y_true, y_pred):
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

def loss_fn(outputs, targets, config):
    if config.LOSS_FN == "CrossEntropyLoss":
        torch.nn.CrossEntropyLoss()(outputs, targets)
    elif config.LOSS_FN == "BCEWithLogitsLoss":
        return torch.nn.BCEWithLogitsLoss()(outputs, targets)
    elif config.LOSS_FN == "MarginLoss":
        return caps_loss(targets, outputs)


def test(model,  test_loader, config, num_classes):
    model.eval()
    with torch.no_grad:
        running_loss = 0.0
        confusionMatrix = torch.zeros(num_classes, num_classes)
        accuracyBoolList = []
        for i, batch in enumerate(test_loader, 0):
            # get the inputs; data is a list of [inputs, labels]
            contentBatch = batch
            batch_1 = tuple(t.to(device) for t in contentBatch)

            labels = batch_1[2]
            inputs = {
                "input_ids": batch_1[0],
                "attention_mask": batch_1[1]
            }

            # zero the parameter gradients

            # forward + backward + optimize
            outputs = model(**inputs)
            loss = loss_fn(outputs, labels, config)

            # print statistics
            running_loss += loss.item()

            _, predicted = torch.max(outputs, 1)
            _, trueClass = torch.max(labels, 1)
            for trueClassLabel, predictedClassLabel in zip(trueClass, predicted):
                confusionMatrix[trueClassLabel, predictedClassLabel] = confusionMatrix[
                                                                           trueClassLabel, predictedClassLabel] + 1
            booleans = (predicted == trueClass)
            accuracyBoolList.extend([boolean.item() for boolean in booleans])

        print('Total validation loss is {}'.format(running_loss))
        accuracy = sum(accuracyBoolList) / len(accuracyBoolList) * 100
        print('Total validation accuracy is {}'.format(accuracy))

        return running_loss, accuracy, confusionMatrix

def train(model, optimizer, scheduler, train_loader, test_loader, config, num_classes):

    losses = []
    accuracies = []
    con_matrices = []

    test_losses = []
    test_accuracies = []
    test_con_matrices = []

    if not os.path.exists(config.BASE_DIR):
        os.makedirs(config.BASE_DIR)

    early_stopping = EarlyStoppingAndCheckPointer(patience=vanillaConfig.PATIENCE, verbose=True, basedir=config.BASE_DIR,
                                                  epoch_level_save=True)
    for epoch in range(config.epoch):  # loop over the dataset multiple times

        model.train()
        running_loss = 0.0
        confusionMatrix = torch.zeros(num_classes, num_classes)
        accuracyBoolList = []
        for i, batch in enumerate(train_loader):
            # get the inputs; data is a list of [inputs, labels]
            contentBatch = batch
            batch_1 = tuple(t.to(device) for t in contentBatch)

            labels = batch_1[2]
            inputs = {
                "input_ids": batch_1[0],
                "attention_mask": batch_1[1]
            }

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = model(**inputs)
            loss = loss_fn(outputs, labels, config)
            loss.backward()
            optimizer.step()
            scheduler.step()

            # print statistics
            running_loss += loss.item()
            print('[epoch {}, batch {}] loss: {}'.format(epoch + 1, i + 1, loss.item()))

            _, predicted = torch.max(outputs, 1)
            _, trueClass = torch.max(labels, 1)
            for trueClassLabel, predictedClassLabel in zip(trueClass, predicted):
                confusionMatrix[trueClassLabel, predictedClassLabel] = confusionMatrix[
                                                                           trueClassLabel, predictedClassLabel] + 1
            booleans = (predicted == trueClass)
            accuracyBoolList.extend([boolean.item() for boolean in booleans])

        print('After epoch {}, the total loss is {}'.format(epoch,running_loss))
        accuracy = sum(accuracyBoolList)/len(accuracyBoolList) * 100
        print('After epoch {}, the total accuracy is {}'.format(epoch,accuracy))
        losses.append(running_loss)
        accuracies.append(accuracy)
        con_matrices.append(confusionMatrix)

        test_loss, test_accuracy, test_confusionMatrix = test(model, test_loader, config, num_classes)
        test_losses.append(test_loss)
        test_accuracies.append(test_accuracy)
        test_con_matrices.append(test_confusionMatrix)

        early_stopping(test_loss, model, optimizer, epoch, scheduler)

    return model, losses, accuracies, con_matrices, test_losses, test_accuracies, test_con_matrices

import seaborn as sns

def visualizeAccuracies( training_accuracies, test_accuracies, test_conf_matrix, config):
    # visualize the loss as the network trained
    fig = plt.figure(figsize=(10, 8))
    plt.plot(range(0, len(test_accuracies)), test_accuracies, label='Validation Accuracy')
    plt.plot(range(0, len(training_accuracies)), training_accuracies, label='Training Accuracy')

    # find position of lowest validation loss

    plt.xlabel('epochs')
    plt.ylabel('accuracy')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()
    fig.savefig(os.path.join(config.BASE_DIR , 'accuracy_plot.png'), bbox_inches='tight')

    fig = plt.figure(figsize=(10, 8))
    confusionMatrix = test_conf_matrix.numpy()
    confusionMatrix = confusionMatrix / confusionMatrix.astype(np.float).sum(axis=1, keepdims=True)
    hmap = sns.heatmap(confusionMatrix , annot=True,
                fmt='.2', cmap='Blues', annot_kws={"size": 6},xticklabels=[0,1], yticklabels=[0,1])
    hmap.set_xticklabels(hmap.get_xmajorticklabels(), fontsize=6)
    hmap.set_yticklabels(hmap.get_ymajorticklabels(), fontsize=6)
    figure = hmap.get_figure()
    figure.savefig(os.path.join(config.BASE_DIR ,'validation_confusion_matrix.png'), dpi=400)

def visualizeLoss(training_losses, test_losses, training_conf_matrix, config):
    # visualize the loss as the network trained
    fig = plt.figure(figsize=(10, 8))
    plt.plot(range(0, len(test_losses)), test_losses, label='Validation Losses')
    plt.plot(range(0, len(training_losses)), training_losses, label='Training Losses')

    # find position of lowest validation loss

    plt.xlabel('epochs')
    plt.ylabel('accuracy')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()
    fig.savefig(os.path.join(config.BASE_DIR , 'accuracy_plot.png'), bbox_inches='tight')

    fig = plt.figure(figsize=(10, 8))
    confusionMatrix = training_conf_matrix.numpy()
    confusionMatrix = confusionMatrix / confusionMatrix.astype(np.float).sum(axis=1, keepdims=True)
    hmap = sns.heatmap(confusionMatrix , annot=True,
                fmt='.2', cmap='Blues', annot_kws={"size": 6},xticklabels=[0,1], yticklabels=[0,1])
    hmap.set_xticklabels(hmap.get_xmajorticklabels(), fontsize=6)
    hmap.set_yticklabels(hmap.get_ymajorticklabels(), fontsize=6)
    figure = hmap.get_figure()
    figure.savefig(os.path.join(config.BASE_DIR ,'training_confusion_matrix.png'), dpi=400)

from datasets import load_dataset
dataset = load_dataset("glue", 'sst2')

train_df = dataset.get("train").data.to_pandas()
val_df = dataset.get("validation").data.to_pandas()
test_df = dataset.get("test").data.to_pandas()
num_classes = len(train_df['label'].unique())


# lb = LabelBinarizer()
# train_df['labelvec'] = lb.fit_transform(train_df['label']).tolist()
# val_df['labelvec'] = lb.transform(val_df['label']).tolist()
# test_df['labelvec'] = lb.transform(test_df['label']).tolist()

config = BaseConfig()
training_set = CustomDataset(train_df, config.tokenizer, config.MAX_LEN)
validation_set = CustomDataset(val_df, config.tokenizer, config.MAX_LEN)
testing_set = CustomDataset(test_df, config.tokenizer, config.MAX_LEN)

from torch.utils.data.sampler import SequentialSampler

training_loader = DataLoader(training_set,
                                     batch_size=config.TRAIN_BATCH_SIZE,
                                     sampler=SequentialSampler(training_set), drop_last=False)

validation_loader = DataLoader(validation_set,
                                    batch_size=config.VALID_BATCH_SIZE,
                                    sampler=SequentialSampler(validation_set), drop_last=False)

testing_loader = DataLoader(testing_set,
                                    batch_size=config.VALID_BATCH_SIZE,
                                    sampler=SequentialSampler(testing_set), drop_last=False)
## Vanilla Model
vanillaConfig = BertModelConfig()
model = BertVanillaModel(num_classes=num_classes)
model.to(device)


no_decay = ["bias", "LayerNorm.weight"]
optimizer_grouped_parameters = [{
                "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            }]

optimizer = torch.optim.AdamW(params=optimizer_grouped_parameters, lr=vanillaConfig.IN_LEARNING_RATE)
scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=vanillaConfig.LR_GAMMA)

model, losses, accuracies, con_matrices, test_losses, test_accuracies, test_con_matrices = train(model, optimizer,scheduler, training_loader,testing_loader, vanillaConfig, num_classes)
visualizeLoss(losses, test_losses, con_matrices[-1],vanillaConfig)
visualizeAccuracies(accuracies, test_accuracies, test_con_matrices[-1], vanillaConfig)

resultDict = {'train_losses':losses,
              'train_accuracies': accuracies,
              'test_losses':test_losses,
              'test_accuracies':test_accuracies}

with open(os.path.join(vanillaConfig.BASE_DIR, 'data.json'), 'w+') as fp:
    json.dump(resultDict, fp)

## Capsule Model
capsuleConfig = BertCapsuleConfig()
model = BertCapsuleModel(num_classes=num_classes)
model.to(device)


no_decay = ["bias", "LayerNorm.weight"]
optimizer_grouped_parameters = [{
                "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            }]

optimizer = torch.optim.AdamW(params=optimizer_grouped_parameters, lr=capsuleConfig.IN_LEARNING_RATE)
scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=capsuleConfig.LR_GAMMA)

model, losses, accuracies, con_matrices, test_losses, test_accuracies, test_con_matrices = train(model, optimizer, scheduler, training_loader,testing_loader, capsuleConfig, num_classes)
visualizeLoss(losses, test_losses, con_matrices[-1],capsuleConfig)
visualizeAccuracies(accuracies, test_accuracies, test_con_matrices[-1], capsuleConfig)

resultDict = {'train_losses':losses,
              'train_accuracies': accuracies,
              'test_losses':test_losses,
              'test_accuracies':test_accuracies}

with open(os.path.join(capsuleConfig.BASE_DIR, 'data.json'), 'w+') as fp:
    json.dump(resultDict, fp)
