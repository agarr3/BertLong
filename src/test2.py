'''
Created on 03-Oct-2020

@author: ragarwal
'''
import joblib
import matplotlib.pyplot as plt

if __name__ == '__main__':
    pass

# lst = [True, False, False, True, True, False, True, False]
# accuracy = sum(lst)/len(lst) * 100
# print(accuracy)
#
# lb = joblib.load("/Users/ragarwal/eclipse-workspace/BertLong/src/bert_ensemble_content/labelEncoder_bert_ensemble.sav")
# print(len(lb.classes_))
#
# import seaborn as sns
# import numpy as np
# confusionMatrix = np.random.rand(16,16)
# confusionMatrix = confusionMatrix / confusionMatrix.astype(np.float).sum(axis=1, keepdims=True)
# hmap = sns.heatmap(confusionMatrix, annot=True,
#                    fmt='.2', cmap='Blues', annot_kws={"size": 6},xticklabels=lb.classes_, yticklabels=lb.classes_)
#
# hmap.set_xticklabels(hmap.get_xmajorticklabels(), fontsize = 6)
# hmap.set_yticklabels(hmap.get_ymajorticklabels(), fontsize = 6)
# figure = hmap.get_figure()
#
# figure.savefig("test_cm.png", dpi=400)
# plt.show()
# import numpy as np
# confusionMatrixTrain = np.array([[3,4,5],
#                                  [4,5,6]])
# confusionMatrixTrain = confusionMatrixTrain / confusionMatrixTrain.astype(np.float).sum(axis=1, keepdims=True)
# print(confusionMatrixTrain)


# lb = joblib.load("/Users/ragarwal/eclipse-workspace/BertLong/src/bert_ensemble_content/labelEncoder_bert_ensemble.sav")
#
# print(lb.classes_)


from transformers import T5Tokenizer, T5Model, T5ForConditionalGeneration

tokenizer = T5Tokenizer.from_pretrained('t5-small')
model = T5ForConditionalGeneration.from_pretrained('t5-small')

# sentence1 = "Representatives for Puretunes could not immediately be reached for comment on Thursday."
# sentence2 = "Puretunes representatives could not be located on Wednesday to comment on the suit"

# sentence1 = "My name is Rajat"
# sentence2 = "Modi is PM of India"

sentence1 = "How old are you?"
sentence2 = "What is your age?"

sentences = "stsb sentence1: " + sentence1 +  " sentence2: " + sentence2
input_ids = tokenizer(sentences, return_tensors='pt')['input_ids']
#print(input_ids)
outs = model.generate(input_ids)
print(outs)
outs = [tokenizer.decode(ids) for ids in outs]
print(outs)