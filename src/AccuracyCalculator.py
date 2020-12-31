'''
Created on 19-Sep-2020

@author: ragarwal
'''
import pandas as pd
from tqdm import tqdm
import requests
import json

from BertEnsemble import BertEnsembleClassifier
from BertLongEnsembleClassifier import BertLongEnsembleClassifierModel


def getTopNCategoriesforText(texts, n):

    API_URL = "http://0.0.0.0:6001"
    categories = []
    for text in tqdm(texts, desc="getting categories"):
        response = requests.post(
               '{}/v1/getCategory/{}'.format(API_URL, n),  data=text
           )
        categories.append(json.loads(response.content))

    return categories

validationData = pd.read_pickle("/home/ec2-user/rajat/doc_category_validation_data.pkl")
#validationData = pd.read_csv("doc_category_validation_data")

debugList = []

#model = BertLongEnsembleClassifierModel()
base_dir = "./bert_ensemble_content"
model = BertEnsembleClassifier(base_dir=base_dir, mode="eval")
 
matchCount = 0
 
for index, row in tqdm(validationData.iterrows()):
    predictedCategory = model.predict(row['content'], row['fileName'], processed=True)[0]
    debugList.append([row['fileName'], predictedCategory, row['label']])
    if(predictedCategory == row['label']):
        matchCount = matchCount +1 



# matchCount = 0
#  
# for index, row in tqdm(validationData.iterrows()):
#     predictedCategory = getTopNCategoriesforText([row['content']],  n=1)[0][0]
#     debugList.append([row['fileName'], predictedCategory, row['label']])
#     if(predictedCategory == row['label']):
#         matchCount = matchCount +1 

        

debugDF = pd.DataFrame(debugList, columns=['filename', 'predictedCategory', 'actualCategory'])

debugDF.to_csv("bert_long_debug.csv")

accuracy = (matchCount/len(debugDF.index)) * 100

print(accuracy)
        