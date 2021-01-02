'''
Created on 04-Oct-2019

@author: ragarwal
'''
import re
import string
import unicodedata

from gensim.parsing.preprocessing import preprocess_documents, strip_tags, \
    strip_punctuation, strip_numeric, remove_stopwords, strip_short, stem_text
from gensim.parsing.preprocessing import strip_multiple_whitespaces
from gensim.utils import tokenize
import nltk
from nltk.corpus import stopwords
from nltk.corpus import stopwords 
from nltk.corpus import wordnet
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
import numpy as np
import TextCleanerDicts
import unidecode




nltk.download('averaged_perceptron_tagger')
stop_words = set(stopwords.words('english')) 

def removeStopWordsAndTokenize(text):
    impWords = [word for word in word_tokenize(text) if word not in stop_words]
    return impWords


def remove_special_characters(text, remove_digits=False):
    pattern = r'[^a-zA-z0-9\s]' if not remove_digits else r'[^a-zA-z\s]'
    text = re.sub(pattern, ' ', text)
    return text

def remove_accented_chars(text):
    text = unicodedata.normalize('NFKD', text).encode('ascii', 'ignore').decode('utf-8', 'ignore')
    return text

def nltk_tag_to_wordnet_tag( nltk_tag):
    if nltk_tag.startswith('J'):
        return wordnet.ADJ
    elif nltk_tag.startswith('V'):
        return wordnet.VERB
    elif nltk_tag.startswith('N'):
        return wordnet.NOUN
    elif nltk_tag.startswith('R'):
        return wordnet.ADV
    else:          
        return None

def lemmatize_sentence(sentence):
    lemmatizer = WordNetLemmatizer()
    #tokenize the sentence and find the POS tag for each token
    nltk_tagged = nltk.pos_tag(nltk.word_tokenize(sentence))  
    #tuple of (token, wordnet_tag)
    wordnet_tagged = map(lambda x: (x[0], nltk_tag_to_wordnet_tag(x[1])), nltk_tagged)
    lemmatized_sentence = []
    for word, tag in wordnet_tagged:
        if tag is None:
            #if there is no available tag, append the token as is
            lemmatized_sentence.append(word)
        else:        
            #else use the tag to lemmatize the token
            lemmatized_sentence.append(lemmatizer.lemmatize(word, tag))
    return " ".join(lemmatized_sentence)    


def transformSingle(x):
    processedx = strip_tags(x)
    processedx = strip_punctuation(processedx)
    processedx = strip_multiple_whitespaces(processedx)
    processedx = strip_numeric(processedx)
    processedx = remove_stopwords(processedx)
    processedx = strip_short(processedx)
    #processedx = lemmatize_sentence(processedx)
    processedx = remove_accented_chars(processedx)
    processedx = remove_special_characters(processedx)
    return processedx.lower()

def convert(name):
    s1 = re.sub(r"([a-z])([A-Z])",r"\1 \2", name)
    return s1

def cleanSingleSentenceWithoutRemovingMeaning(x, minsize=3):
    processedx = strip_tags(x)
    processedx = unidecode.unidecode(processedx)
    processedx = remove_accented_chars(processedx)
    processedx = remove_special_characters(processedx)
    processedx = strip_multiple_whitespaces(processedx)
    processedx = strip_numeric(processedx)
    processedx = strip_short(processedx, minsize=minsize)
    return processedx

# def processFileNamesWithFinanceDictAndClean(line, minsize=3):
#
#     line = re.sub("/"," ",line)
#     line = re.sub("_"," ",line)
#     line = re.sub("-"," ",line)
#     line = re.sub(r'[`\-=~!@#$%^+;\'\"|<,./<>?\s]'," ",line)
#     line = line + ' '
#     for find in sorted(TextCleanerDicts.financialDict,key=len, reverse=True):
#         findregex =  r' '  + find + ' '
#         replaceregex =  r'' +' ' +find + "-" +TextCleanerDicts.financialDict[find] + ' '
#         #print(findregex + ' ' + replaceregex)
#         line = re.sub(findregex,replaceregex,line)
#
#     line = cleanSingleSentenceWithoutRemovingMeaning(line, minsize=minsize)
#     line = convert(line)
#     line = strip_multiple_whitespaces(line)
#     return line.strip()


def processFileNamesWithFinanceDictAndClean(line, minsize=1):
    line = re.sub("/", " ", line)
    line = re.sub("_", " ", line)
    line = re.sub("\d+([.]\d+)*\s*[-]", "", line)
    line = re.sub("-", " ", line)
    line = re.sub(r'[.](pdf|PDF|txt|xlsx|docx|doc|jpg|JPEG|png|PNG|pptx|zip|gif|GIF)', "", line)
    line = re.sub(r'(pdf|PDF|txt|xlsx|docx|doc|jpg|JPEG|png|PNG|pptx|zip|gif|GIF)', "", line)
    line = re.sub(r'[`\-=~!@#$%^+;\'\"|<,./<>?\s]', " ", line)
    # line = ' ' + line + ' '
    line = line + ' '

    for find in sorted(TextCleanerDicts.financialDict, key=len, reverse=True):
        findregex = r' ' + find + ' '
        replaceregex = r'' + ' ' + find + "-" + TextCleanerDicts.financialDict[find] + ' '
        # print(findregex + ' ' + replaceregex)
        line = re.sub(findregex, replaceregex, line)

    line = cleanSingleSentenceWithoutRemovingMeaning(line, minsize=minsize)
    line = convert(line)
    line = strip_multiple_whitespaces(line)
    return line.strip()

if __name__ == "__main__":
    testString = "Q2 2018"
    print(processFileNamesWithFinanceDictAndClean(testString))
    
    
    
    
    
