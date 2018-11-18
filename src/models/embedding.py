"""
@author: Negar Adyaniyazdi
"""

import pandas
import numpy as np
from gensim.test.utils import common_texts
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from nltk.tokenize.toktok import ToktokTokenizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.naive_bayes import MultinomialNB

tokenizer = ToktokTokenizer()


# read data and return preprocessed article text
def readData(filePath):
    data = pandas.read_csv(filePath)
    text = data.preprocessed_text.tolist()
    return text

# tokenization
def tokenizing_text(text):
    i = 0
    tokenized_texts = []
    tokens = []
    while i < len(text):
        tokens = tokenizer.tokenize(text[i])
        tokenized_texts.append(tokens)
        i = i + 1
    return tokenized_texts

# tag each document
def tag_documents(tokenized_texts):
    tagedDocuments = [TaggedDocument(doc, [i]) for i, doc in enumerate(tokenized_texts)]
    return tagedDocuments


# train embedding model on our training dataset or other corpus
def train_embd_model(tagedDocuments):
    model = Doc2Vec(tagedDocuments, vector_size=5, window=5, min_count=1, workers=4)

    return model


# infer vectors from out embedding model
def infer_vectors_from_embdModel(model, documents):
    j = 0
    vectors = []
    while j < len(documents):
        # print(documents[j][0])
        vectors.append(model.infer_vector(documents[j][0]))
        # vectors.append(vec)
        j = j + 1
    return vectors



# steps to get embedding vectors
#trainProcText = readData("C:/Users/Negar/Downloads/train10000_0.csv")
#testProcText = readData("C:/Users/Negar/Downloads/val2500_0.csv")

#tokenizedTrain = tokenizing_text(trainProcText)
#tokenizedTest = tokenizing_text(testProcText)

#trainTagDocuments = tag_documents(tokenizedTrain)
#model = train_embd_model(trainTagDocuments)

#testTagDocuments = tag_documents(tokenizedTest)

#trainVectors = infer_vectors_from_embdModel(model, trainTagDocuments)
#testVectors = infer_vectors_from_embdModel(model, testTagDocuments)


# train classifiers with embedding vectors
#clf = LogisticRegression(C=30.0, class_weight='None', solver='newton-cg')
#clf.fit(vectors, y_train) #X_train === vectors

# Get predictions and evaluate
#predicted = clf.predict(testvectors)
#accuracy = accuracy_score(y_test, predicted)
#print(accuracy)