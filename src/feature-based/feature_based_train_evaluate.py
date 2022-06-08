#!/usr/bin/env python
# coding: utf-8


import torch
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from sklearn.model_selection import train_test_split
from tqdm import tqdm, trange
import pandas as pd
import io
import random
import numpy as np
from sklearn import tree
from sklearn import metrics 
from sklearn.naive_bayes import GaussianNB
from sklearn import linear_model


train_language = "zh"
test_language = "ja"


df_embeds_train = pd.read_csv(f"Models/Monolingual/Embeddings/train_embeddings_{train_language}.csv", delimiter=',', header=0 )
df_embeds_train.head()


df_embeds_test = pd.read_csv(f"Models/Monolingual/Embeddings/test_embeddings_{test_language}.csv", delimiter=',', header=0 )
df_embeds_test.head()

 
X_train = df_embeds_train.values[0:, 1:].astype("float32")
Y_train = df_embeds_train.values[0:, 0].astype("float32")
X_test = df_embeds_test.values[0:, 1:].astype("float32")
Y_test = df_embeds_test.values[0:, 0].astype("float32")


#### Gaussian (Naive Bayes Model)
naive = GaussianNB()
naive.fit(X_train, Y_train)
naive_test = naive.predict(X_test)
print("Gaussian Accuracy:",metrics.accuracy_score(Y_test, naive_test))



## SGD 
svm_final = linear_model.SGDClassifier()
svm_final.fit(X_train, Y_train)
svm_final_test = svm_final.predict(X_test)
print("SGD Accuracy:",metrics.accuracy_score(Y_test, svm_final_test))



# Decision Tree
clf_gini = tree.DecisionTreeClassifier(criterion = "gini", max_depth = 5)
clf_gini.fit(X_train, Y_train) 
clf_gini_test = clf_gini.predict(X_test)
print("Decision Tree Accuracy:",metrics.accuracy_score(Y_test, clf_gini_test))



