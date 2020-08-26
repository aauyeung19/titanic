#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 26 15:19:36 2020

@author: andrew
"""

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import pickle
import titanic_methods as ttm

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix

pd.set_option('display.max_columns',None)
pd.set_option('display.max_rows',15)
pd.set_option('display.precision',3)
pd.set_option('display.width',None)

df = ttm.read_pickle('cleaned_df.pickle')

#Set up KNN estimator
scaler = StandardScaler()
scaler.fit(df[['Pclass','Sex','Age','SibSp','Parch','Fare']])
scaled_df = scaler.transform(df[['Pclass','Sex','Age','SibSp','Parch','Fare']])
scaled_df = pd.DataFrame(scaled_df, columns=['Pclass','Sex','Age','SibSp','Parch','Fare'])

X = scaled_df
y = df.Survived
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42)

knn = KNeighborsClassifier(n_neighbors=1)
knn.fit(X_train,y_train)

prediction = knn.predict(X_test)

print(confusion_matrix(y_test,prediction))
print(classification_report(y_test,prediction))

error_rate = []

for i in range(40,50):
    
    knn = KNeighborsClassifier(n_neighbors=i)
    knn.fit(X_train,y_train)
    pred_i = knn.predict(X_test)
    
    error_rate.append(np.mean(pred_i != y_test))
    
knn = KNeighborsClassifier(n_neighbors=44)
knn.fit(X_train,y_train)

prediction = knn.predict(X_test)

#metrics
print(confusion_matrix(y_test,prediction))
#print(classification_report(y_test,prediction))
    