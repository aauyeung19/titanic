#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 27 14:21:26 2020

KNN Submission Code

@author: andrew
"""

import pandas as pd
import titanic_methods as ttm

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
#from sklearn.metrics import classification_report, confusion_matrix

pd.set_option('display.max_columns',None)
pd.set_option('display.max_rows',15)
pd.set_option('display.precision',3)
pd.set_option('display.width',None)

train_data = ttm.read_pickle('prepped_cleaned_train.pickle')
test_data = ttm.read_pickle('prepped_cleaned_test.pickle')

for each in train_data.columns.drop('Survived'):
    if each not in test_data.columns:
        test_data[each] = 0
        
scaler = StandardScaler()

scaler.fit(train_data.drop(columns=['Survived']))
scaled_train = scaler.transform(train_data.drop(columns=['Survived']))
scaled_train = pd.DataFrame(scaled_train)


scaler.fit(test_data)
scaled_test = scaler.transform(test_data)
scaled_test = pd.DataFrame(scaled_test)

X_train = scaled_train
y_train = train_data.Survived
X_test = scaled_test


#choose parameters to tune
n_neighbors = list(range(1,60))
p = [1,2]

#create dictionary
hyperparameters = dict(n_neighbors=n_neighbors, p=p)

#create new KNN object
knn2 = KNeighborsClassifier()

#Use GridSearchCV
clf = GridSearchCV(estimator=knn2, param_grid=hyperparameters, cv=5)

clf.fit(X_train,y_train)
print(clf.best_params_)
print(clf.best_score_)

n = clf.best_params_['n_neighbors']
p = clf.best_params_['p']
knn = KNeighborsClassifier(n_neighbors=n, p=p)
knn.fit(X_train,y_train)
prediction = knn.predict(X_test)

ttm.write_submission(prediction, test_data, 'KNN_Submission.csv')