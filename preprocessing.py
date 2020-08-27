#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 27 12:01:01 2020

@author: andrew

Data Preprocessing for Titanic Data
"""


import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import pickle
import titanic_methods as ttm

from sklearn.preprocessing import StandardScaler, LabelEncoder

pd.set_option('display.max_columns',None)
pd.set_option('display.max_rows',15)
pd.set_option('display.precision',3)
pd.set_option('display.width',None)

data_to_prep = ['cleaned_train.pickle', 'cleaned_test.pickle']

for file_name in data_to_prep:
    df = ttm.read_pickle(file_name)
    
    #Data Preprocessing
    
    #set up LabelEncoder Object
    le = LabelEncoder()
    df.Sex = le.fit_transform(df.Sex)
    
    
    #Set up dummy variables for other labels
    df = pd.get_dummies(df, columns=['Title','Embarked'])
    
    file_name = 'prepped_'+file_name.split('.')[0]
    ttm.write_pickle_df(df,file_name)
    print('Prepped.  Saved as {}.pickle'.format(file_name))
    




if __name__ == '__main__':
    pass
