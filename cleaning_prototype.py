#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 25 16:20:37 2020

@author: andrew

cleaning_data_prototype

args:
    train.csv (csv) : Titanic Dataset from Kaggle
returns:
    cleaned_train.csv (Pickle) : Pickled cleaned data. 
    
Log:
    8/24 Separated names for Names into first_name, last_name, and title
    8/24 Converted sex into an integer value for KNN  male,female = 0,1
    8/24 Converted NAN Age to average age associated with PClass
    8/26 Began to separate code into cleaning, visualization, KNN prototype
    8/26 TO DO: 
        -start logging Updates to GIT for version control
        -create separate file with methods to import 
        
    
    
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

def sex_to_number(sex):
    """
   
    
   Parameters
    ----------
    sex : str
        male or female

    Returns
    -------
    int
        0 for male
        1 for female
    """
    if sex == 'female':
        return 1
    else:
        return 0
                
           
pd.set_option('display.max_columns',None)
pd.set_option('display.max_rows',15)
pd.set_option('display.precision',3)
pd.set_option('display.width',None)
df = pd.read_csv('train.csv')

#separate names to First, Last, and Title
df[['Last_Name', 'Title_First_Name']] = df['Name'].str.split(
    pat=',',
    expand=True,
    n=1)
df[['Title','First_name']] = df.Title_First_Name.str.split(
    pat='.',
    expand=True,
    n=1)

#Remove transitional columns
df = pd.DataFrame(df[['PassengerId', 'Survived', 'Pclass', 'Sex', 'Age', 'SibSp',
       'Parch', 'Ticket', 'Fare', 'Cabin', 'Embarked', 'Title',
       'First_name','Last_Name']])


    
#create df with numerical data only...
df.Sex = df.Sex.apply(sex_to_number) 
age_by_class_mean = df.groupby(['Pclass'])['Age'].mean()

df.loc[df2['Pclass'] == 1,'Age'] = df.loc[df2['Pclass'] == 1,'Age'].fillna(
    value= age_by_class_mean[1])
df.loc[df2['Pclass'] == 2,'Age'] = df.loc[df2['Pclass'] == 2,'Age'].fillna(
    value= age_by_class_mean[2])
df.loc[df2['Pclass'] == 3,'Age'] = df.loc[df2['Pclass'] == 3,'Age'].fillna(
    value= age_by_class_mean[3])

ttm.write_pickle_df(df,'cleaned_df')


'''
df2 = pd.DataFrame(df[['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 
                       'Fare']])
df2.Sex = df2.Sex.apply(sex_to_number)


#Average Age of Histograms show that we can estimate the age of the
#unknown passengers
no_null_age = df[df['Age'].notnull()]
first_class = no_null_age[no_null_age.Pclass == 1]['Age']
second_class = no_null_age[no_null_age.Pclass == 2]['Age']
third_class = no_null_age[no_null_age.Pclass == 3]['Age']

kwargs = dict(alpha=0.5, bins=100, density=True, stacked=True)
plt.hist(first_class, label='First Class', color='g', **kwargs)
plt.hist(second_class, label='Second Class', color='r', **kwargs)
plt.hist(third_class, label='Third Class', color='b', **kwargs)

#Object with average ages by class

age_by_class_mean = df2.groupby(['Pclass'])['Age'].mean()
df2.loc[df2['Pclass'] == 1,'Age'] = df2.loc[df2['Pclass'] == 1,'Age'].fillna(
    value= age_by_class_mean[1])
df2.loc[df2['Pclass'] == 2,'Age'] = df2.loc[df2['Pclass'] == 2,'Age'].fillna(
    value= age_by_class_mean[2])
df2.loc[df2['Pclass'] == 3,'Age'] = df2.loc[df2['Pclass'] == 3,'Age'].fillna(
    value= age_by_class_mean[3])
'''


    


