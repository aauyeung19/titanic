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
    
    ###THERE IS A WAY TO DO THIS IN SciKitLearn! LabelEncoder! Research this.
                
           
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

df.Title = df.Title.str.strip()



def change_title(df):
    """
    

    Parameters
    ----------
    df : Dataframe

    Returns
    -------
    new_title : Series with converted Title Values
        

    """
    title_dict = {
        'Don' : 'Royalty',
        'Rev' : 'Officer',
        'Dr' : 'Officer',
        'Mme' : 'Royalty',
        'Ms' : 'Miss',
        'Major' : 'Officer',
        'Lady' : 'Royalty',
        'Sir' : 'Royalty',
        'Mlle' : 'Royalty',
        'Col' : 'Officer',
        'Capt' : 'Officer',
        'the Countess' : 'Royalty',
        'Jonkheer' : 'Royalty',
        'Mr' : 'Mr',
        'Mrs' : 'Mrs',
        'Master' : 'Master',
        'Miss' : 'Miss'
        }
    
    new_title = df.Title.map(title_dict)
    
    return new_title

df.Title = change_title(df)


def young_lady(df):
    
    is_young = []
    
    for index, title in df.Title.items():
        if title == 'Miss' and 0 < df.Parch.iloc[index] < 3:
            is_young.append(1)
        else:
            is_young.append(0)
    
    return is_young


df['young_lady'] = young_lady(df)



"""    
#Set nan age for remaining according to class average
There is a way to 
df.Sex = df.Sex.apply(sex_to_number) 
age_by_class_mean = df.groupby(['Pclass'])['Age'].mean()

df.loc[df['Pclass'] == 1,'Age'] = df.loc[df['Pclass'] == 1,'Age'].fillna(
    value= age_by_class_mean[1])
df.loc[df['Pclass'] == 2,'Age'] = df.loc[df['Pclass'] == 2,'Age'].fillna(
    value= age_by_class_mean[2])
df.loc[df['Pclass'] == 3,'Age'] = df.loc[df['Pclass'] == 3,'Age'].fillna(
    value= age_by_class_mean[3])

THE ONE LINE BELOW REPLACES THE ABOVE FOUR LINES
"""
df.Age.fillna(df.groupby(['Pclass','Title','Sex','young_lady'])['Age'].transform('median'), inplace=True)

#Drop any unnecessary Columns here:
df.drop(columns=['Cabin', 'Ticket', 'young_lady', 'First_name', 'Last_Name'], inplace = True)
df.dropna(axis = 0, subset=['Embarked'], how='all', inplace=True)


if __name__ == '__main__':
    ttm.write_pickle_df(df,'cleaned_df')
    print('Cleaned data saved as cleaned_df.pickle')





    


