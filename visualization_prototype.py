#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 26 15:13:26 2020

@author: andrew

Exploratory Data Analysis to Look at relationships between data and Survival

"""

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import pickle
import titanic_methods as ttm

pd.set_option('display.max_columns',None)
pd.set_option('display.max_rows',15)
pd.set_option('display.precision',3)
pd.set_option('display.width',None)

cleaned_df = ttm.read_pickle('cleaned_df.pickle')
raw_df = pd.read_csv('train.csv')

#Compare Survival rate
plt.subplots(figsize=(7, 5))
sns.countplot(x='Survived', data=cleaned_df)
plt.title('Class Distribution')
plt.show()

#Survival Rate based on Class
fig, ax = plt.subplots(nrows=1, ncols=2)
sns.countplot(x=cleaned_df[cleaned_df.Survived == 1]['Pclass'], ax=ax[0], label='Survived')
sns.countplot(x=cleaned_df[cleaned_df.Survived == 0]['Pclass'], ax=ax[1])
ax[0].set_title('Survived')
ax[1].set_title('Deceased')
plt.title('Passenger Class Distribution')

#Survival Rate based on Sex
plt.subplots(figsize=(7,5))
sns.barplot(x='Sex', y='Survived', data=cleaned_df, ci=None)
plt.title('Gender Distribution')

#Survival based on Title
plt.subplots(figsize=(7,5))
sns.barplot(x='Title', y='Survived', data=cleaned_df, ci=None)
plt.title('Title Distribution')

#Surival based on Embarked location
plt.subplots(figsize=(7,5))
sns.barplot(x='Embarked', y='Survived', data=cleaned_df, ci=None)
plt.title('Survival based on Embarked location')

if __name__ == '__main__':
    pass    
    