#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 25 16:20:37 2020

@author: andrew

data_cleaning

cleans train.csv and test.csv

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
import titanic_methods as ttm



          

pd.set_option('display.max_columns',None)
pd.set_option('display.max_rows',15)
pd.set_option('display.precision',3)
pd.set_option('display.width',None)

data_to_clean = ['train.csv','test.csv']

for file_name in data_to_clean:
    df = pd.read_csv(file_name)
    
    #separate names to First, Last, and Title
    df[['Last_Name', 'Title_First_Name']] = df['Name'].str.split(
        pat=',',
        expand=True,
        n=1)
    df[['Title','First_name']] = df.Title_First_Name.str.split(
        pat='.',
        expand=True,
        n=1)
    
    
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
    
    #separates Miss title into older and younger women
    def young_lady(df):
        
        is_young = []
        
        for index, title in df.Title.items():
            if title == 'Miss' and 0 < df.Parch.iloc[index] < 3:
                is_young.append(1)
            else:
                is_young.append(0)
        
        return is_young
    
    df['young_lady'] = young_lady(df)
    
    #transforms nan Age to median ages of people
    df.Age.fillna(df.groupby(['Pclass','Title','Sex','young_lady'])['Age'].transform('median'), inplace=True)
    
    #Drop any unnecessary Columns here:
    df.drop(columns=[
        'Cabin', 
        'Ticket', 
        'young_lady', 
        'First_name', 
        'Last_Name',
        'Title_First_Name',
        'Name'], inplace = True)
    df.dropna(axis = 0, subset=['Embarked'], how='all', inplace=True)
    
    df.Fare.fillna(0, inplace=True)

    file_name = 'cleaned_'+file_name.split('.')[0]
    
    ttm.write_pickle_df(df,file_name)
    print('Data cleaned.  Saved as {}.pickle'.format(file_name))



if __name__ == '__main__':
    pass





    


