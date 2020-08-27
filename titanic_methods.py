#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 26 16:37:52 2020

@author: andrew

Methods and Classes used in Titanic Project

"""
import pickle 
import pandas as pd
def write_pickle_df(df, pickle_name):
    """
    
    
    Parameters
    ----------
    df : DataFrame
        Cleaned data to be pickled
    pickle_name : string
        filename of pickle

    Returns
    -------
    None.

    """
    pickle_name += '.pickle'
    with open(pickle_name, 'wb') as to_write:
        pickle.dump(df, to_write)
        
def read_pickle(file_name):
    """
    

    Parameters
    ----------
    pickle : file
        pickle to be unpacked

    Returns
    -------
    Unpacked pickle

    """
        
    with open(file_name,'rb') as read_file:
        unpacked = pickle.load(read_file)
    
    return unpacked
    
def write_submission(preds, test_data,file_name):
    """
    

    Parameters
    ----------
    preds : numpyarray
        Prediction Array.
    test_data : DataFrame
        Test Dataframe.
    file_name : str
        Submission File name of csv

    Returns
    -------
    None.

    """
    test_data['Survived'] = preds
    predictions = test_data[['PassengerId','Survived']]
    predictions.to_csv(file_name, index=False)
    
   

if __name__ == '__main__':
    print('This is the package containing the methods used in the titanic project')