#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 26 15:13:26 2020

@author: andrew
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

df = ttm.read_pickle('cleaned_df.pickle')




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


if __name__ == '__main__':
    pass    
    