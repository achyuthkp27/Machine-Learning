# -*- coding: utf-8 -*-
"""
Created on Fri May 15 23:14:00 2020

@author: Achyuth
"""

import pandas as pd
import numpy as np

dataset=pd.read_csv('Company.csv')
x=dataset.iloc[:,:-1].values
y=dataset.iloc[:,-1].values

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder

a=ColumnTransformer(transformers=[('encoder',OneHotEncoder(),[3])],remainder='passthrough')
x=np.array(a.fit_transform(x))

from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(x, y)
c=int(input("Select the city where company is:\n1.Mumbai\n2.Hyderabad\n3.Banglore\n-->"))
if c==1:
    m,h,b=1,0,0
elif c==2:
    m,h,b=0,1,0
else:
    m,h,b=0,0,1
pred=np.array([[m,h,b,eval(input('Enter the R&D spent-->')),eval(input('Enter the spent amount on Administrative-->')),eval(input('Enter the Amount Spent on Marketing-->'))]])
y_pred = regressor.predict(pred)
print('Profit-->%.2f'%y_pred[0])
if y_pred[0]>100000:
    print("It is completely safe to Invest in this company")
elif y_pred[0]>50000:
    print('If you don\'t have any other option you can Invest in this company')
else:
    print('It is a risk to Invest in this company')