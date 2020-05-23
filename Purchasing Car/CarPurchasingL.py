# -*- coding: utf-8 -*-
"""
Created on Sat May 23 06:13:16 2020

@author: Achyuth
"""
#logictic Regression
import pandas as pd
import numpy as np

dataset=pd.read_csv('Purchasing.csv')
X=dataset.iloc[:, :-1].values
y=dataset.iloc[:,-1].values

from sklearn.linear_model import LogisticRegression
classifier=LogisticRegression(random_state=0)
classifier.fit(X,y)
X_test=np.array([[int(input("Enter the person\'s Age:")),int(input('Enter person\'s Salary:'))]])

y_pred=classifier.predict(X_test)

if y_pred[0]==1:
    print('This person will Buy the Car')
else:
    print('This person is not planning to Buy the Car')