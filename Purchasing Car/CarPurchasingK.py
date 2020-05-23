# -*- coding: utf-8 -*-
"""
Created on Sat May 23 08:45:19 2020

@author: Achyuth
"""
#K-NN
import pandas as pd
import numpy as np

dataset=pd.read_csv('Purchasing.csv')
X=dataset.iloc[:, :-1].values
y=dataset.iloc[:,-1].values

from sklearn.neighbors import KNeighborsClassifier
classifier=KNeighborsClassifier(n_neighbors=5, metric='minkowski', p=2)
classifier.fit(X,y)
X_test=np.array([[int(input("Enter the person\'s Age:")),int(input('Enter person\'s Salary:'))]])
y_pred=classifier.predict(X_test)
if y_pred[0]==1:
    print('This Person will Buy the Car')
else:
    print('This Person is not planning to Buy the Car')