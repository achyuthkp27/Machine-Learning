# -*- coding: utf-8 -*-
"""
Created on Sun May 24 08:45:26 2020

@author: Achyuth
"""

import numpy as np
import pandas as pd
dataset=pd.read_csv('Restaurant_Reviews.tsv',delimiter='\t',quoting=3)

import re
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
corpus=[]
for i in range(0,1000):
    reviews=re.sub('[^a-zA-Z]',' ',dataset['Review'][i])
    reviews=reviews.lower()
    reviews=reviews.split()
    ps=PorterStemmer()
    all_stopwords=stopwords.words('english')
    all_stopwords.remove('not')
    reviews=[ps.stem(word) for word in reviews if not word in set(all_stopwords)]
    reviews=' '.join(reviews)
    corpus.append(reviews)
    
from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(max_features = 1500)
X = cv.fit_transform(corpus).toarray()
y = dataset.iloc[:, 1].values

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state = 0)

from sklearn.naive_bayes import GaussianNB
classifier = GaussianNB()
classifier.fit(X_train, y_train)

y_pred = classifier.predict(X_test)

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
print(cm)