# -*- coding: utf-8 -*-
"""
Created on Fri May 15 18:22:15 2020

@author: Achyuth
"""

import pandas as pd
import numpy as np
import tensorflow as tf

dataset=pd.read_excel('PowerPLant.xlsx')
x=dataset.iloc[:,:-1].values
y=dataset.iloc[:,-1].values

powerout=tf.keras.models.Sequential()
powerout.add(tf.keras.layers.Dense(units=6,activation='relu'))
powerout.add(tf.keras.layers.Dense(units=6,activation='relu'))
powerout.add(tf.keras.layers.Dense(units=1))
powerout.compile(optimizer='adam',loss='mean_squared_error')
powerout.fit(x, y, epochs = 50)
print("Enter the hourly range ambient values\nPress enter to assign the default Values")
inp=np.array([[int(input("\nTemperature-->")or 17),int(input("\nExhaust vacuum-->")or 44),int(input("\nAmbient Pressure-->")or 1010),int(input("\nRelative Humidity-->")or 60)]])
pred=powerout.predict(inp)
print("The Power Outcome-->%.2f"%pred[0][0])