# -*- coding: utf-8 -*-
"""
Created on Mon Jul 23 10:56:27 2018

@author: Abdullah
"""

from keras.models import Sequential
from keras.layers import LSTM, Dense
from keras.layers import Dense, Dropout
from keras.layers import Flatten
import numpy as np
import pandas as pd
import os
from sklearn.model_selection import train_test_split

os.chdir("E:\Data");
final=pd.read_csv('out4.csv')
final.shape

data_dim = 22
timesteps = 7
num_classes = 2
batch_size = 67



final=final.drop(columns=['visit Date'])
final=final.drop('mrno',axis=1)
final=final.drop('total dialysis',1)
final=final.drop('Unnamed: 0',1)


y=final.ExpSite.values
final=final.drop('ExpSite',axis=1)
x=final.values
x=x.reshape([-1,7,22])
y=y.reshape([-1,7,1])

X_train,X_test,y_train,y_test =train_test_split(x, y, test_size=0.2336,shuffle=False)

#LSTM
model = Sequential()
model.add(LSTM(7, return_sequences=False,batch_input_shape=(batch_size, timesteps, 22)))  
model.add(LSTM(100, return_sequences=False))  
model.add(LSTM(1,activation='softmax', return_sequences=False)) 
model.add(Flatten())
model.add(Dense(2, activation='softmax'))
model.compile(loss='binary_crossentropy',
              optimizer='rmsprop',
              metrics=['accuracy'])

model.summary()
model.fit(x,y,
          batch_size=67, epochs=5
          )
score=model.evaluate(x,y,verbrose=0)

#MLP
model = Sequential()
model.add(Dense(7, input_dim=22, activation='relu'))
model.add(Dense(56, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

model.compile(loss='binary_crossentropy',
              optimizer='rmsprop',
              metrics=['accuracy'])

model.fit(x,y,
          epochs=20,
          batch_size=112)

score = model.evaluate(x,y,batch_size=112)










