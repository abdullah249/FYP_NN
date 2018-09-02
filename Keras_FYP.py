# -*- coding: utf-8 -*-
"""
Created on Sun Sep  2 00:01:45 2018

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

os.chdir("D:\Data"); #change the directory of the
final=pd.read_csv('out3.csv') #importing a csv file
final.shape

data_dim = 22 #total number of features are 22
timesteps = 7 #for every patient there are 7 timesteps
num_classes = 2
batch_size = 67

final=final.drop(columns=['visit Date'])
final=final.drop('mrno',axis=1)
final=final.drop('total dialysis',1)

y=final.ExpSite.as_matrix() #The labeled data set y which contains either the person has died or not
final=final.drop('ExpSite',axis=1) #Dropping the labeled column ExpSite
x=final.as_matrix() #The dataset without labeled column
x=x.reshape([-1,7,22]) #reshaping series into numpy arrays
y=y.reshape([-1,7,1]) #reshaping series into numpy arrays

X_train,X_test,y_train,y_test =train_test_split(x, y, test_size=0.2336,shuffle=False) #splitting the data set

#LSTM
model = Sequential()
model.add(LSTM(7, return_sequences=False,batch_input_shape=(batch_size, timesteps, 22)))   #7 input nodes 
model.add(LSTM(100, return_sequences=True))
model.add(LSTM(1,activation='softmax', return_sequences=False)) #one output node and the output of the network should be within [0,...,1]
model.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
model.summary()
model.fit(x, y, batch_size=67, epochs=5 )

score=model.evaluate(x,y,verbrose=0)