# -*- coding: utf-8 -*-
"""
Created on Mon Sep 24 10:01:11 2018

@author: Abdullah
"""

# -*- coding: utf-8 -*-
"""
Created on Wed Sep 12 08:24:36 2018

@author: Abdullah
"""


from keras.models import Sequential
from keras.layers import LSTM, Dense
from keras.layers import GRU,SimpleRNN
from keras.layers import Dense, Dropout
from keras.layers import Flatten
import numpy as np
import pandas as pd
import os
from pandas import Series
from pandas import TimeGrouper
from matplotlib import pyplot
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from keras.utils import plot_model
from IPython.display import SVG
from keras.utils.vis_utils import model_to_dot
from keras.models import load_model



os.chdir("E:\Data"); #change the directory of the
final=pd.read_csv('final.csv') #importing a csv file

list(final)
final.shape

final=final.drop('Unnamed: 0',1)
final=final.drop( 'mrno',axis=1)


final.shape

y=final['ExpSite'].as_matrix() #The labeled data set y which contains either the person has died or not
final=final.drop('ExpSite',axis=1) #Dropping the labeled column ExpSite
x=final.as_matrix() #The dataset without labeled column
x = np.nan_to_num(x)

scaler = StandardScaler()
x = scaler.fit_transform(x)


x=x.reshape([-1,10,23]) #reshaping series into numpy arrays
y=y.reshape([-1,10,1]) #reshaping series into numpy arrays

yproper = []
for i in y:
	yproper.append(i[0])




y = yproper

x = np.array(x)
y = np.array(y)


X_train,X_test,y_train,y_test =train_test_split(x, y, test_size=0.10,shuffle=False) #splitting the data set
X_train.shape
X_test.shape
#LSTM
model = Sequential()
model.add(LSTM(40, return_sequences=True,input_shape=x.shape[1:]))   #7 input nodes 
model.add(LSTM(80, return_sequences=False))
model.add(Dense(1,activation='sigmoid')) #one output node and the output of the network should be within [0,...,1]
model.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
model.summary()
model.fit(X_train,y_train, batch_size=1, epochs=19)

score=model.evaluate(X_test,y_test)


#GRU
model = Sequential()
model.add(GRU(22, return_sequences=True,input_shape=x.shape[1:]))   #7 input nodes 
model.add(GRU(44, return_sequences=False))   #7 input nodes 
model.add(Dense(1,activation='sigmoid')) #one output node and the output of the network should be within [0,...,1]
model.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
model.summary()
model.fit(x,y, batch_size=1, epochs=19)

score=model.evaluate(X_test,y_test)

#RNN
model = Sequential()
model.add(SimpleRNN(64, return_sequences=True,input_shape=x.shape[1:]))   #7 input nodes 
model.add(SimpleRNN(100, return_sequences=False))   #7 input nodes 
model.add(Dense(1,activation='sigmoid')) #one output node and the output of the network should be within [0,...,1]
model.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
model.summary()
model.fit(X_train, y_train, batch_size=1, epochs=25)

score=model.evaluate(X_test,y_test)


model.save('my_model.h5')  

pred_y=model.predict(x)
pred_y=pred_y*100
pred_y[14]*100

SVG(model_to_dot(model).create(prog='dot', format='svg'))
plot_model(model)
os.environ["PATH"] += os.pathsep + 'F:\SofTwAres\Anaconda3-5.2.0\Lib\site-packages\graphviz'

