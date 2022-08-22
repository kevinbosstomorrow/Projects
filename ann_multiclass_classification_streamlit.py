# -*- coding: utf-8 -*-
"""
Created on Thu Apr 14 15:26:40 2022
streamlit run desktop\a1.py

@author: kevinz
"""

import streamlit as st
import plotly.express as px 
import keras
import matplotlib.pyplot as plt
from sklearn.datasets import make_circles,make_moons,make_blobs,make_friedman1,make_regression
import tensorflow as tf
import pandas as pd
import numpy as np
import os
import time
from sklearn.metrics import accuracy_score
from sklearn.metrics import mean_squared_error,confusion_matrix,accuracy_score,precision_score,classification_report

plt.style.use('dark_background')

st.set_page_config(
    page_title = 'Real-Time Neural network learning',
    page_icon = '✅',
    layout = 'wide'
)



N = 100 # number of points per class
D = 2 # dimensionality
K = 5 # number of classes
x = np.zeros((N*K,D)) # data matrix (each row = single example)
y = np.zeros(N*K, dtype='uint8') # class labels

# 画spiral data的代码
for j in range(K):
    ix = range(N*j,N*(j+1))
    r = np.linspace(0.0,1,N) # radius
    t = np.linspace(j*4,(j+1)*4,N) + np.random.randn(N)*0.2 # theta
    x[ix] = np.c_[r*np.sin(t), r*np.cos(t)]
    y[ix] = j



plt.scatter(x[:, 0], x[:, 1], c= y, cmap=plt.cm.Spectral,edgecolors='black')




activation = 'sigmoid'
neuron = 168
epoch = 200

placeholder = st.empty()

class CusCallback(keras.callbacks.Callback):
    def __init__(self, x, y):
        self.x = x
        self.y = y
    
    def on_epoch_end(self, epoch, logs=None):
        y_pred = self.model.predict(self.x)
        y_pred = np.round(y_pred)

        #ac = np.round(accuracy_score(self.y, np.round(y_pred)),2)
        '''
        c = classification_report(self.y, y_pred, digits = 2,output_dict=True)
        c = pd.DataFrame(c).transpose()
        c = c.iloc[:2,:][['precision', 'recall', 'f1-score', 'support']]
        c.index = ['🔴正常细胞', '🔵癌细胞']
        c = pd.DataFrame(c)
        c.columns = ['精确度 %','识别率 %', '总分 %', '细胞数量']
        c['细胞数量'] = c['细胞数量'].values.astype(int)
        '''

        
        
        with placeholder.container():

            #st.write(f'Activation(激活函数)😈: {activation}')

            #st.write(f'number of Neurons(神经元的数量): {neuron}')
  
            st.subheader(f'Current epoch(当前训练次数): {epoch}')

            #st.title(f'Current accuracy(当前识别癌细胞准确率): {ac}')
            
            #fig = plt.scatter(x[:, 0], x[:, 1], c=np.round(y_pred), cmap=plt.cm.RdYlBu)
            #st.dataframe(c,500,100)
            
            f1,f2 = st.columns(2)
            
            '''
            with f1:
                #st.subheader('Original')
                fig = px.scatter(self.x, self.x[:, 0], self.x[:, 1], color=y,color_continuous_scale=px.colors.sequential.Bluered)
                fig.update_layout(title_text="Original",title_y= 1,title_x=0.5, font=dict(size=20,color="White" ))
                st.write(fig)
            '''
            with f1:
                x_min, x_max = -1.5, 1.5
                y_min, y_max = -1.5, 1.5
                xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100),np.linspace(y_min, y_max, 100))
                x_in = np.c_[xx.ravel(), yy.ravel()]
                
                y_pred2 = self.model.predict(x_in)
                y_pred2 = np.round(y_pred2).reshape(xx.shape)
               
                fig = plt.figure()
                plt.contourf(xx, yy, y_pred2, cmap=plt.cm.Spectral)
                plt.scatter(self.x[:, 0], self.x[:, 1], c= self.y, cmap=plt.cm.Spectral,edgecolors='black')
    
                st.pyplot(fig)                
                

                
            with f2:
                fig = plt.figure()
                plt.scatter(self.x[:, 0], self.x[:, 1], c= np.round(y_pred), cmap=plt.cm.Spectral, edgecolors='black')
                plt.xticks([-1.5,-1,-0.5,0,0.5,1, 1.5])
                plt.yticks([-1.5,-1,-0.5,0,0.5,1, 1.5])
                st.pyplot(fig)
            
    

model = tf.keras.Sequential([
  #tf.keras.layers.Dense(12, activation=tf.keras.activations.linear), # hidden layer 1, ReLU activation
  #tf.keras.layers.Dense(12, activation=tf.keras.activations.linear), # hidden layer 1, ReLU activation
  tf.keras.layers.Dense(neuron, activation = 'relu'), # hidden layer 2, ReLU activation
  tf.keras.layers.Dense(5, activation = activation) # ouput layer, sigmoid activation
])

# Compile the model
model.compile(loss=tf.keras.losses.SparseCategoricalCrossentropy(),
                optimizer=tf.keras.optimizers.Adam(lr = 0.001),
                metrics=['accuracy'])



model.fit(x, y, epochs=epoch, callbacks=[CusCallback(x,y)])
#model.summary()



















