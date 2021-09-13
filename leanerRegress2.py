# 자동차 속도에 따른 제동거리 데이터를 이용한 선형회귀 분석(케라스 API)

import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from tensorflow.keras.layers import Dense
from tensorflow.keras import Sequential
from tensorflow.keras.optimizers import Adam

carDF = pd.read_csv('cars.csv')
carDF.columns = ['speed', 'dist']
carDF

x_data = carDF['speed']
y_data = carDF['dist']

w = tf.Variable(tf.random.uniform([1]) ) 
b = tf.Variable( tf.random.uniform([1]) )

IO = Dense(units = 1, input_dim = 1)
model = Sequential([IO])
model.compile(loss = 'mse', optimizer= Adam(0.1))
model.fit(x_data, y_data, epochs = 500)

# 가중치랑 바이어스 확인
w, b = IO.get_weights()
print(w)
print(b)

model.predict([10, 12])