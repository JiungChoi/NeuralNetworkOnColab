# 자동차 속도에 따른 제동거리 데이터를 이용한 선형회귀 분석 (텐서플로우 API 사용)

import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tensorflow.keras.optimizers import Adam

carDF = pd.read_csv('cars.csv')
carDF.columns = ['speed', 'dist']
carDF

x_data = carDF['speed']
y_data = carDF['dist']

plt.plot(x_data, y_data, 'ro')
plt.show()

x = tf.constant( x_data ,dtype=tf.float32)
y = tf.constant( y_data ,dtype=tf.float32)

w = tf.Variable(tf.random.uniform([1]) ) 
b = tf.Variable( tf.random.uniform([1]) )

def compute_loss():
    hx = w*x + b
    cost = tf.reduce_mean( (hx-y)**2 )
    return cost

optimizer = Adam( learning_rate=0.1)
for i in range(1000):
    optimizer.minimize( compute_loss, var_list=[w,b]) 
    print( '코스트', compute_loss().numpy(), 'w=', w.numpy() ,'b=', b.numpy())

# 예측한 모델 
tt = np.linspace(13,25,50)
plt.plot(tt, w*tt + b)

def hxFn(x_data):
  xd = np.float32(x_data)
  hx = w*xd +b
  return hx.numpy()

hxFn(10)
hxFn([10, 2] )