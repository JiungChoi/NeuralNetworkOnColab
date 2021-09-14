import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.optimizers import Adam

x_data = [1,2,3,4,5] #x_data라는 constantce 노드를 추가한다.
y_data = [5,8,11,14,17] #y_data라는 constantce 노드를 추가한다.

x = tf.constant( x_data ,dtype=tf.float32)
y = tf.constant( y_data ,dtype=tf.float32)

w = tf.Variable( 10.0 ) #  w란 Variable 변수 노드를 추가한다.
# w.assign_sub(3) ##  w노드에 10.0 - 3 한 값을 할당 
b = tf.Variable( 10.0 )

def compute_loss():
    hx = w*x + b
    cost = tf.reduce_mean( (hx-y)**2 )
    return cost

optimizer = Adam( learning_rate=0.1)
for i in range(1000):
    optimizer.minimize( compute_loss, var_list=[w,b]) ## w= w-0.1*미분값 == w.assign_sub( 0.1*미분값 ) && (y값, x값)
    print( '코스트', compute_loss().numpy(), 'w=', w.numpy() ,'b=', b.numpy())