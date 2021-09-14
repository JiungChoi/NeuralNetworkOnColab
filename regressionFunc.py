import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

x = np.array( [1, 2, 3, 4, 5] )
y = np.array( [5, 8, 12, 15, 17] )

w = 5 
b = 5 
n = len(x)
epochs = 3000 
learning_rate = 0.01 

for i in range(epochs):
    hy = w*x + b
    cost = np.sum( (hy-y)**2 )/n # np.sum( (w*x + b-y)**2 )/n
    gradientW = np.sum( (w*x+b-y)*2*x)/n
    gradientB = np.sum( (w*x+b-y)*2)/n
    w = w- learning_rate*gradientW
    b = b- learning_rate*gradientB
    #print("cost:",cost ,"w:",w, "b:",b )
print("최종w",w)
print("최종b",b)