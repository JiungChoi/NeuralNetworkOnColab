import matplotlib.pyplot as plt
import numpy as np

## 입력 데이터 x의 개수 만큼 반복함 

def cost(x, y, w):
  c = 0
  for i in range(len(x)):
    hx = w*x[i]
    c = c + (hx -y[i])**2
  return c/(len(x))

x_data = [1, 2, 3]
y_data = [1, 2, 3]

plt.plot(x_data, y_data, 'ro')
plt.show()

print(cost(x_data, y_data, -1))
print(cost(x_data, y_data, 0))
print(cost(x_data, y_data, 1))
print(cost(x_data, y_data, 2))
print(cost(x_data, y_data, 3))

for w in np.linspace(-3, 5, 50):
  c = cost(x_data, y_data, w)
  print(w, c)
  plt.plot(w, c, 'ro')
plt.show

def gradient(x, y, w):
  c = 0
  for i in range(len(x)):
    hx = w*x[i]
    c = c + (hx -y[i])*x[i]
  return c/(len(x))

def showgradient():
  x_data = [1,2,3]
  y_data = [1,2,3]
  w = 10
  for i in range(200):
    c = cost(x_data, y_data, w)
    g = gradient(x_data, y_data, w)
    w = w - 0.1*g
    print(i, c, 'w=', w)
  print('최종 w= ',w)