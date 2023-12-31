# -*- coding: utf-8 -*-
"""Exercício 6.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1Wd29xO__wqXjHi_im2f4HJOIWB4lEohj
"""

import numpy as np
import matplotlib.pyplot as plt

def model(x_p,w):
  a = w[0] + np.dot(x_p,w[1:])
  return a

def g(w, x, y, P):
  cost = 0
  for p in range(P):
    x_1_p = x[0][p]
    x_2_p = x[1][p]
    y_p = y[0][p]
    x_p = np.array([x_1_p, x_2_p])
    regressao_em_p = model(x_p, w)
    cost += (regressao_em_p - y_p)**2
  return cost

# load in data
csvname = '3d_linregress_data.csv'
data = np.loadtxt(csvname,delimiter=',')
x = data[:-1,:]
y = data[-1:,:]
print(np.shape(x))
print(np.shape(y))

P = np.shape(x)[1]
A = np.array([
        np.ones(P),
        x[0,:],
        x[1,:]
    ])

M = np.matmul(A, A.transpose())
b = np.matmul(A, y[0])
w = np.linalg.solve(M, b)

print("w0: {}\nw1: {}\nw2: {}".format(w[0], w[1], w[2]))

def grad_w(w,X,Y,P):
  N = 3
  grad = np.zeros(N)
  for p in range(P):
    x_1_p = x[0][p]
    x_2_p = x[1][p]
    y_p = y[0][p]
    x_p = np.array([x_1_p, x_2_p])
    res = model(x_p,w)
    grad = grad + (res - y_p) * np.array([1, x_1_p, x_2_p])
    grad = 2 * grad/P
  return grad


def grad_desc(alpha,max_its,w,X,Y,P):
  for k in range(max_its):
    c1 = g(w,X,Y,P)
    w1 = w
    grad_eval = grad_w(w,X,Y,P)
    w = w - alpha * grad_eval
    c2 = g(w,X,Y,P)
    if c2 > c1:
      w = w1
      alpha = alpha/2
  return w, max(c1, c2)


# Otimizando
w_otimizado, custo_final = grad_desc(0.001,1000,w,x,y,P)

print("Função custo final:\n{}".format(custo_final))
print("\nCoeficientes não otimizados:")
print("w0: {}\nw1: {}\nw2: {}".format(w[0], w[1], w[2]))
print("\nCoeficientes otimizados:")
print("w0: {}\nw1: {}\nw2: {}".format(w_otimizado[0], w_otimizado[1], w_otimizado[2]))

def mean_squared_error(w, x, y, P):
  cost = 0
  for p in range(P):
    x_1_p = x[0][p]
    x_2_p = x[1][p]
    y_p = y[0][p]
    x_p = np.array([x_1_p, x_2_p])
    regressao_em_p = model(x_p, w)
    cost += (regressao_em_p - y_p)**2
  return cost / P

def mean_absolute_deviation(w, x, y, P):
  cost = 0
  for p in range(P):
    x_1_p = x[0][p]
    x_2_p = x[1][p]
    y_p = y[0][p]
    x_p = np.array([x_1_p, x_2_p])
    regressao_em_p = model(x_p, w)
    cost += abs(regressao_em_p - y_p)
  return cost / P

mse = mean_squared_error(w, x, y, P)
mad = mean_absolute_deviation(w, x, y, P)

print("MSE: {}\nMAD: {}".format(mse, mad))

