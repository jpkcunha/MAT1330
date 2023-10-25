#%% IMPORTING PACKAGES

import numpy as np
import matplotlib.pyplot as plt


# %%

def pontos_2d(w, P):
    np.random.seed(30)
    x = np.array([np.ones(P),
                  np.random.random_sample(P,),
                  np.random.random_sample(P,)])
    Y = []


def g_ReLU(w, x, y):
    P = len(x[0,:])
    N = len(x[:,0])
    cost = 0
    for p in range(P):
        cost += max(0, -y[p]*np.dot(x[:,p], w))
    return cost/P

def grad_g_ReLU(w, x, y):
    P = len(x[0, :])
    N = len(x[:,0])
    grad = np.zeros(N)
    for p in range(P):
        if -y[p]*np.dot(x[:,p], w) < 0:
            k = 0
        else:
            k = -y[p]
        grad += k * x[:,p]
    grad /= P
    print(f'cost = {grad}')
    return grad

def gradient_descent(w, x, y, alpha, max_its):
    w_h = [w]
    cost_h = [g_ReLU(w, x, y)]
    for _ in range(max_its):
        w -= alpha * grad_g_ReLU()
        w_h.append(w)
        cost_h.append(g_ReLU(w, x, y))
    return w, w_h, cost_h

#%%

P = 100
w = np.array([-0.3, -0.8, 1])
[X, Y] = pontos_2d(w, P)

N = 3
w = np.ones(N)
max_its = 10
alpha = 1


w, w_h, cost_h = gradient_descent(w, X, Y, alpha, max_its)

step = 1