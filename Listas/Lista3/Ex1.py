#%% IMPORTING PACKAGES

import numpy as np
import matplotlib.pyplot as plt

#%%

def pontos_3d(P):
    np.random.seed(31)
    X = [2 * np.random.random_sample(P) - 1,
         2 * np.random.random_sample(P) - 1,
         2 * np.random.random_sample(P) - 1]
    X = np.array(X)
    Xp = []
    t1, t2, t3 = 0.5, 0.5, -0.9
    R1 = [[1, 0, 0], [0, np.cos(t1), -np.sin(t1)], [0, np.sin(t1), np.cos(t1)]]
    R2 = [[np.cos(t2), 0, np.sin(t2)], [0, 1, 0], [-np.sin(t2), 0, np.cos(t2)]]
    R3 = [[np.cos(t3), -np.sin(t3), 0], [np.sin(t3), np.cos(t3), 0], [0, 0, 1]]
    R = np.dot(np.dot(R1, R2), R3)   
    for i in range(P):
        if ((X[0][i]**2)/0.85 + (X[1][i]**2)/0.41 + (X[2][i]**2)/0.626 - 1) < 0:
            Xp.append(np.dot(R, X[:,i]))
    return np.array(Xp)


def centraliza(X):
    P = len(X)
    s = X.shape[1]
    for i in range(P):
        s = s + X[i,:]
    X_mean = s / P
    X_centered = X - X_mean
    return X_centered


def PCA(X):
    P = len(X[0,:])
    Cov = 1/P * np.dot(X,X.T)
    D,V = np.linalg.eigh(Cov)
    return D,V


#%%

P = 70000
X = pontos_3d(P)
X = centraliza(X)
[D,V] = PCA(X.T)

fig = plt.figure(figsize = (12, 10))
plt.xlim(-1,1)
plt.ylim(-1,1)

ax = plt.axes(projection='3d')

# Reproduz os pontos de P
ax.scatter(X[:,0], X[:,1], X[:,2], s=1)

# Cores do gráfico
colors = ['red', 'green', 'orange']

for i in range(len(D)):
    x = [0, 2*np.sqrt(D[i])*V[i][0]]
    y = [0, 2*np.sqrt(D[i])*V[i][1]]
    z = [0, 2*np.sqrt(D[i])*V[i][2]]
    ax.plot(x, y, z, colors[i], linewidth=8)

plt.show()