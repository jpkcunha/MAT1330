#%% IMPORT PACKAGES

import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets


# %% FUNCTIONS

def pontos_2d(X):
    P = X.shape[-1]
    X = [list(X[0]), list(X[1]),[-1]*P]
    X = np.array(X)
    print(X)
    Xp = []
    for i in range(P):
        if ((np.abs(X[0,1]) > 0.2) and (np.abs(X[1,i]) > 0.2)):
            Xp.append(X[:,i])
    Xp = np.array(Xp)
    return Xp


def d(x1, x2):
    return np.sqrt((x1[0]-x2[0])**2 + (x1[1]-x2[1])**2)

def k_means(xp,c,k):
    P = len(xp[:,0])
    for i in range(P):
        minimo = d(xp[i,:], c[0,:])
        ind_k = 0
        for j in range(1,k):
            a = d(xp[i,:], c[j,:])
            if a < minimo:
                minimo = a
                ind_k = j
        xp[i,2] = int(ind_k)

def centroide(c, k, xp):
    P = len(xp[:,0])
    for i in range(k):
        c[i,0] = 0
        c[i,1] = 0
    ind = np.zeros(k)
    for i in range(P):
        cluster = int(xp[i,2])
        c[cluster, 0] += xp[i,0]
        c[cluster, 1] += xp[i,1]
        ind[cluster] += 1
    for j in range(k):
        c[j,0] /= ind[j]
        c[j,1] /= ind[j]
    # print(c)

#%%

# Loading the data
P = 50 # Number of data points
blobs = datasets.make_blobs(n_samples=P, centers=3, random_state=10)
X = np.transpose(blobs[0])
# scatter plot the dataset 
plt.scatter(X[0,:],X[1,:],c = 'k')
plt.show()
X = pontos_2d(X)

#%%



# %%


k = 3 # número de centroides
c = np.zeros((k,2)) # centroides
for i in range(k):
    c[i,0] = X[i+2,0]
    c[i,1] = X[i+2,1]
colors = ['r', 'g', 'b', 'y', 'm', 'c', 'k', 'w']

for m in range(10):
    k_means(X,c,k)
    P = len(X)
    ax = plt.gca()
    for i in range(P):
        ax.scatter(X[i,0], X[i,1], color = colors[int(X[i,2])])
    for i in range(k):
        ax.scatter(c[i,0], c[i,1], color = colors[i], marker = "*", s = 1000)
    plt.savefig(f"img/kmeans_{m}.png")
    plt.show()
    centroide(c, k, X)
    
