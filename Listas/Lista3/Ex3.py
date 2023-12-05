#%% IMPORT PACKAGES

import numpy as np
import matplotlib.pyplot as plt

# %% FUNCTIONS

def pontos_2d(P):
    np.random.seed(31)
    X = [2*np.random.random_sample(P)-1,
         2*np.random.random_sample(P)-1,
         [-1]*P]
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




# %%

P = 150
X = pontos_2d(P)
k = 5 # número de centroides
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
    
    
#%%

def calculate_inertia(xp, c, k):
    inertia = 0
    P = len(xp[:, 0])
    for i in range(P):
        cluster = int(xp[i, 2])
        inertia += d(xp[i, :], c[cluster, :])**2
    return inertia

# Elbow Method
def elbow_method(X, max_k):
    inertias = []
    for k in range(1, max_k + 1):
        c = np.zeros((k, 2))
        for i in range(k):
            c[i, 0] = X[i + 2, 0]
            c[i, 1] = X[i + 2, 1]
        for _ in range(10):
            k_means(X, c, k)
            centroide(c, k, X)
        inertia = calculate_inertia(X, c, k)
        inertias.append(inertia)
        print(k, inertia)
    return inertias

# Main code
P = 150
X = pontos_2d(P)

max_k = 10
inertias = elbow_method(X, max_k)

# Plot the elbow curve
plt.plot(range(1, max_k + 1), inertias, marker='o')
plt.title('Elbow Method for Optimal k')
plt.xlabel('Number of clusters (k)')
plt.ylabel('Inertia')
plt.show()

#%%


def calculate_silhouette(xp, k):
    P = len(xp[:, 0])
    silhouette_values = np.zeros(P)
    
    for i in range(P):
        a_i = 0 
        b_i = float('inf')
        
        cluster_i = int(xp[i, 2])
        
        for j in range(P):
            if i != j:
                cluster_j = int(xp[j, 2])
                if cluster_i == cluster_j:
                    a_i += d(xp[i, :], xp[j, :])
                else:
                    b_ij = d(xp[i, :], xp[j, :])
                    if b_ij < b_i:
                        b_i = b_ij
        
        if (P - 1) > 0:
            a_i /= (P - 1)
        
        silhouette_values[i] = (b_i - a_i) / max(a_i, b_i)
    
    return silhouette_values

def silhouette_method(X, max_k):
    silhouette_scores = []
    for k in range(2, max_k + 1):  # Silhouette method is typically used for k > 1
        c = np.zeros((k, 2))
        for i in range(k):
            c[i, 0] = X[i + 2, 0]
            c[i, 1] = X[i + 2, 1]
        for _ in range(10):  # Number of iterations for k-means
            k_means(X, c, k)
            centroide(c, k, X)

        # Calculate silhouette score
        silhouette_values = calculate_silhouette(X, k)
        silhouette_avg = np.mean(silhouette_values)
        silhouette_scores.append(silhouette_avg)
        print(k, silhouette_avg)
    return silhouette_scores

# Main code
P = 150
X = pontos_2d(P)

max_k = 10
silhouette_scores = silhouette_method(X, max_k)

# Plot the silhouette scores
plt.plot(range(2, max_k + 1), silhouette_scores, marker='o')
plt.title('Silhouette Method for Optimal k')
plt.xlabel('Number of clusters (k)')
plt.ylabel('Average Silhouette Score')
plt.show()

# %%
