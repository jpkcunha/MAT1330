#%%

import numpy as np
import matplotlib.pyplot as plt

#%%


def model(x_p,w):
    y = w[0] + np.dot(x_p,w[1:,])
    return y

def g(w,x,y):
    cost = 0
    P = y.size
    for p in range(P):
        x_p = x[p]
        y_p = y[p]
        # somatorio do custo
        cost = cost + (model(x_p,w) - y_p) ** 2

    return cost/P


def grad_w(w,X,Y):
    N = 2
    P = np.size(X)
    grad = np.zeros(N)
    for p in range(P):
        grad = grad + (model(X[p],w) - Y[p]) * [1,X[p]]
    grad = 2 * grad/P
    return grad


def grad_desc(alpha,max_its,w,X,Y):
    print('c=',g(w,X,Y))
    for k in range(max_its):
        c1 = g(w,X,Y)
        w1 = w
        grad_eval = grad_w(w,X,Y)
        w = w - alpha * grad_eval
        c2 = g(w,X,Y)
        print('c=',g(w,X,Y))
        if c2 > c1:
            w = w1
            alpha = alpha/2
            print('alpha=',alpha)
    return w



#%%


filename = "https://raw.githubusercontent.com/jermwatt/machine_learning_refined/602412f222afe4d5497472037b0e62002d5a1d65/exercises/ed_2/mlrefined_datasets/superlearn_datasets/student_debt.csv"
data = np.loadtxt(filename,delimiter=',')

print(data)
data[0,:] = (data[0,:]-2004)/10
print(data)

#%%

P = data[0].size
#data[1,35] = 0.1
#data[1,34] = 0.1
#data[1,36] = 0.1
#data[1,37] = 0.1
A = np.array([np.ones(P),data[0,:]])
A = A.transpose()
M = np.matmul(A.transpose(),A)
b = np.matmul(A.transpose(),data[1,:])
v = np.linalg.solve(M,b)
w = np.array([2,3])
w = grad_desc(10,100,w,data[0,:],data[1,:])
print('v=',v)
print(g(v,data[0,:],data[1,:]))
print('w=',w)
data[0,:] = 10*data[0,:]+2004
w[1] = w[1] / 10
w[0] = w[0] - 2004 * w[1]
v[1] = v[1] / 10
v[0] = v[0] - 2004 * v[1]
print('vn=',v)
print('wn=',w)


# %%

plt.figure(figsize = (16,8))
plt.scatter(
 data[0,:],
 data[1,:],
 c = 'red')
x = np.linspace(data[0,0],data[0,P-1],100)
yw = w[1] * x + w[0]
yv = v[1] * x + v[0]
#y = model(x,w)
plt.plot(x,yw,'b')
plt.plot(x,yv,'g')
plt.show()

# %%
