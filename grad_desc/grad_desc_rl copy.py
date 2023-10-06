#%%

import numpy as np
import matplotlib.pyplot as plt

#%%

# x: vetor [x1,x2,x3,...,xn]
# x_p: vetor [1,x1,x2,...,xn] 
# Logo: dim(x)=N, dim(w)=dim(x_p)=N+1
def model(x_p: np.array, w: np.array) -> float:
    return w[0] + np.dot(x_p,w[1:,]).item()

def g(w: np.array, x: np.array, y: np.array) -> float:
    cost = 0
    P = x.shape[0]
    for p in range(P):
        # Soma componentes do custo
        cost += (model(x[p],w) - y[p]) ** 2
    return cost/P


def grad(w: np.array, x: np.array, y: np.array) -> np.array:
    P = x.shape[0]
    total = np.zeros(w.size)
    for p in range(P):
        x_p = np.insert(x[p],0,1) # [1, x[p]]
        total += (model(x[p],w) - y[p]) * x_p
    return 2*total/P


def grad_desc(w: np.array, x: float, y: float,
              alpha: float, max_its: int) -> np.array:
    print('c=',g(w,x,y))
    for _ in range(max_its):
        c1 = g(w,x,y)
        w1 = w
        grad_eval = grad(w,x,y)
        w = w - alpha * grad_eval
        c2 = g(w,x,y)
        print('c=',g(w,x,y))
        if c2 > c1:
            w = w1
            alpha = alpha/2
            print('alpha=',alpha)
    return w


#%%


filename = "https://raw.githubusercontent.com/jermwatt/machine_learning_refined/602412f222afe4d5497472037b0e62002d5a1d65/exercises/ed_2/mlrefined_datasets/superlearn_datasets/student_debt.csv"
data = np.loadtxt(filename,delimiter=',')
data[0,:] = (data[0,:]-2004)/10
data

#%%

P = data[0].size
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

# Returning data to its original form
data[0,:] = 10*data[0,:]+2004
w[1] = w[1] / 10
w[0] = w[0] - 2004 * w[1]
v[1] = v[1] / 10
v[0] = v[0] - 2004 * v[1]

print('vn=',[round(num, 2) for num in v])
print('wn=',[round(num, 2) for num in w])

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
