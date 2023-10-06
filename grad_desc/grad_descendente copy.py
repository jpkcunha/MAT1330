#%%

import numpy as np
import matplotlib.pyplot as plt


#%%

def g(x: float, y:float) -> float:
    return x**2 + y**2 + 2 * np.sin(1.5 * (x + y))**2 + 2


def grad(x:float,y:float) -> np.array:
   return np.array([2*x+6*np.sin((1.5*(x+y)))*np.cos((1.5*(x+y))),
                    2*y+6*np.sin((1.5*(x+y)))*np.cos((1.5*(x+y)))])


def gradient_descent(w, alpha, max_its, tolerance):
    # gradient descent loop 
    cost_history = [g(w[0],w[1])]
    for k in range(max_its):
        # evaluate the gradient
        grad_eval = grad_w(w[0],w[1])
        print('grad = ', grad_eval)
        # take gradient descent step
        print([1,g(w[0],w[1])])
        custo1 = g(w[0],w[1])
        w = w - alpha*grad_eval
        print([2,g(w[0],w[1])])
        custo2 = g(w[0],w[1])
        #if custo2 > custo1:
        # w = w + alpha*grad_eval
        # alpha = alpha/2
        # print('retornou w = ',w)
        cost_history.append(g(w[0],w[1])) 
    return w, cost_history

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





#%%
csvname = 'https://raw.githubusercontent.com/jermwatt/machine_learning_refined/main/notes/5_Linear_regression/chapter_5_datasets/3d_linregress_data.csv'
data = np.loadtxt(csvname,delimiter=',')
x = data[:-1,:]
x1, x2 = x
y = data[-1:,:]
print(np.shape(x))
print(np.shape(y))



#%%

P = len(x)
# Criar matriz A
At = np.array([np.ones(P),x])
A = At.transpose()
M = np.matmul(At,A)
print(M.shape)
b = np.matmul(At,y)
print(b.shape)
w = np.linalg.solve(M,b)
w


#%%

# fig, ax = plt.subplots(subplot_kw={"projection": "3d"})

# Make data.

# x1, x2 = np.meshgrid(x1, x2)
# Create a 3D figure
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# # Extract the individual components from the input array x
# x1 = x[0, :]  # First component of x
# x2 = x[1, :]  # Second component of x

# Plot the 3D scatter plot
ax.scatter(x1, x2, y, c='b', marker='o', label='Data Points')

# Set labels for the axes
ax.set_xlabel('X1')
ax.set_ylabel('X2')
ax.set_zlabel('Y')
ax.legend()
 
plt.show()


#%%

# Create a 3D figure
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# # Extract the individual components from the input array x
# x1 = x[0, :]  # First component of x
# x2 = x[1, :]  # Second component of x

# Create a meshgrid for x1 and x2
x1_grid, x2_grid = np.meshgrid(x1, x2)

# Reshape y to match the shape of x1_grid and x2_grid
y_grid = y.reshape(x1_grid.shape)

# Plot the 3D surface
surface = ax.plot_surface(x1_grid, x2_grid, y_grid, cmap='viridis')

# Set labels for the axes
ax.set_xlabel('X1')
ax.set_ylabel('X2')
ax.set_zlabel('Y')

# Add a colorbar to indicate the values of the surface
fig.colorbar(surface, shrink=0.5, aspect=10)

# Show the plot
plt.show()

#%%

X = np.linspace(x[0],x[-1],100)
Y = X * w[1] + w[0]
plt.plot(X,Y,c='blue')
plt.show()

#%%

# import autograd-wrapped numpy
import numpy as np
w = np.array([3,3]) # w=1, w = 4
[w, cost_h1] = gradient_descent(0.01,10,w)
[w, cost_h2] = gradient_descent(0.1,10,w)
[w, cost_h3] = gradient_descent(1,10,w)
print(w)
print(cost_h)
1
2
3
4
5
6
7
8
9
8/31/23, 12:14 AM Cap5_Gradiente_Descent_ex02 - Jupyter Notebook
localhost:8888/notebooks/Dropbox/MAT2470-MAchineLearning_2023/Python/Cap5/Cap5_Gradiente_Descent_ex02.ipynb 6/7
In [72]:
import matplotlib.pyplot as plt
plt.figure(figsize = (16,8))
x = np.linspace(-4,4,100)
y = np.linspace(-4,4,100)
X,Y = np.meshgrid(x,y)
Z = g(X,Y)
# the function, 
ax = plt.axes(projection='3d')
#ax.plot_surface(X, Y, Z, rstride=1, cstride=1,
# cmap='viridis', edgecolor='none')
ax.contour3D(X, Y, Z, 50, cmap='binary')
#ax.view_init(90, 0)
plt.show()
plt.figure(figsize = (16,8))
plt.plot(cost_h1,c='r')
plt.plot(cost_h2,c='b')
plt.plot(cost_h3,c='g')
plt.plot(cost_h4,c='y')
plt.show()
1
2
3
4
5
6
7
8
9
10
11
12
13
14
15
16
17
18
19
20
21
8/31/23, 12:14 AM Cap5_Gradiente_Descent_ex02 - Jupyter Notebook
localhost:8888/notebooks/Dropbox/MAT2470-MAchineLearning_2023/Python/Cap5/Cap5_Gradiente_Descent_ex02.ipynb 7/7
In [ ]:
In [ ]:
1
1