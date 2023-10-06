#%%

import numpy as np
import matplotlib.pyplot as plt

#%%

# Primeiros exemplos
# a least squares function for linear regression 

def model(x_p,w):
    # compute linear combination and return 
    a = w[0] + np.dot(x_p,w[1:])
    return a

def g(x,y):
    # return x**2+y**2+2*(np.sin((1.5*(x+y))))**2 + 2
    return 1 - x**2 - y**2 + 2*x + 4*y


def grad_w(x,y):
#    grad = np.array([2*x+6*np.sin((1.5*(x+y)))*np.cos((1.5*(x+y))),
#                     2*y+6*np.sin((1.5*(x+y)))*np.cos((1.5*(x+y)))])
   grad = np.array([-2*x + 2,
                    -2*y + 4])
   return grad


def gradient_descent(alpha, max_its, w):
    # gradient descent loop 
    cost_history = [g(w[0],w[1])]
    for k in range(max_its):
        # evaluate the gradient
        grad_eval = grad_w(w[0],w[1])
        # print('grad = ', grad_eval)
        # take gradient descent step
        # print([1,g(w[0],w[1])])
        custo1 = g(w[0],w[1])
        w = w - alpha*grad_eval
        # print([2,g(w[0],w[1])])
        custo2 = g(w[0],w[1])
        if custo2 > custo1:
            w = w + alpha*grad_eval
            alpha = alpha/2
            # print('retornou w = ',w)
        cost_history.append(g(w[0],w[1])) 
    # print(custo2)
    return w, cost_history


#%%

# import autograd-wrapped numpy
w = np.array([3,3])
dict_costs = dict()
l = [-3, 0,3]
w_combinations = [(l[i], l[j]) for i in range(len(l)) for j in range(len(l))]

iterations = 50
for alpha in [0.001, 0.01, 0.1, 1, 2, 10]:
    for comb in w_combinations:
        w = np.array(comb)
        title = f"w={tuple(w)}; alpha={alpha}"
        response = gradient_descent(alpha,iterations,w)
        print(response)
        dict_costs[title] = response

# [w, cost_h1] = gradient_descent(0.01,10,w)
# [w, cost_h2] = gradient_descent(0.1,10,w)
# [w, cost_h3] = gradient_descent(1,10,w)
# print(w)
# print(cost_h)




#%%
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


# Create a figure and axis
fig, ax = plt.subplots()

# Plot each line with its legend
for key, val in dict_costs.items():
    ax.plot(val[1], label=key)

# Add labels and a legend
ax.set_xlabel('X-axis')
ax.set_ylabel('Y-axis')
# ax.legend()

# Show the plot
plt.show()

#%%

for key, val in dict_costs.items():
    print(key, val[0], val[1][-1])