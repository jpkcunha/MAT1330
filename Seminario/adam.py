#%%

import numpy as np
import matplotlib.pyplot as plt

#%%


def model(x_p: np.array,
          w: np.array) -> float:
    return w[0] + np.dot(x_p,w[1:,])

def g(w: np.array,
      x: np.array,
      y: np.array,) -> float:
    cost = 0
    P = y.size
    for p in range(P):
        cost += (model(x[p],w) - y[p]) ** 2
    cost /= P
    # print(f"c = {cost}")
    return cost



def gradient(w,X,Y) -> np.array:
    N = 2
    P = np.size(X)
    grad = np.zeros(N)
    for p in range(P):
        grad = grad + (model(X[p],w) - Y[p]) * [1,X[p]]
    grad = 2 * grad/P
    return np.array(grad)


def gradient_desc(w: np.array,
                  X: np.array,
                  Y: np.array,
                  max_its: int = 1000,
                  alpha: float = 10,
                  tolerance: float = 1e-7) -> np.array: 
    result = w.copy()
    cost = g(result,X,Y) # Custo da estimativa inicial
    for k in range(max_its):
        result_ant = result
        grad_eval = gradient(result,X,Y)
        result = result - alpha * grad_eval
        cost_ant = cost
        cost = g(result,X,Y)
        if abs(cost) < tolerance: break
        if cost > cost_ant:
            result = result_ant # Volta passo
            alpha = alpha/2
            # print('alpha=',alpha)
    print(f"Fim do gradiente => w={result}, c={cost}, {k+1} iteracoes")
    return result


def adam(w: np.array,
         X: np.array,
         Y: np.array,
         beta1: float = 0.9,
         beta2: float = 0.999,
         max_its: int = 1000,
         alpha: float = 10,
         tolerance: float = 1e-7) -> np.array:
    
    result = w.copy()
    cost = g(result,X,Y) # Custo da estimativa inicial

    if beta1 < 0 or beta1 > 1: beta1 = 0.9
    if beta2 < 0 or beta2 > 1: beta2 = 0.999
    d, h = 0, 0
    for k in range(max_its):
        result_ant = result
        d_ant = d
        h_ant = h
        grad_eval = gradient(result,X,Y)

        # Valor de d_{k-1} e h_{k-1}
        if k == 0:
            d, h = grad_eval, grad_eval**2
        else:
            d = beta1*d_ant + (1-beta1)*grad_eval
            h = beta2*h_ant + (1-beta2)*grad_eval**2

        # Computa novo passo com exponential average e normalização
        cost_ant = cost
        result = result - alpha * d/(h**0.5)
        cost = g(result,X,Y)

        if abs(cost) < tolerance: break
        if cost > cost_ant:
            result = result_ant # Volta passo
            alpha = alpha/2
            # print('alpha=',alpha)
    print(f"Fim do gradiente => w={result}, c={cost}, {k+1} iteracoes")
    return result


def denormalize(vector: np.array):
    vector[1] = vector[1] / 10
    vector[0] = vector[0] - 2004 * vector[1]
    print(vector)

#%%


filename = "https://raw.githubusercontent.com/jermwatt/machine_learning_refined/602412f222afe4d5497472037b0e62002d5a1d65/exercises/ed_2/mlrefined_datasets/superlearn_datasets/student_debt.csv"
data = np.loadtxt(filename,delimiter=',')
X,Y = data
print(data)
X = (X-2004)/10
print(data)

P = X.size
A = np.array([np.ones(P),X])
A = A.transpose()
M = np.matmul(A.transpose(),A)
b = np.matmul(A.transpose(),Y)
v = np.linalg.solve(M,b)

w_inicial = np.array([3,2])
results = {}
results['MSE solution'] = v
results['Gradient Descent (1000)'] = gradient_desc(w_inicial,X,Y,1000,10, 1e-6)
results['Adam (0.9, 0.999, 1000)'] = adam(w_inicial,X,Y,0.9, 0.999, 1000, 10, 1e-6)
results['Adam (0.9, 0.999, 2000)'] = adam(w_inicial,X,Y,0.9, 0.999, 2000, 10, 1e-6)
results['Adam (0.9, 0.999, 5000)'] = adam(w_inicial,X,Y,0.9, 0.999, 5000, 10, 1e-6)
results['Adam (0.9, 0.999, 10000)'] = adam(w_inicial,X,Y,0.9, 0.999, 10000, 10, 1e-6)



# %%

X = 10*X+2004
plt.figure(figsize = (16, 8))
plt.scatter(X, Y, c = 'red')
x = np.linspace(X[0], X[-1], 100)
for legend, result in results.items():
    denormalize(result)
    y = result[1] * x + result[0]
    plt.plot(x, y, label=legend, linewidth=3)
# plt.plot(x, yv, 'g')
plt.legend()
plt.show()

# %%
