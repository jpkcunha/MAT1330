#%%

# import numpy as np
# import autograd-wrapped numpy
import autograd.numpy as np
import matplotlib.pyplot as plt

#%%

def pontos_8d(x,y):
    np.random.seed(30)
    # Gera um conjunto de pontos aleatorios no plano
    P = x.shape[-1]
    X = [np.ones(P)]
    for el in x:
        X.append(el)
    X = np.array(X)
    Y = list(y)
    return X,Y

def normalize(v: np.array) -> float:
    squared_sum = np.sum(v ** 2)
    norm = np.sqrt(squared_sum)
    return norm



### SOFTMAX
def g_softmax(w: np.array,
              X: np.array,
              Y: np.array,) -> float:
    cost = 0
    P = len(X[0,:])
    for p in range(P):
        x, y = X[:,p], Y[p]
        e = np.exp(-y*np.dot(x,w))
        cost += np.log(1+e)
    cost /= P
    print(f"c = {cost}")
    return cost

def grad_softmax(w: np.array,
                 X: np.array,
                 Y: np.array,) -> np.array:
    P = len(X[0,:])
    N = len(X[:,0])
    grad = np.zeros(N)
    for p in range(P):
        x, y = X[:,p], Y[p]
        e = np.exp(-y*np.dot(x,w))
        grad += e/(1+e) * y * x
    grad /= -P
    return np.array(grad)


### PERCEPTRON
def g_perceptron(w: np.array,
                 X: np.array,
                 Y: np.array,) -> float:
    cost = 0
    P = len(X[0,:])
    for p in range(P):
        x, y = X[:,p], Y[p]
        cost += max(0,-y*np.dot(x,w))
    cost /= P
    print(f"c = {cost}")
    return cost

def grad_perceptron(w: np.array,
                    X: np.array,
                    Y: np.array,) -> np.array:
    P = len(X[0,:])
    N = len(X[:,0])
    grad = np.zeros(N)
    for p in range(P):
        x, y = X[:,p], Y[p]
        k = 0
        if -y*np.dot(x,w) >= 0:
            k = -y
        grad += k * x
    grad /= P
    return np.array(grad)




def gradient_descent_softmax(w: np.array,
                             X: np.array,
                             Y: np.array,
                             max_its: int = 1000,
                             alpha: float = 10) -> np.array: 
    result = w.copy()
    cost = g_softmax(result,X,Y) # Custo da estimativa inicial
    for k in range(max_its):
        result_ant = result
        grad_eval = grad_softmax(result,X,Y)
        norm = normalize(grad_eval)
        result = result - alpha * grad_eval / norm
        cost_ant = cost
        cost = g_softmax(result,X,Y)
        # if abs(cost) < tolerance: break
        if cost > cost_ant:
            result = result_ant # Volta passo
            alpha = alpha/2
            # print('alpha=',alpha)
    print(f"===== Fim do gradiente =====\nw={result}, c={cost}, {k+1} iteracoes")
    return result



def gradient_descent_perceptron(w: np.array,
                                X: np.array,
                                Y: np.array,
                                max_its: int = 1000,
                                alpha: float = 10) -> np.array: 
    result = w.copy()
    cost = g_perceptron(result,X,Y) # Custo da estimativa inicial
    for k in range(max_its):
        result_ant = result
        grad_eval = grad_perceptron(result,X,Y)
        norm = normalize(grad_eval)
        result = result - alpha * grad_eval / norm
        cost_ant = cost
        cost = g_perceptron(result,X,Y)
        # if abs(cost) < tolerance: break
        if cost > cost_ant:
            result = result_ant # Volta passo
            alpha = alpha/2
            # print('alpha=',alpha)
    print(f"===== Fim do gradiente =====\nw={result}, c={cost}, {k+1} iteracoes")
    return result



#%%

# load in data 
datapath = 'https://raw.githubusercontent.com/jermwatt/machine_learning_refined/602412f222afe4d5497472037b0e62002d5a1d65/exercises/ed_2/mlrefined_datasets/superlearn_datasets/'
csvname = 'breast_cancer_data.csv'
data = np.loadtxt(datapath + csvname,delimiter = ',')

# get input/output pairs
x = data[:-1,:]
y = data[-1:,:][0]
X,Y = pontos_8d(x,y)
print(np.shape(X))
print(np.shape(Y))
X

#%%

N=9
max_its, alpha = 10000, 10
w_inicial = np.ones(N)

gradient_descent_softmax(w_inicial,X,Y,max_its,alpha)
gradient_descent_perceptron(w_inicial,X,Y,max_its,alpha)




#%%


# colors = ['b' if el>0.5 else 'r' for el in y ]
# plt.scatter(x,y, c = colors)

# x = np.linspace(0,5,100)
# # x = np.linspace(X[0], X[-1], 100)
# for legend, result in results.items():
#     y = 2/(1+np.exp(-(result[0]+result[1]*x))) - 1
#     plt.plot(x, y, label=legend, linewidth=2)
# plt.legend()
# plt.show()

# #%%

