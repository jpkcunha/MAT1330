#%%

# import numpy as np
# import autograd-wrapped numpy
import autograd.numpy as np
import matplotlib.pyplot as plt

#%%

def pontos_1d(x,y):
    np.random.seed(30)
    # Gera um conjunto de pontos aleatorios no plano
    P = len(x)
    X = [np.ones(P),x]
    X = np.array(X)
    Y = list(y)
    return X,Y

def normalize(v: np.array) -> float:
    squared_sum = np.sum(v ** 2)
    norm = np.sqrt(squared_sum)
    return norm



# Função custo
def g(w: np.array,
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


def gradient(w: np.array,
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



def gradient_descent(w: np.array,
                     X: np.array,
                     Y: np.array,
                     max_its: int = 1000,
                     alpha: float = 10) -> np.array: 
    result = w.copy()
    cost = g(result,X,Y) # Custo da estimativa inicial
    for k in range(max_its):
        result_ant = result
        grad_eval = gradient(result,X,Y)
        norm = normalize(grad_eval)
        result = result - alpha * grad_eval / norm
        cost_ant = cost
        cost = g(result,X,Y)
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
csvname = '2d_classification_data_v1.csv'
data = np.loadtxt(datapath + csvname,delimiter = ',')

# get input/output pairs
x = data[:-1,:]
y = data[-1:,:]
x,y = x[0], y[0]
X,Y = pontos_1d(x,y)
print(np.shape(X))
print(np.shape(Y))
X

#%%

N=2
ALPHA = 10
w_inicial = np.ones(N)

results = {}
for max_its in [100,1000,10000]:
    results[f'{max_its} passos'] = gradient_descent(w_inicial,X,Y,max_its,ALPHA)




#%%


colors = ['b' if el>0.5 else 'r' for el in y ]
plt.scatter(x,y, c = colors)

x = np.linspace(0,5,100)
# x = np.linspace(X[0], X[-1], 100)
for legend, result in results.items():
    # y = 2/(1+np.exp(-(result[0]+result[1]*x))) - 1
    y = np.tanh(result[0]+result[1]*x)
    plt.plot(x, y, label=legend, linewidth=2)
plt.legend()
plt.show()

#%%

