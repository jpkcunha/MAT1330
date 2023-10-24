#%%

# import numpy as np
# import autograd-wrapped numpy
import autograd.numpy as np
import matplotlib.pyplot as plt

#%%

def pontos_2d(w,P):
    np.random.seed(30)
    # Gera um conjunto de pontos aleatorios no plano
    X = [np.ones(P),
         np.random.random_sample(P,),
         np.random.random_sample((P,))]
    X = np.array(X)
    Y = []
    # w representa os coeficientes de uma reta escolhida ao acaso.
    # Esta reta separa os pontos em dois conjuntos, acima e abaixo da reta.
    for i in range(P):
        if X[2][i]*w[2] + X[1][i]*w[1] + X[0][i]* w[0] < 0:
            Y.append(1)
        else:
            Y.append(0)
    return X,Y


def sigma(x_p: np.array,
          w: np.array) -> float:
    return 1/(1+np.exp(-np.dot(x_p,w)))

def g(w: np.array,
      X: np.array,
      Y: np.array,) -> float:
    cost = 0
    P = len(X[0,:])
    N = len(X[:,0])
    for p in range(P):
        x, y = X[:,p], Y[p]
        cost += (sigma(w,x) - y)**2 # 1-y[p] pata SVM
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
        e = np.exp(-np.dot(x,w))
        grad += (sigma(w,x) - y)*e/(1+e)**2 * x
    grad = 2 * grad/P
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
        result = result - alpha * grad_eval
        cost_ant = cost
        cost = g(result,X,Y)
        # if abs(cost) < tolerance: break
        if cost > cost_ant:
            result = result_ant # Volta passo
            alpha = alpha/2
            # print('alpha=',alpha)
    print(f"Fim do gradiente => w={result}, c={cost}, {k+1} iteracoes")
    return result

#%%

P = 100
# Reta escolhida para classificar os pontos iniciais
w0 = np.array([-0.3,-0.8,1])
# Geracao dos pontos
[X,Y] = pontos_2d(w0,P)
X,Y

N = 3
w = np.ones(N)

max_its, alpha = 1000,10
w = gradient_descent(w,X,Y,max_its,alpha)

# Plota os pontos e a reta obtida pela regressao
plt.figure(figsize = (8,8))
for i in range(P):
    if Y[i] > 0:
        plt.scatter(X[1,i],X[2,i],c='b')
    else:
        plt.scatter(X[1,i],X[2,i],c='r')
x = np.linspace(0,1,100)
y = - (w[1]*x+w[0])/w[2]
plt.plot(x,y,'g')
plt.show()




#%%

# load in data 
datapath = 'https://raw.githubusercontent.com/jermwatt/machine_learning_refined/602412f222afe4d5497472037b0e62002d5a1d65/exercises/ed_2/mlrefined_datasets/superlearn_datasets/'
csvname = datapath + '2d_classification_data_v1_entropy.csv'
data = np.loadtxt(csvname,delimiter = ',')

# get input/output pairs
x = data[:-1,:]
y = data[-1:,:] 

print(np.shape(x))
print(np.shape(y))

plt.scatter(x,y)
plt.show()








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
