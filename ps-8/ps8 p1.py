

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import optimize

data = pd.read_csv('survey.csv')
xs = data['age'].to_numpy()
ys = data['recognized_it'].to_numpy()
x_sort = np.argsort(xs)
age = xs[x_sort]
recognize = ys[x_sort]

def p(x, beta0, beta1): return 1/(1+np.exp(-(beta0+beta1*x)))


def log_likelihood(beta, xs, ys):
    beta_0 = beta[0]
    beta_1 = beta[1]
    epsilon = 1e-16
    l_list = [ys[i]*np.log(p(xs[i], beta_0, beta_1)/(1-p(xs[i], beta_0, beta_1)+epsilon)) 
              + np.log(1-p(xs[i], beta_0, beta_1)+epsilon) for i in range(len(xs))]
    ll = np.sum(np.array(l_list), axis = -1)
    return -ll 

pstart = [1,42]

def Covariance(hess_inv, resVariance):
    return hess_inv * resVariance


def error(hess_inv, resVariance):
    covariance = Covariance(hess_inv, resVariance)
    return np.sqrt( np.diag( covariance ))


b = np.linspace(-4, 4, 100)
result = optimize.minimize(lambda b, age, recognize: log_likelihood(b, age, recognize), [0,0], args=(age, recognize))
hess_inv = result.hess_inv 
var = result.fun/(len(recognize)-len(pstart)) 
dFit = error( hess_inv,  var)
print('Optimal parameters and error:\n\tp: ' , result.x, '\n\tdp: ', dFit)
print('Covariance matrix of optimal parameters:\n\tC: ' , Covariance( hess_inv,  var))


plt.plot(age, p(age, result.x[0], result.x[1]), label='Logistic Function',color='pink',linewidth='3')
plt.plot(age, recognize, 'o', label='Data',color='green')
plt.title('Probability of Hearing Phrase')
plt.xlabel('Age (years)')
plt.ylabel('Probabilty')
plt.legend()
plt.show()