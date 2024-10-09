#Newman 5.17
import numpy as np
import matplotlib.pyplot as plt
from scipy import integrate
import scipy as sc

def integrand(x,a):

    return x**(a-1)*np.exp(-x)

x = np.linspace(0,5,1000)
a = [2, 3, 4]

for i in range(len(a)):
    plt.plot(x, integrand(x,a[i]))
    plt.legend(['a = 2', 'a = 3', 'a = 4'])
    plt.xlabel('x', fontsize=16)
    plt.ylabel('Integrand', fontsize=16)
    plt.grid()
    plt.savefig('Integrand for a = 2, 3, 4')

#E
def transform(x,a):
    return (a-1)*x/(1-x)
def newintegrand(x,a):
    return np.exp((a-1)*np.log(x)-x)

# Gaussian quadrature
def gamma(a):
    gam = lambda x: newintegrand(transform(x,a), a)*(a-1)/((1-x)**2)
    (s, none) = integrate.fixed_quad(gam, 0, 1, n=100)

    return s

print('For a=3/2 the gamma function is', gamma(3/2))
print('For a=3 the gamma function is', gamma(3))
print('For a=6 the gamma function is', gamma(6))
print('For a=10 the gamma function is', gamma(10))