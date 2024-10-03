
import numpy as np
import matplotlib.pyplot as plt
import math
from scipy.special import roots_hermite


def gaussxw(N):

    a = np.linspace(3,4*N-1,N)/(4*N+2)
    x = np.cos(np.pi*a+1/(8*N*N*np.tan(a)))
    epsilon = 1e-15
    delta = 1.0
    while delta>epsilon:
        p0 = np.ones(N,float)
        p1 = np.copy(x)
        for k in range(1,N):
            p0,p1 = p1,((2*k+1)*x*p1-k*p0)/(k+1)
        dp = (N+1)*(p0-x*p1)/(1-x*x)
        dx = p1/dp
        x -= dx
        delta = max(abs(dx))

    w = 2*(N+1)*(N+1)/(N*N*(1-x*x)*dp*dp)

    return x,w

def gaussxwab(N,a,b):
    x,w = gaussxw(N)
    return 0.5*(b-a)*x+0.5*(b+a),0.5*(b-a)*w


#a
def H(n, x):
    if n==0:
        return np.ones(x.shape)
    elif n==1:
        return 2*x
    else:
        return 2*x*H(n-1, x)-2*(n-1)*H(n-2, x)

def psi(n):
    psi_list = []
    x = np.linspace(-4, 4, 1000)
    H_n = H(n, x)
    psi = 1/np.sqrt(np.power(2, n)*math.factorial(n)*np.sqrt(np.pi)) * np.exp(-1*np.power(x, 2)/2) * H_n

    return psi

plt.plot(np.linspace(-4, 4, 1000), psi(0), label = 'n = 0')
plt.plot(np.linspace(-4, 4, 1000), psi(1), label = 'n = 1')
plt.plot(np.linspace(-4, 4, 1000), psi(2), label = 'n = 2')
plt.plot(np.linspace(-4, 4, 1000), psi(3), label = 'n = 3')
plt.xlabel('Position')
plt.ylabel('Probability Density')
plt.title('Harmonic Oscillator Wave Function')
plt.legend()
plt.savefig("Harmonic Oscillator Wave Function")
plt.show()
plt.clf

#b

def psi(n):
    psi_list = []
    x = np.linspace(-10, 10, 1000)
    H_n = H(n, x)
    psi = 1/np.sqrt(np.power(2, n)*math.factorial(n)*np.sqrt(np.pi)) * np.exp(-1*np.power(x, 2)/2) * H_n

    return psi
n = 30
psi_list = psi(n)

plt.plot(np.linspace(-10, 10, 1000), psi_list)
plt.xlabel('Position')
plt.ylabel('Probability Density')
plt.title('Wave Function of Harmonic Oscillator for N = 30')
plt.savefig("p3b")
plt.show()

#c

def pos_uncertainty(n, N):
    def f(x):
        x_new = x/(1 - np.power(x, 2))
        H_n = H(n, np.array([x_new]))
        return (1 + np.power(x, 2))/np.power((1 - np.power(x, 2)),2) * np.power(x_new, 2) * np.power(np.abs(1/np.sqrt(np.power(2, n)*math.factorial(n)*np.sqrt(np.pi)) * np.exp(-1*np.power(x_new, 2)/2) * H_n[0]), 2)
    x, w = gaussxwab(N, -1, 1)
    integral = 0.0

    for k in range(N):
        integral += w[k]*f(x[k])
    return integral

ps_gq = np.sqrt(pos_uncertainty(5, 100))
print(ps_gq)

