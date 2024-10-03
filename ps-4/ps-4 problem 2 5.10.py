#Newman 5.10
import numpy as np
import matplotlib.pyplot as plt

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



def f(x, b):
  return (b**4-x**4)**(-1/2)

N = 20
a = 0
p = []
amp = np.linspace(0, 2, 100)
for A in amp:
    b = A
    x, w = gaussxw(N)
    xp = 0.5*(b-a)*x + 0.5*(b-a)
    wp = 0.5*(b-a)*w
    
    s = 0.0
    for i in range(N):
        s+=wp[i]*f(xp[i], b)
    
    p.append(s*np.sqrt(8))
plt.plot(amp, p)
plt.xlabel("Amplitude")
plt.ylabel("Period")
plt.title("Period for Different Amplitudes")
#plt.savefig("periodamplitude")
plt.show()
