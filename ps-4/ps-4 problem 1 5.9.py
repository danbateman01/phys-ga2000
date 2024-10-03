#Newman Exercise 5.9
import numpy as np
import matplotlib.pyplot as plt
N=1
V = 1000*1/1e6
rho = 6.022*1e28
theta = 428
kb = 1.38e-23
C=9*V*rho*kb
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
def cv(T, N):
    def f(x):
        return ((x**4*np.exp(x))/((np.exp(x)-1)**2))
    xp, wp = gaussxwab(N,0,theta/T)
    integral = sum(f(xp)*wp)
    return ((T/theta)**3)*integral*C
shcpoints = []
for i in np.arange(5, 500, 1):
    shc = cv(i, N)
    shcpoints.append(shc)
plt.figure()
plt.plot(np.arange(5, 500, 1),shcpoints,color=('orange'))
plt.grid()
plt.xlabel("Temp(K)")
plt.ylabel("Specific Heat Capacity")
plt.title("SHC of Aluminium vs Temperature")
#plt.savefig("schplot.png")
  


n = np.arange(1, 80)
sh = np.zeros(len(n))
for i in range(0, len(sh)):
    sh[i] = cv(100, n[i])
plt.figure()
plt.plot(n, sh)
plt.xlabel('N')
plt.ylabel('Heat Capacity')
plt.title('Convergence Test')
plt.grid()
#plt.savefig("convergancetest.png")
plt.show()

