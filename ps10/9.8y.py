import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation  
from pylab import *

sigma = 10**(-10)
K = 5 * 10**10
L = 10**(-8)
x0 = L/2
N=1000
a = L/N
h = 10**(-18)
hbar = 1*10**(-34)
m = 9.109 * 10**(-31)


def banded(Aa,va,up,down):
    A = copy(Aa)
    v = copy(va)
    N = len(v)
    for m in range(N):
        div = A[up,m]
        v[m] /= div
        for k in range(1,down+1):
            if m+k<N:
                v[m+k] -= A[up+k,m]*v[m]
        for i in range(up):
            j = m + up - i
            if j<N:
                A[i,j] /= div
                for k in range(1,down+1):
                    A[i+k,j] -= A[up+k,m]*A[i,j]
    for m in range(N-2,-1,-1):
        for i in range(up):
            j = m + up - i
            if j<N:
                v[m] -= A[i,j]*v[j]
    return v




a1 = 1+h*1j*hbar/(2 * m * a**2)
a2 = -h*1j*hbar/(4 * m * a**2)
b1 = 1-h*1j*hbar/(2 * m * a**2)
b2 = h*1j*hbar/(4 * m * a**2)

def psi0(x):
    return np.exp(- (x-x0)**2/(2 * sigma**2))*np.exp(1j*K*x)
psi = psi0((1+np.arange(0, N))*a)

def V(i, psi):
    return b1 * psi[i]+ b2*(np.take(psi, i+1, mode='clip') + np.take(psi, i-1, mode='clip'))

xpoints = np.linspace(0,L,N)

bands = np.zeros((3, N), dtype=complex)
bands[0] = a2
bands[1] = a1
bands[2] = a2

T = 5000
psis = np.zeros((T, N))

for i in range(T):
    v = (V(np.arange(0, N), psi))
    psi = banded(bands, v, 1, 1)
    psis[i] = psi

fig,ax=plt.subplots()
line, = ax.plot(xpoints, psis[0])
ax.set_xlabel('$x$ (m)')
ax.set_ylabel('$\psi(x)$  $(1/\sqrt{m})$')

def frame(i):
   line.set_data(xpoints, psis[i])
   return line,

ani = FuncAnimation(fig, frame, frames = T, interval=10, blit=True)
plt.show()

