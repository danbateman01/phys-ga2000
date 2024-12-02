
import numpy as np
import matplotlib.pyplot as plt

def harmonic(r,t,args): 
    x = r[0]
    y = r[1]
    omega = args[0]
    dx = y
    dy = -x*omega**2

    return np.array([dx,dy], float)

def anharmonic(r,t,args): 
    x = r[0]
    y = r[1]
    omega = args[0]
    dx = y
    dy = -x**3*omega**2

    return np.array([dx, dy], float)

def vanderpol(r, t, args): 
    x = r[0]
    y = r[1]
    omega = args[0]
    mu = args[1]
    dx = y
    dy = mu*(1-x**2)*y - x*omega**2

    return np.array([dx, dy], float)

def solve(f, x0, y0, args = [1], t0=1, tf=50, N=1000):

    h = (tf - t0)/N
    tpoints = np.arange(t0, tf, h)
    xpoints = []
    ypoints = []

    r = np.array([x0, y0], float)
    for t in tpoints:
        xpoints.append(r[0])
        ypoints.append(r[1])
    
        k1 = h*f(r,t, args)
        k2 = h*f(r+0.5*k1,t+0.5*h, args)
        k3 = h*f(r+0.5*k2, t+0.5*h, args)
        k4 = h*f(r+k3, t+h, args)
        r += (k1 + 2*k2 + 2*k3 + k4)/6
    
    return (tpoints, xpoints, ypoints)

# Harmonic Oscillator
sol = solve(harmonic, 1, 0)
sol1 = solve(harmonic, 2, 0)

plt.plot(sol[0],sol[1])
plt.plot(sol1[0], sol1[1])
plt.title('Harmonic Oscillator')
plt.legend(['x0 = 1', 'x0 = 2'])
plt.xlabel('t')
plt.ylabel('x')
plt.show()

# Anharmonic Oscillator
sol = solve(anharmonic, 1, 0)
sol1 = solve(anharmonic, 2, 0)

plt.plot(sol[0],sol[1])
plt.plot(sol1[0], sol1[1])
plt.title('Anharmonic Oscillator')
plt.legend(['x0 = 1', 'x0 = 2'])
plt.xlabel('time')
plt.ylabel('x')
plt.show()

# Harmonic Oscillator
sol = solve(harmonic, 1, 0)
sol1 = solve(harmonic, 2, 0)

plt.plot(sol[1],sol[2])
plt.plot(sol1[1], sol1[2])
plt.title('Harmonic Oscillator')
plt.legend(['x0 = 1', 'x0 = 2'])
plt.xlabel('x')
plt.ylabel('dx/dt')
plt.show()

# Anharmonic Oscillator
sol = solve(anharmonic, 1, 0)
sol1 = solve(anharmonic, 2, 0)

plt.plot(sol[1],sol[2])
plt.plot(sol1[1], sol1[2])
plt.title('Anharmonic Oscillator')
plt.legend(['x0 = 1', 'x0 = 2'])
plt.xlabel('x')
plt.ylabel('dx/dt')
plt.show()

# Van der Pol Oscillator
sol = solve(vanderpol, 1, 0, tf=20, args=[1, 1])
sol1 = solve(vanderpol, 1, 0, tf=20, args=[1, 2])
sol2 = solve(vanderpol, 1, 0, tf=20, args=[1, 4])


plt.plot(sol[1],sol[2])
plt.plot(sol1[1], sol1[2])
plt.plot(sol2[1], sol2[2])
plt.title('van der Pol Oscillator')
plt.legend(['$\mu = 1$', '$\mu = 2$', '$\mu = 4$'])
plt.xlabel('x')
plt.ylabel('dx/dt')
plt.show()