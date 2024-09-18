import numpy as np
import matplotlib.pyplot as plt
 
def mandel(x, y, N):
    c = complex(x, y)
    z = 0
    for i in range(1, N):
        if abs(z) > 2:
            return False
        z = z**2 + c
    return True



N = 1000
x = np.linspace(-2,2, N,dtype=np.float32)
y = np.linspace(-2,2, N,dtype=np.float32)
xpoints = np.zeros(N**2)
ypoints = np.zeros(N**2)  

i=0
for a in x:
    for b in y:
       d=mandel(a,b,N)
       if d:
           xpoints[i] = a
           ypoints[i] = b
           i=i+1
        
plt.scatter(xpoints, ypoints, color='black', s=.1)
plt.xlabel("Real Part")
plt.ylabel("Imaginary part")
plt.title("Mandelbrot Set")
plt.savefig("mandelbrot.png")
plt.show()
