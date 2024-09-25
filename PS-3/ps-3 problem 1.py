import numpy as np
import math
import time
import matplotlib.pyplot as plt

a=10
b=150
c=5

timeE= []
for N in range(a,b,c):
    startE = time.time()
    C = np.zeros([N, N], float)
    A = np.zeros([N, N], float)
    B = np.zeros([N, N], float)
    for i in range(N):
        for j in range(N):
            for k in range(N):
                C[i, j] += A[i, k] * B[k, j]
    endE = time.time()
    timeE.append(endE - startE)


timeD=[]
for N in range(a,b,c):
    startD=time.time()
    C = np.zeros([N, N], float)
    A = np.zeros([N, N], float)
    B = np.zeros([N, N], float)
    np.dot(A,B)
    endD=time.time()
    timeD.append(endD-startD)
plt.figure(figsize=[8,5])
plt.plot(range(a,b,c), timeE,".-",color=('red'),label='Using For Loops')
plt.plot(range(a,b,c), timeD,'x-',color=('blue'), label="Using NumPy dot Tool")
plt.legend()
plt.title("Time Taken To Compute Matrix Multiplication Using For Loops Vs Using NumPy Dot Tool")
plt.xlabel('NxN Matrix')
plt.ylabel('Time (s)')
plt.xlim(0,b)
#plt.savefig("matrix_multiplication.png")
plt.show()




  


