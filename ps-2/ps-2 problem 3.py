from math import sqrt
import numpy as np
import time

startL = time.time()
def madelung(L):

    M = 0.0
    for i in range(-L,L+1):
        for j in range(-L,L+1):
            for k in range(-L,L+1):
                if i==j==k==0:
                    continue
                M += (-1)**(i+j+k)/sqrt((i**2+j**2+k**2))
    
    return(M)

M1 = madelung(100)

print(("M1="),M1)
endL = time.time()
TimeL=endL - startL
print("Time taken with for loop=",TimeL)



startM = time.time()
def madelungmesh(L):

    a = np.arange(-L,L+1, dtype=float)

    i,j,k = np.meshgrid(a,a,a)
    M2 =(-1)**(i+j+k)/np.sqrt(i**2+j**2+k**2)
    M2[(i == 0)*(j == 0)*(k == 0)] = 0
    M2=np.sum(M2)
    
    return(M2)


M2 = madelungmesh(100)

print("M2=",M2)

endM = time.time()
TimeM=endM - startM
print("Time taken with meshgrid=",TimeM)

TimeD=abs(TimeL-TimeM)
print("Time Difference=",TimeD)
