import numpy as np
import matplotlib.pyplot as plt
def is_float(string):
    try:
        return float(string)
    except ValueError:
        return False

data = []
with open('signal.dat', 'r') as f:
    d = f.readlines()
    for i in d:
        k = i.rstrip().split('|')
        for i in k:
            if is_float(i):
                data.append(float(i)) 

data = np.array(data, dtype='float')
time = data[::2]/np.max(data[::2])
signal = data[1::2]

plt.scatter(time, signal)
plt.xlabel('Time')
plt.ylabel('Signal')
plt.title('Raw Data')
#plt.savefig('signaldat.png')
plt.show()

#B

order = 36
A = np.zeros((len(time), order))
for i in range(order):
    A[:,i]=time**i
    

(u, w, vt) = np.linalg.svd(A, full_matrices=False)
Ainv = vt.transpose().dot(np.diag(1./w)).dot(u.transpose())
C = Ainv.dot(signal)
newA = A.dot(C)


plt.scatter(time,signal,label = 'Raw Data',color = 'green')
plt.scatter(time,newA,label = '3rd order fit Curve', color= 'black' )
plt.xlabel('Time (s)')
plt.ylabel('Signal')
plt.legend(['3rd order fit Curve', 'Raw Data'])
plt.title('SVD Technique finding the fitted curve')
#plt.savefig('3b signal with svd order=3.png')
plt.show()


#c

residuals = newA - signal
plt.plot(time, residuals, '.')
plt.xlabel('time')
plt.ylabel('Residuals')
#plt.savefig("3c residuals order 10.png")
plt.show()

#e



