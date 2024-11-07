import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft, fftfreq

dow = np.loadtxt('dow.txt')

N = len(dow)
t = np.linspace(0, 10, 1000)
T = t[1]-t[0]
x = np.arange(1, N+1)
#xf = fftfreq(N, T)[:N//2]



fft_coeffs = np.fft.rfft(dow)
Ncoeffs=len(fft_coeffs)
last90=int(Ncoeffs*1)
fft_coeffs[last90:]=0


inversefft = np.fft.irfft(fft_coeffs)

plt.plot(x, dow, label ='Raw Data')



plt.plot(x,inversefft, label ='Inverse FFT', color= 'black')
plt.xlabel('Time')
plt.ylabel('Price')
plt.title('Dow Jones 2006 - 2010')
plt.legend()
plt.grid(True)
plt.figure(figsize=(15,15))
plt.show()