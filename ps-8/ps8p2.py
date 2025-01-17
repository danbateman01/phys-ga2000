
import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft, fftfreq

trumpet = np.loadtxt('trumpet.txt')
piano = np.loadtxt('piano.txt')
N = len(piano)
t = np.linspace(0, 10, 1000)
T = t[1]-t[0]# sample spacing
x = np.arange(1, N+1)
xf = fftfreq(N, T)[:N//2]
plt.plot(x, piano)
plt.xlabel('Time')
plt.ylabel('Amplitude')
plt.title('Amplitude vs.Time for a Single Note on Piano')
plt.show()

plt.plot(x, trumpet)
plt.xlabel('Time')
plt.ylabel('Amplitude')
plt.title('Amplitude vs.Time for a Single Note on Trumpet')
plt.show()
pianof = fft(piano)
trumpetf = fft(trumpet)
plt.plot(xf, 2.0/N * np.abs(pianof[0:N//2]))
plt.xlabel('Frequency [Hz]')
plt.ylabel('Magnitude')
plt.title('FFT for Piano Recording')
plt.show()

plt.plot(xf, 2.0/N * np.abs(trumpetf[0:N//2]))
plt.xlabel('Frequency [Hz]')
plt.ylabel('Magnitude')
plt.title('FFT for Trumpet Recording')
plt.show()