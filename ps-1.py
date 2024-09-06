import numpy as np
import matplotlib.pyplot as plt


mean=0
sd=3
x_values=np.linspace(-10,10,5000)
y_values=(1/np.sqrt(2*np.pi*(sd**2)))*np.exp(-(x_values**2/(2*(sd**2))))

plt.plot(x_values,y_values,color=('blue'),)
plt.title('Normalised Gaussian Distribution')
plt.xlabel('Values')
plt.ylabel('Probability Density')
plt.text(3,0.13,"Mean = 0",size='medium',color=('black'))
plt.text(3,0.12,"Standard Deviation = 3",size='medium',color=('black'))
plt.axvline(mean,linestyle='--',color='black')
plt.show()
#plt.savefig('Gaussian.png')