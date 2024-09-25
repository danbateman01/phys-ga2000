import numpy as np
import matplotlib.pyplot as plt

N_values = [10, 50, 100, 1000]
samples = 10000 
def random(N, scale=1.0):
    x = np.random.exponential(scale, N)
    y = np.mean(x)
    return y


plt.figure(figsize=(12, 8))
for N in N_values:
    yset = [random(N) for i in range(samples)]
plt.hist(yset, bins=30, density=True, alpha=0.5, label=f'N={N}', edgecolor='black')


plt.title('Distribution of y for Different N')
plt.xlabel('Value of y')
plt.legend()
plt.grid()
plt.show()
