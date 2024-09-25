#10.4 in Newman

from random import random
import numpy as np
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.pyplot as plt

N = 1000
T = 3.053*60

decay = (-1/(np.log(2)/T)*np.log(1-np.random.random(N)))
decay = np.sort(decay)
decayed = np.arange(1,N+1)
remaining = -decayed + N

plt.plot(decay,remaining,color='purple')
plt.xlabel('Time (s)')
plt.ylabel('Undecayed Atoms')
plt.title('Decay of Tl208')
plt.savefig('Decay of Tl208.png')
plt.show()
