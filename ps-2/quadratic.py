import numpy as np

def quadratic(a,b,c):
    
    x3 = (2*c)/(-b-np.sqrt(b**2-4*a*c))
    x4 = (2*c)/(-b+np.sqrt(b**2-4*a*c))

    return x3,x4

