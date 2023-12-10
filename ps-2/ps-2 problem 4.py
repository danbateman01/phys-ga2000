import numpy as np

def quadratic1(a,b,c):

    x1 = (-b + np.sqrt(b**2-(4*a*c)))/(2*a)
    x2 = (-b - np.sqrt(b**2-(4*a*c)))/(2*a)

    return x1,x2
solve1=quadratic1(0.001,1000,0.001)

def quadratic2(a,b,c):
    
    x3 = (2*c)/(-b-np.sqrt(b**2-4*a*c))
    x4 = (2*c)/(-b+np.sqrt(b**2-4*a*c))

    return x3,x4
solve2=quadratic2(0.001,1000,0.001)


print("Solutions using method 1=", solve1[0], solve1[1])
print("Solutions using method 2=", solve2[0], solve2[1])
