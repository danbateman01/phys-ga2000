import numpy as np 
import matplotlib.pyplot as plt 

rho = 1.22
c=0.47
r= 0.08
const = 9.81*np.pi*r**2 * rho *c /2
def f1(coeff, t):
    m=1
    A= const/m
    x = coeff[0]
    vx= coeff[1]
    y = coeff[2]
    vy= coeff[3]
    fx = vx 
    fvx = -A*vx*np.sqrt (vx**2+vy**2)
    fy = vy 
    fvy = -1 -A*vy*np.sqrt(vx**2+vy**2)
    return np.array([fx, fvx,fy,fvy],float)

def f2(coeff, t):
    m=2
    A= const/m
    x = coeff [0]
    vx= coeff [1]
    y = coeff [2]
    vy= coeff [3]
    fx = vx 
    fvx = -A*vx*np.sqrt (vx**2+vy**2)
    fy = vy
    fvy = -1 -A*vy*np.sqrt(vx**2+vy**2)
    return np.array ([fx, fvx, fy,fvy], float)

def f3(coeff, t):
    m=3
    A= const/m
    x = coeff [0]
    vx= coeff [1]
    y = coeff [2]
    vy= coeff [3]
    fx = vx 
    fvx = -A*vx*np.sqrt (vx**2+vy**2)
    fy = vy 
    fvy = -1 -A*vy*np.sqrt(vx**2+vy**2)
    return np.array ([fx, fvx, fy,fvy], float)

def f4(coeff, t):
    m=4
    A= const/m
    x = coeff [0]
    vx= coeff [1]
    y = coeff [2]
    vy= coeff [3]
    fx = vx
    fvx = -A*vx*np.sqrt (vx**2+vy**2)
    fy = vy 
    fvy = -1 -A*vy*np.sqrt(vx**2+vy**2)
    return np.array ([fx, fvx, fy,fvy], float)


def rk4_step(func, x, t, h):
    k1 = h*func(x, t)
    k2= h*func(x+0.5*k1,t+0.5*h)
    k3= h*func(x+0.5*k2,t+0.5*h)
    k4 = h*func(x+k3, t+h)
    return x + (k1+2*k2+2*k3+k4)/6



def rungekutta(func, x0,vx0, y0,vy0, h, t):
    tpoints = np.arange(0, t,h)
    xlist =[]
    vxlist =[]
    ylist = []
    vylist = []
    coeff = np.array ([x0, vx0,y0, vy0], float)
    for t in tpoints: 
        if coeff[2] < 0: 
            break
        xlist.append (coeff [0])
        vxlist.append (coeff [1])
        ylist.append (coeff[2])
        vylist.append (coeff[3])
        coeff = rk4_step(func,coeff, t, h)
    return tpoints, xlist, vxlist, ylist, vylist

tpoints, xlist, vxlist, ylist, vylist = rungekutta(f1, 0, 100*np.cos(np.pi/6),0, 100*np.sin(np.pi/6), 0.01, 1000)
plt. plot(xlist,ylist, label='m = 1')
tpoints, xlist, vxlist, ylist, vylist = rungekutta(f2, 0, 100*np.cos(np.pi/6),0, 100*np.sin(np.pi/6), 0.01, 1000)
plt.plot(xlist,ylist, label = 'm = 2')
tpoints, xlist, vxlist,ylist,vylist = rungekutta(f3, 0, 100*np.cos(np.pi/6),0, 100*np.sin(np.pi/6), 0.01, 1000)
plt.plot(xlist,ylist, label = 'm = 3')
tpoints, xlist, vxlist, ylist, vylist = rungekutta(f4, 0, 100*np.cos(np.pi/6),0, 100*np.sin(np.pi/6), 0.01, 1000)
plt.plot(xlist,ylist, label= 'm = 4')
plt.xlabel('Distance (m)')
plt.ylabel('Height (m)')
plt.legend()
plt.show()


