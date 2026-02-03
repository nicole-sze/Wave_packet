#! /usr/bin/env python3

import math as maths
import matplotlib.pyplot as plt
import matplotlib.image
import numpy as np
import cmath
from matplotlib import cm

xvals = np.arange(-10,10,0.1)
yvals = np.arange(-10,10,0.1)
xvals, yvals = np.meshgrid(xvals, yvals)
t = [0,0.25,0.5,0.75]
a = 1

def psix(x,t,a):
    return np.abs((2*a/np.pi)**(1/4)*1/np.sqrt(1+2j*a*t)*np.exp(-a*x**2/(1+2j*a*t)))**2

def psi3d(x,y,t,a):
    return (np.abs((2*a/np.pi)**(1/4)*1/np.sqrt(1+2j*a*t)*np.exp(-a*x**2/(np.sqrt(1+2j*a*t))))**2) * (np.abs((2*a/np.pi)**(1/4)*1/np.sqrt(1+2j*a*t)*np.exp(-a*y**2/(np.sqrt(1+2j*a*t))))**2)

for counter in range(len(t)):
    plt.subplot(2,2,counter+1) 
    tn = t[counter]
    plt.plot(xvals, psix(xvals,tn,a))
    plt.title("t = "+str(tn))
    plt.xlabel("x")
    plt.ylabel("Psi squared")
    plt.ylim([0,1])

plt.suptitle("Stationary Wave")
plt.tight_layout()
plt.show()
plt.savefig("1D_stationary.pdf")

fig = plt.figure()

for counter2 in range(len(t)):
    ax = fig.add_subplot(2, 2, counter2+1, projection='3d')
    tn = t[counter2]
    ax.plot_surface(xvals, yvals, psi3d(xvals,yvals,t[counter2],a), cmap=cm.coolwarm, linewidth=0)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    ax.set_zlim(0, 1)

plt.suptitle("2D Stationary Wave")
plt.show()
plt.savefig("2D_stationary.pdf")
plt.show()