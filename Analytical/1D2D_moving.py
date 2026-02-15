#! /usr/bin/env python3

import math as maths
import matplotlib.pyplot as plt
import matplotlib.image
import numpy as np
import cmath
from matplotlib import cm
from matplotlib.animation import FuncAnimation
from scipy.integrate import simpson

"""
    Produces static time progression images for the movement 
    of a wavepacket in free space in 1 and 2D and a gif of the 1D.
    Inputs are not taken as the parameters are standardised to be
    compared against the initial stationary code and the analytical
    equivalents.
"""


# Setting up grid values for graph
xvals = np.arange(-3,3,0.01)
yvals = np.arange(-3,3,0.01)
xvals, yvals = np.meshgrid(xvals, yvals)

# Hard coding in parameters for consistency
n = 9
t = np.arange(0,1,1/n)
a = 1
b = 1
kx0 = 1.5
ky0 = 1.5

"""
    Defining the functions
"""
# Function for the 1D graph
def psix(x,t,a,kx0):
    # Takes in the x grid, each time step, the width of the gaussian and the initial velocity
    # Returns the probability density
      return (np.abs((2*a/np.pi)**(1/4)*(1/np.sqrt(1+2j*a*t))*np.exp((-a*x**2+1j*x*kx0)/(np.sqrt(1+2j*a*t))**2+(kx0**2/(4*a))*(1/(1+2j*a*t)-1))))**2

# Function for the 2D graph
def psi3d(x,y,t,a,b,kx0,ky0):
    # Takes in the x and y grids, each time step, the width of the gaussian and the initial velocity
    # Returns the probability density
      return (np.abs((2*a/np.pi)**(1/4)*1/np.sqrt(1+2j*a*t)*np.exp((-a*x**2+1j*x*kx0)/(1+2j*a*t)+(kx0**2/(4*a))*(1/(1+2j*a*t)-1))) * (np.abs((2*b/np.pi)**(1/4)*1/np.sqrt(1+2j*b*t)*np.exp((-b*y**2+1j*y*ky0)/(1+2j*b*t)+(ky0**2/(4*b))*(1/(1+2j*b*t)-1)))))**2

"""
    1D Graph
"""
# Plotting each of the 9 time stamps for the 1D graph
for counter in range(len(t)):
      plt.subplot(int(np.sqrt(len(t))), int(np.sqrt(len(t))),counter+1) 
      tn = t[counter]
      plt.plot(xvals, psix(xvals,tn,a,kx0))
      plt.title("t = "+str(round(tn,3)))
      plt.xlabel("x")
      plt.ylabel(r"$(|\Psi|)^2$")
      plt.ylim([0,1])
      plt.grid(True, which='both')

plt.suptitle("Moving Wave")
plt.tight_layout()
plt.show()
plt.savefig("MovingWave.pdf")

"""
    2D Graph
"""
fig = plt.figure(figsize=(16,15))
# Plotting each of the 9 time stamps for the 3D graphs
for counter2 in range(len(t)):
      ax = fig.add_subplot(int(np.sqrt(len(t))), int(np.sqrt(len(t))), counter2+1, projection='3d')
      tn = t[counter2]
      ax.plot_surface(xvals, yvals, psi3d(xvals,yvals,t[counter2], a, b, kx0, ky0), cmap=cm.coolwarm, linewidth=0)
      ax.set_title("t = "+str(round(tn,3)),fontsize=20)
      ax.set_xlabel("x")
      ax.set_ylabel("y")
      ax.set_zlabel(r"$(|\Psi|)^2$")
      ax.set_zlim(0, 0.85)
      ax.grid()
      z = psi3d(xvals,yvals,t[counter2], a, b, kx0, ky0)
      prob = 0
      for i in range(len(xvals)):
            x_integral = simpson(z[i,:], x=xvals[0])
            prob += x_integral*0.01
      print(prob)

fig.suptitle("2D Moving Wave", fontsize=40)
fig.tight_layout()
fig.show()
fig.savefig("2DMovingWave.pdf")
Fig.show()