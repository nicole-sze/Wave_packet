#! /usr/bin/env python3

import math as maths
import matplotlib.pyplot as plt
import matplotlib.image
import numpy as np
import cmath
from matplotlib import cm
from scipy.integrate import simpson

"""
    Produces static time progression images for the stationary 
    wavepacket in free space in 1 and 2D. Inputs are not taken 
    as the parameters are standardised to be compared against 
    the dynamic code.
"""

# Setting up the grid values for the graph
xvals = np.arange(-3,3,0.01)
yvals = np.arange(-3,3,0.01)
xvals, yvals = np.meshgrid(xvals, yvals)

# Hard coded time-stamps and gaussian width
n = 9
t = np.arange(0,1,1/n)
a = 1

"""
    Defining the functions
"""
# Stationary 1D
def psix(x,t,a):
    # Takes in the x grid, the time stamp and the gaussian width
    # Returns the probability density
      return np.abs((2*a/np.pi)**(1/4)*1/np.sqrt(1+2j*a*t)*np.exp(-a*x**2/((1+2j*a*t))))**2

# Stationary 3D
def psi3d(x,y,t,a):
    # Takes in the x and y grids, the time stamp and the gaussian width
    # Returns the probability density
      return (np.abs((2*a/np.pi)**(1/4)*1/np.sqrt(1+2j*a*t)*np.exp(-a*x**2/(np.sqrt(1+2j*a*t))))**2) * (np.abs((2*a/np.pi)**(1/4)*1/np.sqrt(1+2j*a*t)*np.exp(-a*y**2/((1+2j*a*t))))**2)

"""
    1D Graph
"""
# Creating the 1D graph for each time stamp
for counter in range(len(t)):
      plt.subplot(int(np.sqrt(n)), int(np.sqrt(n)),counter+1) 
      tn = t[counter]
      plt.plot(xvals, psix(xvals,tn,a))
      plt.title("t = "+str(round(tn,3)))
      plt.xlabel("x")
      plt.ylabel(r"$(|\Psi|)^2$")
      plt.ylim([0,1])


plt.suptitle("Stationary Wave")
plt.tight_layout()
plt.show()
plt.savefig("StaionaryWave.pdf")

"""
    2D Graph
"""
fig = plt.figure(figsize=(16,15))
# Creating the 2D graph for each time stamp
for counter2 in range(len(t)):
      ax = fig.add_subplot(int(np.sqrt(n)), int(np.sqrt(n)), counter2+1, projection='3d')
      tn = t[counter2]
      ax.plot_surface(xvals, yvals, psi3d(xvals,yvals,t[counter2],a), cmap=cm.coolwarm, linewidth=0)
      ax.set_title("t = "+str(round(tn,3)),fontsize=20)
      ax.set_xlabel("x")
      ax.set_ylabel("y")
      ax.set_zlabel(r"$(|\Psi|)^2$")
      ax.set_zlim(0, 0.85)

plt.suptitle("2D Stationary Wave", fontsize=40)
plt.tight_layout()
plt.show()
plt.savefig("2DStaionaryWave.pdf")
plt.show()