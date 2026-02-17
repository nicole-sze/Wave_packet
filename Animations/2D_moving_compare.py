#! /usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse.linalg import spsolve
from matplotlib import cm
from scipy.sparse import csc_matrix
from matplotlib.animation import FuncAnimation

# Script to solve 2D time-dependent Schrodinger equation wihout potential numerically

def wavepacket(t, kx, ky):
    '''
        Solving the 2D time-dependent Schrodinger equation without
        potential using the Crank-Nicolson numerical method.

        Variables:
        t = Time period (s)
        kx = Wave number in x direction
        ky = Wave number in y direction

        Outputs:
        psi_n1 = Wave function
        grid_x = Spatial grid in x direction
        grid_y = Spatial grid in y direction
    '''

    # Setting parameters
    dt = 0.01
    dx = 0.2
    a = 1
    b = 1
    
    # Upper limit is given as 10+dx since arange generates a half-open interval
    times = np.arange(0, t, dt)
    grid_x = np.arange(-5, 5+dx, dx)
    grid_y = np.arange(-5, 5+dx, dx)
    Nx = len(grid_x)  # Same as len(grid_y)
    Nt = len(times)

    # Making psi_n into a 2D matrix for plotting
    psi_n = np.zeros((Nx, Nx), dtype = complex)
    for i, xi in enumerate(grid_x):
        for j, xj in enumerate(grid_y):
            psi_n[i, j] = (2*a/np.pi)**0.25*(2*b/np.pi)**0.25*np.exp(-a*xi**2-b*xj**2)*np.exp(1j*(kx*xi+ky*xj))  # Initial condition
    psi_n[0, :] = psi_n[-1, :] = 0  # Boundary condition
    psi_n[:, 0] = psi_n[:, -1] = 0  # Boundary condition

    # Converting psi_n into a 1D for calculations
    psi_n = psi_n.flatten()

    alpha = dt/(2*dx**2)
    N = Nx*Nx  # To accomodate flattened psi_n

    # Setting up matrix A
    array_A = np.zeros((N,N),dtype=complex)
    for s in range(N):
        for r in range(N):
            if s == r:
                array_A[s,r] = complex(1+1j*alpha+1j*alpha)
            elif s == r + 1 or s == r - 1:
                array_A[s,r] = complex(-1j*alpha/2)
            elif s == r + Nx or s == r - Nx:
                array_A[s,r] = complex(-1j*alpha/2)

    # Setting up matrix B
    array_B = np.zeros((N,N),dtype=complex)
    for n in range(N):
        for m in range(N):
            if n == m:
                array_B[n,m] = complex(1-1j*alpha-1j*alpha)
            elif n == m + 1 or n == m - 1:
                array_B[n,m] = complex(1j*alpha/2)
            elif n == m + Nx or n == m - Nx:
                array_B[n,m] = complex(1j*alpha/2)

    psi_n1 = psi_n.copy()
    
    for k in range(Nt):
        vector_b = array_B @ psi_n1
        psi_n1 = spsolve(csc_matrix(array_A),vector_b)
        psi_n1[0] = 0
        print(k)
        # Boundary conditions
        for q in range(N):
            if q <= Nx  or q >= (Nx*(Nx-1)):
                psi_n1[q] = 0
            elif ((q%Nx)==0):
                psi_n1[q] = 0
                psi_n1[q-1] = 0
    
    # Converting psi_n1 to 2D for plotting
    psi_n1 = psi_n1.reshape((Nx, Nx))
    z = np.abs(psi_n1)**2
    # prob = 0
    # da = dx**2
    # for i in range(len(grid_x)):
    #     for j in range(len(grid_x)):
    #         x_integral = z[i,j]*da
    #         prob += x_integral

    # print(prob)
        
    return (psi_n1, grid_x, grid_y, dt)


def psi3d(x,y,t,a,b,kx0,ky0):
    # Takes in the x and y grids, each time step, the width of the gaussian and the initial velocity
    # Returns the probability density
    return (np.abs((2*a/np.pi)**(1/4)*1/np.sqrt(1+2j*a*t)*np.exp((-a*x**2+1j*x*kx0)/(1+2j*a*t)+(kx0**2/(4*a))*(1/(1+2j*a*t)-1))) * (np.abs((2*b/np.pi)**(1/4)*1/np.sqrt(1+2j*b*t)*np.exp((-b*y**2+1j*y*ky0)/(1+2j*b*t)+(ky0**2/(4*b))*(1/(1+2j*b*t)-1)))))**2

inputs = float(input('Enter time period (t): '))

n = 10
times = np.arange(0,inputs,inputs/n)

# Precompute the wavefunction over time once
psi_frames = []

for t in times:
    psi_n1, grid_x, grid_y, dt = wavepacket(t,1,1)
    psi_a1 = np.zeros((len(grid_x),len(grid_x)))
    for counter in range(len(grid_x)):
        for counter2 in range(len(grid_x)):
            psi_a1[counter][counter2] = psi3d(grid_x[counter], grid_x[counter2], t, 1, 1 , 1, 1)
    x, y = np.meshgrid(grid_x, grid_y)
    z = psi_a1-np.abs(psi_n1)**2
    psi_frames.append(z)

fig = plt.figure()
ax = fig.add_subplot(projection='3d')

x, y = np.meshgrid(grid_x, grid_y)

ax.set_xlabel('Position (x)')
ax.set_ylabel('Position (y)')
ax.set_zlabel('|ψ|²')
ax.set_zlim(-0.01, 0.01)

# Creating initial surface:

surface = ax.plot_surface(x, y , psi_frames[0], cmap=cm.coolwarm, linewidth=0)

# Removes old surface, replaces w/ new one, updates title the returns the surface
def update(frame):
    global surface

    surface.remove()

    surface = ax.plot_surface(x, y, psi_frames[frame], cmap=cm.coolwarm, linewidth=0)

    ax.set_title(f"t = {times[frame]:.3f}")

    return surface

ani = FuncAnimation(fig, update, frames=len(times), interval=1000, blit=False)

ani.save("2d_moving_compare.gif", writer="pillow", fps=5)
plt.close(fig)