#! /usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse.linalg import spsolve
from matplotlib import cm
from scipy.sparse import csc_matrix


# Script to simulate a single slit experiment numerically

def wavepacket(t, ky):
    '''
        Simulating a single slit experiment using 
        the Crank-Nicolson numerical method.

        Variables:
        t = Time period (s)
        ky = Wave number in y direction (because does not move in x-direction)

        Outputs:
        psi_n1 = Wave function
        grid_x = Spatial grid in x direction
        grid_y = Spatial grid in y direction
    '''

    # Setting parameters
    dt = 0.01
    dx = 0.15
    a = b = 1
    
    # Upper limit is given as 10+dx since arange generates a half-open interval
    times = np.arange(0, t+dt, dt)
    grid_x = np.arange(-5, 5+dx, dx)
    grid_y = np.arange(-5, 5+dx, dx)
    Nx = len(grid_x)  # Same as len(grid_y)
    Nt = len(times)

    # Making psi_n into a 2D matrix for plotting
    psi_n = np.zeros((Nx, Nx), dtype = complex)
    for i, xi in enumerate(grid_x):
        for j, xj in enumerate(grid_y):
            psi_n[i, j] = (2*a/np.pi)**0.25*(2*b/np.pi)**0.25*np.exp(-a*(xi+2)**2-b*xj**2)*np.exp(1j*ky*(xi+2))  # Initial condition
    psi_n[0, :] = psi_n[-1, :] = 0  # Boundary condition
    psi_n[:, 0] = psi_n[:, -1] = 0  # Boundary condition

    # Converting psi_n into a 1D for calculations
    psi_n = psi_n.flatten()

    alpha = dt/(2*dx**2)
    N = Nx*Nx  # To accomodate flattened psi_n

    # defining vector which contains potential for each space index
    # x_1 = centre of barrier(units of dx)
    # x_2 = width of barrier(units of dx and even number)
    # y_1 = width of barrier in y (units of dx)
    x_1 = np.round(Nx/2.0,0)
    x_2 = 8
    y_1 = 8
    # Loops through each space index and inserts corresponding potential
    v = np.zeros((Nx, Nx))
    for i in range(Nx):
        for j in range(Nx):
            if j > x_1 - y_1/2.0 and j < x_1 + y_1/2.0:
                if i > x_1 - x_2/2.0 and i < x_1 + x_2/2.0:
                    v[j,i] = 0
                else:
                    v[j,i] = 2000000000

    v = v.flatten()

    # Setting up matrix A
    array_A = np.zeros((N,N),dtype=complex)
    for s in range(N):
        for r in range(N):
            if s == r:
                array_A[s,r] = complex(1+1j*alpha+1j*alpha) + 1j*dt*v[r]/2
                #print(array_A[s,r])
            elif s == r + 1 or s == r - 1:
                array_A[s,r] = complex(-1j*alpha/2)
            elif s == r + Nx or s == r - Nx:
                array_A[s,r] = complex(-1j*alpha/2)

    # Setting up matrix 
    array_B = np.zeros((N,N),dtype=complex)
    for n in range(N):
        for m in range(N):
            if n == m:
                array_B[n,m] = complex(1-1j*alpha-1j*alpha) - 1j*dt*v[m]/2
                #print(array_B[n,m])
            elif n == m + 1 or n == m - 1:
                array_B[n,m] = complex(1j*alpha/2)
            elif n == m + Nx or n == m - Nx:
                array_B[n,m] = complex(1j*alpha/2)
        
    psi_n1 = psi_n.copy()
    
    for k in range(Nt):
        vector_b = array_B @ psi_n1
        psi_n1 = spsolve(csc_matrix(array_A),vector_b)
    
    # Converting psi_n1 to 2D for plotting
    psi_n1 = psi_n1.reshape((Nx, Nx))
    
    return (psi_n1, grid_x, grid_y)

inputs = list(map(float, input('Last time, and wave number in y direction (ky): ').split()))

n=9
times = np.arange(0,inputs[0],inputs[0]/n)

# 3D plot with 9 different time periods
fig = plt.figure()
for counter in range(n):
    psi_n1, grid_x, grid_y = wavepacket(times[counter], inputs[1])
    x, y = np.meshgrid(grid_x, grid_y)
    z = np.abs(psi_n1)**2
    ax = fig.add_subplot(int(np.sqrt(n)),int(np.sqrt(n)), counter +1, projection="3d")
    ax.plot_surface(x, y, z, cmap=cm.coolwarm, linewidth=0)
    ax.set_xlabel('Position (x)')
    ax.set_ylabel('Position (y)')
    ax.set_zlabel('(|ψ|^2)')
    ax.set_zlim([0, 0.85])
    plt.title("t = "+str(round(times[counter],2)))
plt.tight_layout()
plt.savefig('single_slit.pdf')
plt.show()