#! /usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse.linalg import spsolve
from matplotlib import cm
from scipy.sparse import csc_matrix


# Script to solve 2D time-dependent Schrodinger equation numerically

def wavepacket(t, dt, dx, a, k1, k2, v0):
    '''
        Solving the 2D time-dependent Schrodinger equation using the 
        Crank-Nicolson numerical method.

        Variables:
        t = Time period (s)
        dt = Time step
        dx = Grid spacing
        a = Normalised Gaussian width in x direction
        b = Normalised Gaussian width in y direction
        kx = wave number in x direction
        ky = wave number in y direction

        Outputs:
        psi_n1 = Wave function
        grid_x = Spatial grid in x direction
        grid_y = Spatial grid in y direction
    '''

    # Setting parameters
    # Upper limit is given as 10+dx since arange generates a half-open interval
    times = np.arange(0, t+dt, dt)
    grid_x1 = np.arange(-5, 5+dx, dx)
    grid_x2 = np.arange(-5, 5+dx, dx)
    Nx = len(grid_x1)  # Same as len(grid_y)
    Nt = len(times)

    # Making psi_n into a 2D matrix for plotting
    psi_n = np.zeros((Nx, Nx), dtype = complex)
    for i, xi in enumerate(grid_x1):
        for j, xj in enumerate(grid_x2):
            psi_n[i, j] = (2*a/np.pi)**0.5*np.exp(-a*((xi+2)**2+(xj-2)**2))*np.exp(1j*(k1*xi + k2*xj))  # Initial condition
    psi_n[0, :] = psi_n[-1, :] = 0  # Boundary condition
    psi_n[:, 0] = psi_n[:, -1] = 0  # Boundary condition

    # Converting psi_n into a 1D for calculations
    psi_n = psi_n.flatten()

    alpha = dt/(2*dx**2)
    N = Nx*Nx  # To accomodate flattened psi_n

    # Creating potential matrix
    alpha = 1
    v = np.zeros((Nx, Nx))
    for q, qi in enumerate(grid_x1):
        for w, wi in enumerate(grid_x2):
            v[q, w] = v0*np.exp(-alpha*np.abs(qi-wi)**2)

    v = v.flatten()

    # Setting up matrix A
    array_A = np.zeros((N,N),dtype=complex)
    for s in range(N):
        for r in range(N):
            if s == r:
                array_A[s,r] = complex(1+1j*alpha+1j*alpha + 1j*dt*v[r]/2)
            elif s == r + 1 or s == r - 1:
                array_A[s,r] = complex(-1j*alpha/2)
            elif s == r + Nx or s == r - Nx:
                array_A[s,r] = complex(-1j*alpha/2)

    # Setting up matrix 
    array_B = np.zeros((N,N),dtype=complex)
    for n in range(N):
        for m in range(N):
            if n == m:
                array_B[n,m] = complex(1-1j*alpha-1j*alpha - 1j*dt*v[m]/2)
            elif n == m + 1 or n == m - 1:
                array_B[n,m] = complex(1j*alpha/2)
            elif n == m + Nx or n == m - Nx:
                array_B[n,m] = complex(1j*alpha/2)
        
    psi_n1 = psi_n.copy()

    for k in range(Nt):
        vector_b = array_B @ psi_n1
        psi_n1 = spsolve(csc_matrix(array_A),vector_b)
        psi_n1[0] = 0

        # Boundary conditions
        for q in range(N):
            if q <= Nx  or q >= (Nx*(Nx-1)):
                psi_n1[q] = 0
            elif ((q%Nx)==0):
                psi_n1[q] = 0
                psi_n1[q-1] = 0
         
    # Converting psi_n1 to 2D for plotting
    psi_n1 = psi_n1.reshape((Nx, Nx))
    
    return (psi_n1, grid_x1, grid_x2)

inputs = list(map(float, input('Time period (t), Time step (dt), grid spacing (x direction) (dx), normalised Gaussian width (x direction) (a), and wave number for x1 and x2, potential (v0): ').split()))

# 3D plot
psi_n1, grid_x, grid_y = wavepacket(inputs[0], inputs[1], inputs[2], inputs[3], inputs[4], inputs[5], inputs[6])
x, y = np.meshgrid(grid_x, grid_y)
z = np.abs(psi_n1)**2

plt.contourf(x, y, z)
plt.xlabel('x1')
plt.ylabel('x2')
plt.savefig('collision.pdf')
plt.show()