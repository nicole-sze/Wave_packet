#! /usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse.linalg import spsolve
from matplotlib import cm
from scipy.sparse import csc_matrix


# Script to solve 2D time-dependent Schrodinger equation numerically

def wavepacket(t, dt, dx, a, b, kx, ky):
    '''
        Solving the 2D time-dependent Schrodinger equation using the Crank-Nicolson
        numerical method. This results in a large, complex matrix which is then computed
        with the Thomas Algorithm.

        Variables:
        t = Time period (s)
        dt = Time step
        dx = Grid spacing in x direction
        a = Normalised Gaussian width in x direction
        b = Normalised Gaussian width in y direction
        kx = wave number in x direction
        ky = wave number in y direction

        Outputs:
        grid_x = Spatial grid in x direction
        grid_y = Spatial grid in y direction
        psi_n1 = Wave function
    '''

    # Setting parameters
    # Upper limit is given as 10+dx since arange generates a half-open interval
    times = np.arange(0, t+dt, dt)
    grid_x = np.arange(-5, 5+dx, dx)
    grid_y = np.arange(-5, 5+dx, dx)
    Nx = len(grid_x)
    Nt = len(times)

    psi_n = np.zeros(Nx, dtype = complex)
    
    for s in range(len(grid_x)):
        for r in range(len(grid_y)):
            psi_n[s+r] = (2*a/np.pi)**0.25*(2*b/np.pi)**0.25*np.exp(-a*s**2-b*r**2)*np.exp(1j*(kx*s+ky*r))  # Initial condition
    psi_n[0] = psi_n[-1] = 0  # Boundary condition

    alpha = dt/(2*dx**2)

    # Setting up matrix A
    array_A = np.zeros((Nx,Nx),dtype=complex)
    for i in range(Nx):
        for j in range(Nx):
            if i == j:
                array_A[i,j] = complex(1+1j*alpha+1j*alpha)
            elif i == j + 1 or i == j - 1:
                array_A[i,j] = complex(-1j*alpha/2)
            elif i == j + Nx or i == j - Nx:
                array_A[i,j] = complex(-1j*alpha/2)

    # Setting up matrix 
    array_B = np.zeros((Nx,Nx),dtype=complex)
    for n in range(Nx):
        for m in range(Nx):
            if n == m:
                array_B[n,m] = complex(1-1j*alpha-1j*alpha)
            elif n == m + 1 or n == m - 1:
                array_B[n,m] = complex(1j*alpha/2)
            elif n == m + Nx or n == m - Nx:
                array_B[n,m] = complex(1j*alpha/2)
    
    vector_b = array_B @ psi_n
    
    psi_n1 = spsolve(csc_matrix(array_A),vector_b)
    
    return (psi_n1, grid_x, grid_y)

inputs = list(map(float, input('Enter time period (t), time step (dt), grid spacing (x direction) (dx), normalised Gaussian width (x direction) (a), normalised Gaussian width (y direction) (b), wave number (kx), and wave number (ky): ').split()))

# Setting 2D matrix for plotting
psi_n1, grid_x, grid_y = wavepacket(inputs[0], inputs[1], inputs[2], inputs[3], inputs[4], inputs[5], inputs[6])
array = np.zeros((len(grid_x), len(grid_y)), dtype = complex)
counter = 0
for i in range(len(grid_x)):
    for j in range(len(grid_y)):
        array[i,j] = psi_n1[j]

print(psi_n1z)

# 3D plot
fig = plt.figure()
psi_n1z = np.zeros((501,501))
for n in range(9):
    ax = fig.add_subplot(3, 3, n+1, projection="3d")
    xs, ys, z = wavepacket(inputs[0], inputs[1], inputs[2], inputs[3], inputs[4], inputs[5], inputs[6])
    ax.plot_surface(grid_x, grid_y, psi_n1z, cmap=cm.coolwarm, linewidth=0)
    ax.set_xlabel('Position (x)')
    ax.set_ylabel('Position (y)')
    ax.set_zlabel('(|ψ|^2)')
    ax.set_zlim([0, 0.85])

plt.tight_layout()
plt.savefig('3Dnumerical_1Dschrodinger.pdf')
plt.show()