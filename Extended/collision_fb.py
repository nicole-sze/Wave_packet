#! /usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse.linalg import spsolve
from scipy.sparse import csc_matrix
from scipy.integrate import simpson


# Script to solve the 2 particle Schrodinger equation numerically

def wavepacket(t, k1, k2, v0):
    '''
        Solving the 2 particle Schrodinger equation using 
        the Crank-Nicolson numerical method.

        Variables:
        t = Time period (s)
        k1 = 1st wave's wave number
        k2 = 2nd wave's wave number

        Outputs:
        psi_n1 = Wave function
        grid_x = Spatial grid in x direction
        grid_y = Spatial grid in y direction
    '''

    # Setting parameters
    dt = 0.01
    dx = 0.15
    a = 1
    
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
            psi_n[i, j] = ((2*a/np.pi)**0.5*np.exp(-a*((xi+3)**2+(xj-3)**2))*np.exp(1j*(k1*xi + k2*xj)) - (2*a/np.pi)**0.5*np.exp(-a*((xj+3)**2+(xi-3)**2))*np.exp(1j*(k1*xj + k2*xi)))/(np.sqrt(2))  # Initial condition
    psi_n[0, :] = psi_n[-1, :] = 0  # Boundary condition
    psi_n[:, 0] = psi_n[:, -1] = 0  # Boundary condition

    # Converting psi_n into a 1D for calculations
    psi_n = psi_n.flatten()

    alpha = dt/(2*dx**2)
    N = Nx*Nx  # To accomodate flattened psi_n

    # Creating potential matrix
    alpha_v = 5
    v = np.zeros((Nx, Nx))
    for q, qi in enumerate(grid_x1):
        for w, wi in enumerate(grid_x2):
            v[q, w] = v0*np.exp(-alpha_v*np.abs(qi-wi)**2)

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

        print(k)
         
    # Converting psi_n1 to 2D for plotting
    psi_n1 = psi_n1.reshape((Nx, Nx))
    
    return (psi_n1, grid_x1, grid_x2)

inputs = list(map(float, input('Time period (t), wave number for x1 and x2, potential (v0): ').split()))

# Contour plot
psi_n1, grid_x1, grid_x2 = wavepacket(inputs[0], inputs[1], inputs[2], inputs[3])
x1, x2 = np.meshgrid(grid_x1, grid_x2)
z = np.abs(psi_n1)**2

plt.figure()
plt.contourf(x1, x2, z)
plt.plot(grid_x1, grid_x1, '--')
plt.xlabel('x1')
plt.ylabel('x2')
plt.title('2-particle Schrodinger equation probability density')
plt.savefig('fb_collision_contour.pdf')
plt.show()

# Separate into 2 waves by marginalisation
# i.e. integrating over probability density with respect to the other wave
wave1 = simpson(z, x=grid_x2, axis=1)
wave2 = simpson(z, x=grid_x1, axis=0)

# 2D plot
plt.figure()
plt.plot(grid_x1, wave1, label='1st wave (x1)')
plt.plot(grid_x2, wave2, label='2nd wave (x2)')
plt.xlabel('Position (x)')
plt.ylabel(r'$(|\Psi|)^2$')
plt.title('2-particle Schrodinger equation collision')
plt.legend()
plt.savefig('fb_collision_2D.pdf')
plt.show()