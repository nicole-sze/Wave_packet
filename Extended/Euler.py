#! /usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import simpson


def Euler_schrodinger( dt, dx, t):
    '''
    Function to solve the time-dependent Schrödinger equation using the Euler method with no potential component.
    Parameters:
    - dt: Time step
    - dx: Spatial step
    - t: Total time
    Returns:
    - psi_n: Wave function at each time step
    - grid_x: Spatial grid points
    '''
   
    # Initialize the wave function (psi) and grids
    times = np.arange(0, t+dt, dt)
    grid_x = np.arange(-5, 5+dx, dx)
    Nx = len(grid_x)
    Nt = len(times)
    psi_n = np.zeros((Nx, Nx),dtype = complex)
   
    # Euler method coefficient
    alpha = dt / (2 * dx**2)

    # Initial condition: Gaussian wave packet centered at x=0 with width 0.5
    psi = np.exp(-0.5 * (grid_x / 0.5)**2).astype(complex)
   
    # Normalize
    psi /= np.sqrt(np.sum(np.abs(psi)**2) * dx)
    psi_n[0, :] = psi

    # Euler time-stepping loop
    for n in range(1, Nt):
        psi_n[n, 1:-1] = (psi_n[n-1, 1:-1] - 1j * alpha * (psi_n[n-1, 2:] - 2.0 * psi_n[n-1, 1:-1] + psi_n[n-1, :-2]))
       
       
        # Boundary conditions (Hard walls)
        psi_n[n, 0] = 0.0
        psi_n[n, -1] = 0.0

    # Calculate probability density
    prob = np.sum(np.abs(psi_n)**2, axis=0) * dx



    return psi_n, grid_x, times, prob


# Main block

# Parameters
dt = 0.01
dx = 0.1
T = 0.1

psi_n, grid_x, times, prob = Euler_schrodinger(dt, dx, T)

plt.plot(grid_x, prob)
plt.xlabel("position (x)")
plt.ylabel(r'Probability density (|$\psi_{n}^2$|)')
plt.grid()
plt.savefig('Euler.pdf')
plt.show()
