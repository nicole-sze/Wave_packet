#! /usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt

# Script to solve a quantum harmonic oscillator numerically
def wavepacket(t, k, spring):
    '''
        Solving a quantum harmonic oscillator numerically via
        the Crank-Nicholson method.

        Variables:
        t = Time period (s)
        k = Wave number
        spring = Spring constant

        Outputs:
        grid = Spatial grid
        psi_n1 = Wave function
    '''
    # Setting parameters
    dt = 0.001
    dx = 0.02
    a = 1

    # Upper limit is given as 10+dx since arange generates a half-open interval
    times = np.arange(0, t+dt, dt)
    grid = np.arange(-5, 5+dx, dx)
    Nx = len(grid)
    Nt = len(times)

    interior_Nx = Nx-2
    alpha = dt/(2*dx**2)

    # Setting up harmonic oscillator potential
    v = 0.5*spring*grid[1:-1]**2  # grid[1:-1] to match interior_Nx dimensions

    # Setting values for matrix_a
    lowerA = np.full(interior_Nx-1, -1j*alpha/2, dtype=complex)
    diagA = np.full(interior_Nx, 1+1j*alpha, dtype=complex) + 1j*dt*v/2
    upperA = np.full(interior_Nx-1, -1j*alpha/2, dtype=complex)

    # Setting values for matrix_b
    lowerB = np.full(interior_Nx-1, 1j*alpha/2, dtype=complex)
    diagB = np.full(interior_Nx, 1-1j*alpha, dtype=complex) - 1j*dt*v/2
    upperB = np.full(interior_Nx-1, 1j*alpha/2, dtype=complex)

    psi_n = (2*a/np.pi)**0.25*np.exp(-a*grid**2)*np.exp(1j*k*grid)  # Initial condition
    psi_n[0] = psi_n[-1] = 0  # Boundary condition
    psi_n1 = np.zeros(Nx, dtype=complex)

    for i in range(Nt):
        rhs = diagB*psi_n[1:-1]
        rhs[1:] += lowerB*psi_n[1:-2]
        rhs[:-1] += upperB*psi_n[2:-1]

        # To reset matices every time
        A = lowerA.copy()
        B = diagA.copy()
        C = upperA.copy()

        # Performing the Thomas Algorithm
        # Forward elimination
        for j in range(1, interior_Nx):
            ratio = A[j-1]/B[j-1]
            B[j] = B[j] - ratio*C[j-1]
            rhs[j] = rhs[j] - ratio*rhs[j-1]

            # Saftey check:
            if abs(B[j-1]) < 1e-12:
                print("Zero pivot encountered — Thomas algorithm fails.")

        # Back substitution
        interior_psi = np.zeros(interior_Nx, dtype=complex)
        interior_psi[-1] = rhs[-1]/B[-1]  # Computes last unknown
        for n in range(interior_Nx-2, -1, -1):
            interior_psi[n] = (rhs[n] - C[n]*interior_psi[n+1])/B[n]

        psi_n1[0] = psi_n1[-1] = 0
        psi_n1[1:-1] = interior_psi
        psi_n = psi_n1.copy()

    return(grid, psi_n1)

T_values = list(map(float, input('Enter 9 time periods (t): ').split()))
inputs = list(map(float, input('Enter normalised wave number (k) and spring constant (k): ').split()))

# 2D plot with 9 different time periods
for m in range(9):
    grid, psi_n1 = wavepacket(T_values[m], inputs[0], inputs[1])
    plt.subplot(3, 3, m+1)
    plt.plot(grid, np.abs(psi_n1)**2)
    plt.xlabel('Position (x)')
    plt.ylabel('(|ψ|^2)')
    plt.ylim([0, 0.85])
    plt.title('t ='+str(round(T_values[m], 3)))

plt.tight_layout()
plt.savefig('harmonic_numerical.pdf')
plt.show()