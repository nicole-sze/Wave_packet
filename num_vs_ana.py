#! /usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import simpson

# Script to compare the 1D TDSE numerical to analytical
# Numerical function 
def wavepacket(t):
    '''
        Comparing the numerical and analytical 1D time dependent 
        Schrodinger equation results.

        Variables:
        t = Time period (s)
        dt = Time step
        dx = Grid spacing
        a = Normalised Gaussian width
        k = wave number

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

    # Setting values for matrix_a
    lowerA = np.full(interior_Nx-1, -1j*alpha/2, dtype=complex)
    diagA = np.full(interior_Nx, 1+1j*alpha, dtype=complex)
    upperA = np.full(interior_Nx-1, -1j*alpha/2, dtype=complex)

    # Setting values for matrix_b
    lowerB = np.full(interior_Nx-1, 1j*alpha/2, dtype=complex)
    diagB = np.full(interior_Nx, 1-1j*alpha, dtype=complex)
    upperB = np.full(interior_Nx-1, 1j*alpha/2, dtype=complex)

    # Initial condition
    psi_n = (2*a/np.pi)**0.25*np.exp(-a*grid**2)

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

# Analytical solution
def psix(x,t,a):
    return np.abs((2*a/np.pi)**(1/4)*1/np.sqrt(1+2j*a*t)*np.exp(-a*x**2/(1+2j*a*t)))**2

T_values = list(map(float, input('Enter 9 time periods (t): ').split()))

grid, psi_n1 = wavepacket(0)
print(f'Initial analytical normalisation: {simpson(psix(grid, 0, 1), x=grid):.14f}')

# 2D plot
fig = plt. figure()
fig.suptitle('Analytical VS numerical 1D probabilities without potential')

for m in range(9):
    grid, psi_n1 = wavepacket(T_values[m])

    initial = psix(grid, 0, 1)
    prob_density = simpson(np.abs(psi_n1)**2, x=grid)
    print('Normalisation (t ='+str(round(T_values[m], 3))+f'): {prob_density:.14f}')
    
    plt.subplot(3, 3, m+1)
    plt.plot(grid, np.abs(psi_n1)**2)
    plt.plot(grid, psix(grid, T_values[m], 1))
    plt.plot(grid, np.abs(psi_n1)**2-psix(grid, T_values[m], 1))
    plt.xlabel('Position (x)')
    plt.ylabel(r'$(|\Psi|)^2$')
    plt.ylim([0, 0.85])
    plt.title('t ='+str(round(T_values[m], 3)))
plt.tight_layout()
plt.savefig('num_vs_ana.pdf')
plt.show()