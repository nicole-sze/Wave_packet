#! /usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import simpson

# Script to calculate the probability of the particle crossing the barrier as a function of its kinetic energy(1D)
def wavepacket(t, dt, dx, a, k, v0):
    '''
        Variables:
        t = Time period (s)
        k = wave number
                v0 = hieght of potential barrier
       
                Outputs:
        crossing probabilty
    '''

    # Setting parameters
    # Upper limit is given as 10+dx since arange generates a half-open interval
    times = np.arange(0, t+dt, dt)
    grid = np.arange(-10, 10+dx, dx)
    Nx = len(grid)
    Nt = len(times)
       

       
    interior_Nx = Nx-2
    alpha = dt/(2*dx**2)
       
        # defining vector which contains potential for each space index
   
        # p = centre of barrier(units of dx)
        # b = width of barrier(units of dx and even number)
    p = 65
    b = 8
        # Loops through each space index and inserts corresponding potential
    v = np.zeros(interior_Nx)
    for i in range(interior_Nx):
        if i < p - b/2 or i > p + b/2:
            v[i] = 0
        else:
            v[i] = v0
        
               
        # Setting values for matrix_a
    lowerA = np.full(interior_Nx-1, -1j*alpha/2, dtype=complex)
    diagA = np.full(interior_Nx, 1+1j*alpha, dtype=complex) + 1j*dt*v/2
    upperA = np.full(interior_Nx-1, -1j*alpha/2, dtype=complex)

        # Setting values for matrix_b
    lowerB = np.full(interior_Nx-1, 1j*alpha/2, dtype=complex)
    diagB = np.full(interior_Nx, 1-1j*alpha, dtype=complex) - 1j*dt*v/2
    upperB = np.full(interior_Nx-1, 1j*alpha/2, dtype=complex)   
    # Initial condition
    psi_n = (2*a/np.pi)**0.25*np.exp(-a*grid**2)*np.exp(1j*k*grid) 
    # Boundary condition
    psi_n[0] = psi_n[-1] = 0 
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
    interval = grid[69:100]
    prob = np.abs(psi_n1[69:100])**2
    crossprob = simpson(prob, x=interval)
    return(crossprob)

inputs = list(map(float, input('Enter a time period, and a potential barrier height:').split()))
x = np.arange(0, 100.2, 0.2)
plt.plot(x, [wavepacket(inputs[0], 0.01, 0.2, 1, np.sqrt(2*i), inputs[1]) for i in x])
plt.xlabel('Kinetic energy')
plt.ylabel('Crossing probability')
plt.savefig('cross_probability_plot.pdf')
plt.show()