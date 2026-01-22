#! /usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt

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

    # Meshgrid for 2D array
    x, y = np.meshgrid(grid_x, grid_y)
    psi_n = (2*a/np.pi)**0.25*(2*b/np.pi)**0.25*np.exp(-a*x**2-b*y**2)*np.exp(1j*(kx*x+ky*y))  # Initial condition
    psi_n[0,:] = psi_n[-1,:] = 0  # Boundary condition
    psi_n[:,0] = psi_n[:,-1] = 0  # Boundary condition

    interior_Nx = (Nx-2)**2
    alpha = dt/(2*dx**2)
    
    # Setting values for matrix_a
    lowerA = np.full(interior_Nx-1, -1j*alpha/2, dtype=complex)
    diagA = np.full(interior_Nx, 1+1j*alpha+1j*alpha, dtype=complex)
    upperA = np.full(interior_Nx-1, -1j*alpha_x/2, dtype=complex)
    otherA = np.full(interior_Nx/2.0, -1j*alpha/2, dtype=complex)

    # Setting values for matrix_b
    lowerB = np.full(interior_Nx-1, 1j*alpha/2, dtype=complex)
    diagB = np.full(interior_Nx, 1-1j*alpha-1j*alpha, dtype=complex)
    upperB = np.full(interior_Nx-1, 1j*alpha/2, dtype=complex)
    otherB = np.full(interior_Nx/2.0, 1j*alpha/2, dtype=complex)

    rhs = diagB*psi_n[1:-1]
    rhs[1:] += lowerB*psi_n[1:-2]
    rhs[:-1] += upperB*psi_n[2:-1]

    # Setting arrays to solve pentadiagonal matrix
    A = np.zeros(interior_Nx, dtype=complex)
    B = np.zeros(interior_Nx-1, dtype=complex)
    C = np.zeros(interior_Nx-2, dtype=complex)
    D = np.zeros(interior_Nx, dtype=complex)
    E = np.zeros(interior_Nx, dtype=complex)
    psi_n1 = np.zeros(interior_Nx, dtype=complex)

    # Start solving for pentadiagonal matrix
    A[0] = diagA[0]
    B[0] = upperA[0]/diag[0]
    C[0] = otherA[0]/diag[0]

    A[1] = diagA[1]-upperA[1]*B[0]
    B[1] = (upperA[1]-otherA[0]*B[0])/A[1]
    C[1] = otherA[1]/A[1]

    for i in range(2, interior_Nx-2):
        A[i] = diagA[i]-otherA[i-2]*C[i-2]-A[i-1]*B[i-1]**2
        B[i] = (upperA[i]-otherA[i-1]*B[i-1])/A[i]
        C[i] = otherA[i]/A[i]

    A[-2] = diagA[-2]-otherA[-4]*C[-4]-A[-3]*B[-3]**2
    B[-2] = (upperA[-2]-otherA[-3]*B[-3])/A[-2]
    A[-1] = diagA[-1]-otherA[-3]*C[-3]-A[-2]*B[-2]**2

    # Solving for the right hand side
    E[0] = rhs[0]
    E[1] = rhs[1]-B[0]*E[0]
    for j in range(2, interior_Nx):
        E[j] = rhs[j]-B[j-1]*E[j-1]-C[j-2]*E[j-2]

    D = E/A

    # Back substitution
    psi_n1[-1] = D[-1]
    psi_n1[-2] = D[-2]-B[-2]*psi_n1[-1]
    for k in range(interior_Nx-3, -1, -1):
        psi_n1[k] = D[k]-B[k]*psi_n1[k+1]-C[k]*psi_n1[k+2]

    return(grid_x, grid_y, psi_n1)

T_values = list(map(float, input('Enter 9 time periods (t): ').split()))
inputs = list(map(float, input('Enter time step (dt), grid spacing (x direction) (dx), normalised Gaussian width (x direction) (a), normalised Gaussian width (y direction) (b), wave number (k): ').split()))

# 3D plot
fig = plt.figure()
psi_n1z = np.zeros((501,501))
for n in range(9):
    ax = fig.add_subplot(3, 3, n+1, projection="3d")
    gridx, psi_n1x = wavepacket(T_values[n], inputs[0], inputs[1], inputs[2], inputs[3])
    gridy, psi_n1y = wavepacket(T_values[n], inputs[0], inputs[1], inputs[2], inputs[3])
    for counter in range(len(gridx)):
        for counter2 in range(len(gridy)):
            psi_n1z[counter][counter2] = (np.abs(psi_n1x[counter] * psi_n1y[counter2]))**2
    ax.plot_surface(gridx, gridy, psi_n1z, cmap=cm.coolwarm, linewidth=0)
    ax.set_xlabel('Position (x)')
    ax.set_ylabel('Position (y)')
    ax.set_zlabel('(|ψ|^2)')
    ax.set_zlim([0, 0.85])

plt.tight_layout()
plt.savefig('3Dnumerical_1Dschrodinger.pdf')
plt.show()