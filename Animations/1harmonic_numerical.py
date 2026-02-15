#! /usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

from scipy.integrate import simpson

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
    dt = 0.1
    dx = 0.15
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

    psi_frames = []
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
        psi_frames.append(np.abs(psi_n1)**2)
        psi_n = psi_n1.copy()

    return(psi_frames, grid, times)

# User input
T = float(input("Enter time period t: "))
k, spring = map(float, input("Enter k and spring: ").split())

psi_frames, grid, times = wavepacket(T, k, spring)

integral = simpson(psi_frames[-1], x=grid)
print(f'integral: {integral}')

# Setup figure
fig, ax = plt.subplots()
ax.set_xlabel('Position (x)')
ax.set_ylabel('(|ψ|^2)')

# Initial frame
plot, = ax.plot(grid, psi_frames[0])

# Update function for animation
def update(frame):
    plot.set_ydata(psi_frames[frame])
    ax.set_title(f"t = {times[frame]:.3f}")
    print(frame)
    return plot,

# Create animation
animation = FuncAnimation(fig, update, frames=len(times), interval=50, blit=False)

animation.save("harmonic_numerical.gif", writer="pillow", fps=40)
plt.show()