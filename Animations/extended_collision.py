#! /usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse.linalg import spsolve
from scipy.sparse import csc_matrix
from matplotlib.animation import FuncAnimation
from matplotlib import cm
from scipy.integrate import simpson

# Script to solve the 2 particle Schrodinger equation with harmonic oscilltor numerically.

def wavepacket(t, dt, dx, a, k1, k2, v0, spring):
    '''
        Solving the 2 particle Schrodinger equation using 
        the Crank-Nicolson numerical method.

        Variables:
        t = Time period (s)
        dt = Time step
        dx = Grid spacing
        a = Normalised Gaussian width
        k1 = 1st wave's wave number
        k2 = 2nd wave's wave number
        v0 = Potential
        spring = Spring constant

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
            psi_n[i, j] = (2*a/np.pi)**0.5*np.exp(-a*((xi+3)**2+(xj-3)**2))*np.exp(1j*(k1*xi + k2*xj))  # Initial condition
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
            v[q, w] = v0*np.exp(-alpha_v*np.abs(qi-wi)**2)+0.5*spring*(qi+wi)**2

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

    psi_frames = []
    psi_n1 = psi_n.copy()

    for k in range(Nt):
        vector_b = array_B @ psi_n1
        psi_n1 = spsolve(csc_matrix(array_A),vector_b)

        # Boundary conditions
        for q in range(N):
            if q <= Nx  or q >= (Nx*(Nx-1)):
                psi_n1[q] = 0
            elif ((q%Nx)==0):
                psi_n1[q] = 0
                psi_n1[q-1] = 0

        psi_n1 = psi_n1.reshape((Nx, Nx))
        psi_frames.append(np.abs(psi_n1)**2)
        print(k)
        psi_n1 = psi_n1.flatten()
    
    return (psi_frames, grid_x1, grid_x2)

# User input
T = float(input("Enter time period t: "))
dt, dx, a, k1, k2, v0, spring = map(float, input("Enter dt, dx, a, k1, k2, v0, spring constant: ").split())
times = np.arange(0, T+dt, dt)

psi_frames, grid_x1, grid_x2 = wavepacket(T, dt, dx, a, k1, k2, v0, spring)

# Plotting contour graph -------------------------
# Setup figure
fig, ax = plt.subplots()
x1, x2 = np.meshgrid(grid_x1, grid_x2)
plt.plot(grid_x1, grid_x1, '--')
ax.set_xlabel('x1')
ax.set_ylabel('x2')

# Initial frame
contour = ax.contourf(x1, x2, psi_frames[0], cmap=cm.coolwarm)

# Update function for animation
def update(frame):
    global contour
    contour.remove()
    contour = ax.contourf(x1, x2, np.abs(psi_frames[frame])**2, cmap=cm.coolwarm, levels=50)
    ax.set_title(f"t = {times[frame]:.3f}")
    return contour

# Create animation
animation = FuncAnimation(fig, update, frames=len(times), interval=50, blit=False)

animation.save("contour_extended_collision.gif", writer="pillow", fps=15)
plt.show()

# Plotting 2D graph -----------------------------
# Setup figure
fig, ax = plt.subplots()
ax.set_xlabel('Position (x)')
ax.set_ylabel('(|ψ|^2)')

# Separate into 2 waves by marginalisation
# i.e. integrating over probability density with respect to the other wave
wave1 = simpson(psi_frames[0], x=grid_x2, axis=1)
wave2 = simpson(psi_frames[0], x=grid_x1, axis=0)

# Initial frame
plot1 = ax.plot(grid_x1, wave1, label='1st wave (x1)')[0]
plot2 = ax.plot(grid_x2, wave2, label='2nd wave (x2)')[0]
ax.legend()

# Update function for animation
def update(frame):
    wave1 = simpson(psi_frames[frame], x=grid_x2, axis=1)
    wave2 = simpson(psi_frames[frame], x=grid_x1, axis=0)
    plot1.set_ydata(wave1)
    plot2.set_ydata(wave2)
    ax.set_title(f"t = {times[frame]:.3f}")
    print(frame)
    return plot1, plot2

# Create animation
animation = FuncAnimation(fig, update, frames=len(times), interval=50, blit=False)

animation.save("2D_extended_collision.gif", writer="pillow")
plt.show()