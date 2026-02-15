#! /usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse.linalg import spsolve
from scipy.sparse import csc_matrix
from matplotlib.animation import FuncAnimation
from matplotlib import cm
from scipy.integrate import simpson

# Script to solve the 2 particle Schrodinger equation numerically
# Bosonic
def wavepacket_bosonic(t, k1, k2, v0):
    '''
        Solving the 2 particle Schrodinger equation using 
        the Crank-Nicolson numerical method.

        Variables:
        t = Time period (s)
        k1 = 1st wave's wave number
        k2 = 2nd wave's wave number
        v0 = Potential

        Outputs:
        psi_n1 = Wave function
        grid_x = Spatial grid in x direction
        grid_y = Spatial grid in y direction
    '''

    # Setting parameters
    dt = 0.1
    dx = 0.25
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
            psi_n[i, j] = ((2*a/np.pi)**0.5*np.exp(-a*((xi+3)**2+(xj-3)**2))*np.exp(1j*(k1*xi + k2*xj)) + (2*a/np.pi)**0.5*np.exp(-a*((xj+3)**2+(xi-3)**2))*np.exp(1j*(k1*xj + k2*xi)))/(np.sqrt(2))  # Initial condition
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
        psi_n1 = psi_n1.flatten()
    
    return (psi_frames, grid_x1, grid_x2)

# Fermionic
def wavepacket_fermionic(t, k1, k2, v0):
    '''
        Solving the 2 particle Schrodinger equation using 
        the Crank-Nicolson numerical method.

        Variables:
        t = Time period (s)
        k1 = 1st wave's wave number
        k2 = 2nd wave's wave number
        v0 = Potential

        Outputs:
        psi_n1 = Wave function
        grid_x = Spatial grid in x direction
        grid_y = Spatial grid in y direction
    '''

    # Setting parameters
    dt = 0.1
    dx = 0.25
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
        psi_n1 = psi_n1.flatten()
    
    return (psi_frames, grid_x1, grid_x2)

# User input
T = float(input("Enter time period t: "))
k1, k2, v0 = map(float, input("Enter k1, k2, v0: ").split())
times = np.arange(0, T+0.1, 0.1)

psi_frames_b, grid_x1_b, grid_x2_b = wavepacket_bosonic(T, k1, k2, v0)
psi_frames_f, grid_x1_f, grid_x2_f = wavepacket_fermionic(T, k1, k2, v0)

# Plotting contour graph (bosonic) -------------------------
# Setup figure
fig, ax = plt.subplots()
x1_b, x2_b = np.meshgrid(grid_x1_b, grid_x2_b)
plt.plot(grid_x1_b, grid_x1_b, '--')
ax.set_xlabel('x1')
ax.set_ylabel('x2')

# Initial frame
contour_b = ax.contourf(x1_b, x2_b, psi_frames_b[0], cmap=cm.coolwarm)

# Update function for animation
def update(frame):
    global contour_b
    contour_b.remove()
    contour_b = ax.contourf(x1_b, x2_b, np.abs(psi_frames_b[frame])**2, cmap=cm.coolwarm, levels=50)
    ax.set_title(f"t = {times[frame]:.3f}")
    return contour_b

# Create animation
animation = FuncAnimation(fig, update, frames=len(times), interval=50, blit=False)

animation.save("bosonic_contour_collision.gif", writer="pillow", fps=15)
plt.show()

# Plotting contour graph (fermionic) -------------------------
# Setup figure
fig, ax = plt.subplots()
x1_f, x2_f = np.meshgrid(grid_x1_f, grid_x2_f)
plt.plot(grid_x1_f, grid_x1_f, '--')
ax.set_xlabel('x1')
ax.set_ylabel('x2')

# Initial frame
contour_f = ax.contourf(x1_f, x2_f, psi_frames_f[0], cmap=cm.coolwarm)

# Update function for animation
def update(frame):
    global contour_f
    contour_f.remove()
    contour_f = ax.contourf(x1_f, x2_f, np.abs(psi_frames_f[frame])**2, cmap=cm.coolwarm, levels=50)
    ax.set_title(f"t = {times[frame]:.3f}")
    return contour_f

# Create animation
animation = FuncAnimation(fig, update, frames=len(times), interval=50, blit=False)

animation.save("fermionic_contour_collision.gif", writer="pillow", fps=15)
plt.show()

# Plotting 2D graph -----------------------------
# Setup figure
fig, ax = plt.subplots()
ax.set_xlabel('Position (x)')
ax.set_ylabel(r'$(|\Psi|)^2$')
ax.set_ylim(0, 0.85)

# Separate into 2 waves by marginalisation
# i.e. integrating over probability density with respect to the other wave
wave1_b = simpson(psi_frames_b[0], x=grid_x2_b, axis=1)
wave2_b = simpson(psi_frames_b[0], x=grid_x1_b, axis=0)

wave1_f = simpson(psi_frames_f[0], x=grid_x2_f, axis=1)
wave2_f = simpson(psi_frames_f[0], x=grid_x1_f, axis=0)

# Initial frame
plot1_b = ax.plot(grid_x1_b, wave1_b, label='1st wave bosonic')[0]
plot2_b = ax.plot(grid_x2_b, wave2_b, label='2nd wave bosonic')[0]

plot1_f = ax.plot(grid_x1_f, wave1_f, label='1st wave fermionic')[0]
plot2_f = ax.plot(grid_x2_f, wave2_f, label='2nd wave fermionic')[0]

ax.legend()

# Update function for animation
def update(frame):
    wave1_b = simpson(psi_frames_b[frame], x=grid_x2_b, axis=1)
    wave2_b = simpson(psi_frames_b[frame], x=grid_x1_b, axis=0)
    wave1_f = simpson(psi_frames_f[frame], x=grid_x2_f, axis=1)
    wave2_f = simpson(psi_frames_f[frame], x=grid_x1_f, axis=0)
    plot1_b.set_ydata(wave1_b)
    plot2_b.set_ydata(wave2_b)
    plot1_f.set_ydata(wave1_f)
    plot2_f.set_ydata(wave2_f)
    ax.set_title(f"t = {times[frame]:.3f}")
    return plot1_b, plot2_b, plot1_f, plot2_f

# Create animation
animation = FuncAnimation(fig, update, frames=len(times), interval=50, blit=False)

animation.save("fb_2D_collision.gif", writer="pillow")
plt.show()