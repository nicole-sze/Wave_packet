#! /usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse.linalg import spsolve
from matplotlib import cm
from scipy.sparse import csc_matrix
from matplotlib.animation import FuncAnimation


# Schrödinger 2D single slit - Animated

def wavepacket(total_time, dt, dx, a, b, ky):
    """
    Solving 2D time-dependent Schrödinger equation using Crank-Nicolson
    Returns: psi_frames, grid_x, grid_y
    """
    # Time and spatial grids
    times = np.arange(0, total_time+dt, dt)
    grid_x = np.arange(-5, 5+dx, dx)
    grid_y = np.arange(-5, 5+dx, dx)
    Nx = len(grid_x)
    Nt = len(times)
    N = Nx * Nx

    # Initial wavefunction
    psi_n = np.zeros((Nx, Nx), dtype=complex)
    for i, xi in enumerate(grid_x):
        for j, xj in enumerate(grid_y):
            psi_n[i,j] = (2*a/np.pi)**0.25 * (2*b/np.pi)**0.25 * np.exp(-a*(xi+2)**2 - b*xj**2) * np.exp(1j*ky*(xi+2))
    psi_n[0,:] = psi_n[-1,:] = 0
    psi_n[:,0] = psi_n[:,-1] = 0
    psi_n = psi_n.flatten()

    # Potential for single slit
    x_1 = np.round(Nx/2.0)
    x_2 = 8
    y_1 = 8
    v = np.zeros((Nx, Nx))
    for i in range(Nx):
        for j in range(Nx):
            if j > x_1 - y_1/2.0 and j < x_1 + y_1/2.0:
                if i > x_1 - x_2/2.0 and i < x_1 + x_2/2.0:
                    v[j,i] = 0
                else:
                    v[j,i] = 2e9
    v = v.flatten()

    # Crank-Nicolson matrices
    alpha = dt/(2*dx**2)
    array_A = np.zeros((N,N),dtype=complex)
    array_B = np.zeros((N,N),dtype=complex)
    for s in range(N):
        for r in range(N):
            if s == r:
                array_A[s,r] = 1 + 2j*alpha + 1j*dt*v[r]/2
                array_B[s,r] = 1 - 2j*alpha - 1j*dt*v[r]/2
            elif s == r+1 or s == r-1 or s == r+Nx or s == r-Nx:
                array_A[s,r] = -1j*alpha/2
                array_B[s,r] = 1j*alpha/2

    # Time evolution
    psi_frames = []
    psi_n1 = psi_n.copy()
    for k in range(Nt):
        vector_b = array_B @ psi_n1
        psi_n1 = spsolve(csc_matrix(array_A), vector_b)
        psi_2d = psi_n1.reshape((Nx, Nx))
        psi_frames.append(np.abs(psi_2d)**2)

    return np.array(psi_frames), grid_x, grid_y


# User input / parameters
inputs = list(map(float, input(
    "Enter: total_time dt dx a b ky (space-separated): "
).split()))

total_time = inputs[0]
dt = inputs[1]
dx = inputs[2]
a = inputs[3]
b = inputs[4]
ky = inputs[5]

# Precompute wavefunction frames
psi_frames, grid_x, grid_y = wavepacket(total_time, dt, dx, a, b, ky)
times = np.arange(0, total_time+dt, dt)

# Setup figure
fig = plt.figure()
ax = fig.add_subplot(projection='3d')

x, y = np.meshgrid(grid_x, grid_y)

ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('|ψ|²')
ax.set_zlim(0, 0.85)

# Initial surface
surface = ax.plot_surface(x, y, psi_frames[0], cmap=cm.coolwarm, linewidth=0)

# Update function for animation
def update(frame):
    global surface
    surface.remove()
    surface = ax.plot_surface(x, y, psi_frames[frame], cmap=cm.coolwarm, linewidth=0)
    ax.set_title(f"t = {times[frame]:.3f}")
    return surface,

# Create animation
ani = FuncAnimation(fig, update, frames=len(times), interval=500, blit=False)
ani.save("single_slit.gif", writer="pillow", fps=15)

plt.show()