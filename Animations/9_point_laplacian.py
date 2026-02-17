#! /usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from scipy.sparse import lil_matrix, eye
from scipy.sparse.linalg import splu

# Code to solve the 2D Schrödinger equation using a 9-point Laplacian stencil and a Crank-Nicolson time evolution.

# Parameters
dx = 0.10
dt = 0.1
tmax = 50.0

xmin, xmax = -5, 5
x = np.arange(xmin, xmax + dx, dx)
Nx = len(x)
N = Nx * Nx
Nt = int(tmax / dt)

# Grid and Indexing
def idx(i, j):
    return i * Nx + j

# Initial wave packet
X, Y = np.meshgrid(x, x, indexing="ij")
psi = np.exp(-(X**2 + Y**2)) * np.exp(1j * (2.5*X + 1.5*Y))

# Hard-wall boundaries
psi[0, :] = psi[-1, :] = psi[:, 0] = psi[:, -1] = 0
psi = psi.ravel()

# Nine-point Laplacian
L = lil_matrix((N, N), dtype = float)

for i in range(1, Nx-1):
    for j in range(1, Nx-1):
        k = idx(i, j)

        c = 1.0 / (6.0 * dx * dx)
        L[k, k] = -20 * c

        # Axial neighbours
        for di, dj, w in [(1,0,4), (-1,0,4), (0,1,4), (0,-1,4)]:
            L[k, idx(i+di, j+dj)] = w * c

        # Diagonal neighbours
        for di, dj in [(1,1), (1,-1), (-1,1), (-1,-1)]:
            L[k, idx(i+di, j+dj)] = c

# Only store non-zero values column-wise
L = L.tocsc()

# Construct the Hamiltonian and crank Nicolson matrices
H = -0.5 * L
I = eye(N, format="csc")

A = (I + 1j * dt * 0.5 * H).tocsc()
B = (I - 1j * dt * 0.5 * H).tocsc()

A_solver = splu(A)

frames = []
probability = []

# Crank-Nicolson time stepping loop
for n in range(Nt):
    psi = A_solver.solve(B @ psi)

    # Enforce hard wall bounrdaries
    psi[:Nx] = psi[-Nx:] = 0
    psi[::Nx] = psi[Nx-1::Nx] = 0

    frames.append(np.abs(psi.reshape(Nx, Nx)**2))
    probability.append(np.sum(np.abs(psi)**2) * dx * dx)

print(f"Probability drift: {max(probability) - min(probability):.3e}")

# Animation
fig, ax = plt.subplots()
global_max = max(np.max(f) for f in frames)
im = ax.imshow(frames[0], extent=[xmin, xmax, xmin, xmax], origin="lower", cmap="viridis", vmin=0, vmax=np.max(frames[0]))

ax.set_title("2D Schrödinger Equation using 9-point Laplacian")
ax.set_xlabel("x")
ax.set_ylabel("y")
cbar = plt.colorbar(im)
cbar.set_label(r"$|\psi|^2$")

def update(frame):
    im.set_data(frames[frame])
    ax.set_title(f"t = {frame * dt:.2f}")
    return im,

ani = FuncAnimation(fig, update, frames=len(frames), interval=40)
ani.save("9_point_laplacian.gif", writer="pillow", fps=50)
plt.show()