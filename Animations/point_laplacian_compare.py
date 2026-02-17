#!/usr/bin/env python3

"""
Comparison script for 5-point vs 9-point Laplacian stencils
Solves 2D Schrödinger equation with both methods and visualizes differences
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import lil_matrix, eye
from scipy.sparse.linalg import splu

# Paramters
dx = 0.10
dt = 0.01
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
psi_init = np.exp(-(X**2 + Y**2)) * np.exp(1j * (2.5*X + 1.5*Y))

# Hard-wall boundaries
psi_init[0, :] = psi_init[-1, :] = psi_init[:, 0] = psi_init[:, -1] = 0
psi_init = psi_init.ravel()


# 5-Point Laplacian
L_5 = lil_matrix((N, N), dtype=float)

for i in range(1, Nx-1):
    for j in range(1, Nx-1):
        k = idx(i, j)
        c = 1.0 / (dx * dx)
        L_5[k, k] = -4 * c
        for di, dj in [(1,0), (-1,0), (0,1), (0,-1)]:
            L_5[k, idx(i+di, j+dj)] = c

L_5 = L_5.tocsc()


# 9-Point Laplacian
L_9 = lil_matrix((N, N), dtype=float)

for i in range(1, Nx-1):
    for j in range(1, Nx-1):
        k = idx(i, j)
        c = 1.0 / (6.0 * dx * dx)
        L_9[k, k] = -20 * c

        # Axial neighbors
        for di, dj, w in [(1,0,4), (-1,0,4), (0,1,4), (0,-1,4)]:
            L_9[k, idx(i+di, j+dj)] = w * c

        # Diagonal neighbors
        for di, dj in [(1,1), (1,-1), (-1,1), (-1,-1)]:
            L_9[k, idx(i+di, j+dj)] = c

L_9 = L_9.tocsc()

# Hamiltonian and Crank-Nicolson Matrices (5-point)
H_5 = -0.5 * L_5
I = eye(N, format="csc")

A_5 = (I + 1j * dt * 0.5 * H_5).tocsc()
B_5 = (I - 1j * dt * 0.5 * H_5).tocsc()
A_5_solver = splu(A_5)

# Hamiltonian and Crank-Nicolson Matrices (9-point)
H_9 = -0.5 * L_9

A_9 = (I + 1j * dt * 0.5 * H_9).tocsc()
B_9 = (I - 1j * dt * 0.5 * H_9).tocsc()
A_9_solver = splu(A_9)

# Time-stepping loops
psi_5 = psi_init.copy()
psi_9 = psi_init.copy()

error_norms = []
times = []

for n in range(Nt):
    # 5-point update
    psi_5 = A_5_solver.solve(B_5 @ psi_5)
    psi_5[:Nx] = psi_5[-Nx:] = 0
    psi_5[::Nx] = psi_5[Nx-1::Nx] = 0
   
    # 9-point update
    psi_9 = A_9_solver.solve(B_9 @ psi_9)
    psi_9[:Nx] = psi_9[-Nx:] = 0
    psi_9[::Nx] = psi_9[Nx-1::Nx] = 0
   
    # Compute error metric (L2 norm of difference)
    prob_5 = np.abs(psi_5.reshape(Nx, Nx))**2
    prob_9 = np.abs(psi_9.reshape(Nx, Nx))**2
    diff = np.abs(prob_5 - prob_9)
   
    error_norms.append(np.sqrt(np.sum(diff**2)) * dx * dx)
    times.append(n * dt)


# Error growth plot
fig, ax = plt.subplots(figsize=(10, 6))
ax.semilogy(times, error_norms, 'b-', linewidth=2.5)     # Set log-plot for y-axis and keep x-axis linear
ax.set_xlabel("Time", fontsize=12)
ax.set_ylabel("L2-norm of Difference", fontsize=12)
ax.set_title("Error Growth: 5-Point vs 9-Point Stencil", fontsize=14)
ax.grid(True, which="both", alpha=0.3)
plt.tight_layout()
plt.savefig('point_laplacian_compare.pdf', dpi=100, bbox_inches='tight')

print(f"Final error (L2-norm): {error_norms[-1]:.6e}")
print(f"Maximum error: {max(error_norms):.6e}")
print(f"Average error: {np.mean(error_norms):.6e}")
print(f"\nError growth rate (final/initial): {error_norms[-1]/error_norms[0]:.2e}x")
plt.show()