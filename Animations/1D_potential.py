#! /usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

def wavepacket(t, dt, dx, a, k, v0):
    times = np.arange(0, t+dt, dt)
    grid = np.arange(-5, 5+dx, dx)
    Nx = len(grid)
    Nt = len(times)

    interior_Nx = Nx - 2
    alpha = dt/(2*dx**2)

    p = 37
    b = 8

    v = np.zeros(interior_Nx)
    for i in range(interior_Nx):
        if i < p - b/2 or i > p + b/2:
            v[i] = 0
        else:
            v[i] = v0

    lowerA = np.full(interior_Nx-1, -1j*alpha/2, dtype=complex)
    diagA = np.full(interior_Nx, 1+1j*alpha, dtype=complex) + 1j*dt*v/2
    upperA = np.full(interior_Nx-1, -1j*alpha/2, dtype=complex)

    lowerB = np.full(interior_Nx-1, 1j*alpha/2, dtype=complex)
    diagB = np.full(interior_Nx, 1-1j*alpha, dtype=complex) - 1j*dt*v/2
    upperB = np.full(interior_Nx-1, 1j*alpha/2, dtype=complex)

    psi_n = (2*a/np.pi)**0.25*np.exp(-a*grid**2)*np.exp(1j*k*grid)
    psi_n[0] = psi_n[-1] = 0

    psi_time = np.zeros((Nt, Nx), dtype=complex)
    psi_time[0] = psi_n.copy()

    for i in range(1, Nt):
        rhs = diagB*psi_n[1:-1]
        rhs[1:] += lowerB*psi_n[1:-2]
        rhs[:-1] += upperB*psi_n[2:-1]

        A = lowerA.copy()
        B = diagA.copy()
        C = upperA.copy()

        for j in range(1, interior_Nx):
            ratio = A[j-1]/B[j-1]
            B[j] -= ratio*C[j-1]
            rhs[j] -= ratio*rhs[j-1]

        interior_psi = np.zeros(interior_Nx, dtype=complex)
        interior_psi[-1] = rhs[-1]/B[-1]
        for n in range(interior_Nx-2, -1, -1):
            interior_psi[n] = (rhs[n] - C[n]*interior_psi[n+1])/B[n]

        psi_n1 = np.zeros(Nx, dtype=complex)
        psi_n1[0] = psi_n1[-1] = 0
        psi_n1[1:-1] = interior_psi
        psi_n = psi_n1.copy()
        psi_time[i] = psi_n

    return grid, psi_time, p, b, dx, times

# User input
T = float(input("Enter time period t: "))
dt, dx, a, k, v0 = map(float, input("Enter dt, dx, a, k, v0: ").split())

grid, psi_time, p, b, dx, times = wavepacket(T, dt, dx, a, k, v0)

# Plot setup
fig, axis = plt.subplots()

axis.set_xlim([grid.min(), grid.max()])
axis.set_ylim([0, np.max(np.abs(psi_time)**2) * 1.1])

animated_plot, = axis.plot([], [])

# Draw barrier (fixed position)
rect = plt.Rectangle(((p-b/2)*dx-5, 0), b*dx, np.max(np.abs(psi_time)**2) * 1.1,
                     edgecolor='orange', facecolor='none')
axis.add_patch(rect)

def update_data(frame):
    y = np.abs(psi_time[frame])**2
    animated_plot.set_data(grid, y)
    axis.set_title(f"t = {times[frame]:.3f}")
    return animated_plot,

animation = FuncAnimation(fig=fig, func=update_data, frames=len(times), interval=50)

animation.save("1D_potential.gif")
plt.show()