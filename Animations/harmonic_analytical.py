#! /usr/bin/env python3
import matplotlib.pyplot as plt
import numpy as np
import scipy as sc
from matplotlib.animation import FuncAnimation
import math

# Analytical solution for quantum harmonic oscillatorspring:  4
# Units: h_bar = 1, m = 1

omega = 2.0
n = 0
x = np.linspace(-10, 10, 1000)
dx = x[1] - x[0]
dt = 0.1  # Time step

# Energy for level n
E = omega * (n + 0.5)

# Spatial part
H_n = sc.special.eval_hermite(n, np.sqrt(omega) * x)
psi_x = (1/np.sqrt(2**n * math.factorial(n)) * (omega/np.pi)**(0.25) * np.exp(-0.5 * omega * x**2) * H_n)

norm = np.sqrt(np.sum(np.abs(psi_x)**2) * dx)
psi_x = psi_x / norm

# Setup the figure
fig, ax = plt.subplots()
line, = ax.plot(x, psi_x)
ax.set_ylim(0, 1)
ax.set_title(f'Probability Density $|\psi_{n}(x,t)|^2$')
ax.set_xlabel('Position (x)')
ax.set_ylabel('Probability')

def update(t):
    psi_t = psi_x * np.exp(-1j * E * t)
    prob_density = np.abs(psi_t)**2
    line.set_ydata(prob_density)
    ax.set_title(f"Probability Density $|\psi_{n}(x,t)|^2$, t = {t:.3f}")
    return line,

# Create animation
ani = FuncAnimation(fig, update, frames=np.arange(0, 10, dt), interval=30)
ani.save("harmonic_analytical.gif", writer="pillow", fps=15)
plt.grid(True)