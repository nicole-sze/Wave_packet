#! /usr/bin/env python3
import matplotlib.pyplot as plt
import numpy as np
import scipy as sc
from matplotlib.animation import FuncAnimation
import math

# Constants
omega = 2.0
a = 1
n = 0
x = np.linspace(-5, 5, 1000)
dx = x[1] - x[0]
dt = 0.1  # Time step

# Energy for level n
E = omega * (n + 0.5)

# Spatial part
H_n = sc.special.eval_hermite(n, np.sqrt(omega) * x)
psi_x = (1/np.sqrt(2**n * math.factorial(n)) * (omega/np.pi)**(0.25) * np.exp(-0.5 * omega * x**2) * H_n)

norm = np.sqrt(np.sum(np.abs(psi_x)**2) * dx)
psi_x = psi_x / norm

# 2D plot with 9 different time periods
t = [1, 2, 3, 4, 5, 6, 7, 8, 9]

for m in range(9):
    psi_t = psi_x * np.cos(E * t[m])
    prob_density = np.abs(psi_t)**2
    plt.subplot(3, 3, m+1)
    plt.plot(x, np.abs(prob_density)**2)
    plt.xlabel('Position (x)')
    plt.ylabel('(|ψ|^2)')
    plt.ylim([0, 0.85])
    plt.title('t ='+str(round(t[m], 3)))

plt.tight_layout()
plt.savefig('harmonic_analytical.pdf')
plt.show()