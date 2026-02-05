#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse.linalg import spsolve
from scipy.sparse import csc_matrix

def wavepacket(t, ky0):
    '''
        Simulating a single slit experiment using 
        the Crank-Nicolson numerical method.

        Variables:
        t = Time period (s)
        ky = Wave number in y direction (because does not move in x-direction)

        Outputs:
        psi_n1 = Wave function
        grid_x = Spatial grid in x direction
        grid_y = Spatial grid in y direction
    '''
    # Setting up parameters
    dt = 0.01
    dx = 0.15
    a = b = 1
    
    # Grids
    times = np.arange(0, t + dt, dt)
    grid_x = np.arange(-5, 5 + dx, dx)
    grid_y = np.arange(-5, 5 + dx, dx)
    Nx = len(grid_x)
    Nt = len(times)

    # Initial wavepacket
    psi_n = np.zeros((Nx, Nx), dtype=complex)
    for i, x in enumerate(grid_x):
        for j, y in enumerate(grid_y):
            psi_n[i, j] = (2*a/np.pi)**0.25 * (2*b/np.pi)**0.25 \
                * np.exp(-a*(x+4)**2 - b*y**2) \
                * np.exp(1j * ky0 * (x+4))

    # hard walls
    psi_n[0, :] = psi_n[-1, :] = 0
    psi_n[:, 0] = psi_n[:, -1] = 0

    psi_n = psi_n.flatten()

    # Constants 
    alpha = dt / (2 * dx**2)
    N = Nx * Nx

    # Symmetric single-slit potential
    v = np.zeros((Nx, Nx))
    centre = Nx // 2

    slit_half_width  = 4   # controls angular width
    slit_half_height = 4   # wall thickness

    for i in range(Nx):
        for j in range(Nx):
            if abs(j - centre) <= slit_half_width:
                if abs(i - centre) <= slit_half_height:
                    v[j, i] = 0.0
                else:
                    v[j, i] = 2e9

    v = v.flatten()

    # Crank–Nicolson matrices
    A = np.zeros((N, N), dtype=complex)
    B = np.zeros((N, N), dtype=complex)

    for r in range(N):
        A[r, r] = 1 + 2j*alpha + 1j*dt*v[r]/2
        B[r, r] = 1 - 2j*alpha - 1j*dt*v[r]/2

        for s in (r-1, r+1, r-Nx, r+Nx):
            if 0 <= s < N:
                A[r, s] = -1j * alpha / 2
                B[r, s] =  1j * alpha / 2

    # Time evolution
    psi = psi_n.copy()
    A = csc_matrix(A)

    for _ in range(Nt):
        psi = spsolve(A, B @ psi)

    psi = psi.reshape((Nx, Nx))

    # Far-field scattering analysis
    # Isolated transmitted wave
    x_cut = centre + 15
    psi_trans = psi[x_cut:, :]

    # Fourier transform to momentum space
    fft_psi = np.fft.fftshift(np.fft.fft2(psi_trans))
    P = np.abs(fft_psi)**2     # Momentum space probability density

    # Construct the momentum axes
    kx_vals = 2*np.pi * np.fft.fftshift(
        np.fft.fftfreq(psi_trans.shape[0], dx)
    )
    ky_vals = 2*np.pi * np.fft.fftshift(
        np.fft.fftfreq(psi_trans.shape[1], dx)
    )
    kx, ky = np.meshgrid(kx_vals, ky_vals, indexing='ij')

    # Convert momentum to scattering angle
    theta = np.degrees(np.arctan2(ky, kx))
    k = np.sqrt(kx**2 + ky**2)

    return theta, P, k


# Main block

inputs = list(map(float, input(
    "t, ky: "
).split()))

theta, P, k = wavepacket(*inputs)

# Ensure that we keep only forward moving particles
mask = (np.cos(np.radians(theta)) > 0)

theta_vals = theta[mask]
weights = (P * k)[mask]     # Jacobian weighted probability

# Angular Integration using histogram 
theta_bins = np.linspace(-90, 90, 200)
prob_dist, bin_edges = np.histogram(theta_vals, bins=theta_bins, weights=weights, density=True)

bin_centers= 0.5 * (bin_edges[:-1] + bin_edges[1:])

# Plot angular probability density
plt.plot(bin_centers, prob_dist)
plt.xlabel(r"Scattering angle $\theta$ (degrees)")
plt.ylabel(r"$P(\theta)$")
plt.title("Single-slit diffraction pattern")
plt.grid()
plt.savefig("angle.pdf")
plt.show()