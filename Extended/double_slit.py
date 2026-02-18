#! /usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse.linalg import spsolve
from matplotlib import cm
from scipy.sparse import csc_matrix
from scipy.integrate import simpson


# Script to solve 2D time-dependent Schrodinger equation numerically

def wavepacket(t, ky):
    '''
        Solving the 2D time-dependent Schrodinger equation using the 
        Crank-Nicolson numerical method.

        Variables:
        t = Time period (s)
        dt = Time step
        dx = Grid spacing
        a = Normalised Gaussian width in x direction
        b = Normalised Gaussian width in y direction
        kx = wave number in x direction
        ky = wave number in y direction

        Outputs:
        psi_n1 = Wave function
        grid_x = Spatial grid in x direction
        grid_y = Spatial grid in y direction

        t = 3.7
        k = 1
    '''
    # Setting parameters
    a = 1
    b = 1
    dt = 0.01
    dx = 0.15
    # Upper limit is given as 10+dx since arange generates a half-open interval
    times = np.arange(0, t+dt, dt)
    grid_x = np.arange(-5, 5+dx, dx)
    grid_y = np.arange(-5, 5+dx, dx)
    Nx = len(grid_x)  # Same as len(grid_y)
    Nt = len(times)
    alpha = dt/(2*dx**2)
    N = int(Nx*Nx)  # To accomodate flattened psi_n
    nn = int(t*1/dt)
    intensity = [0]*Nx
    inten = [0]*Nx
    psi_nt = np.zeros(shape=(nn+1,Nx,Nx))
    
    z = 0
    
    # Making psi_n into a 2D matrix for plotting
    psi_n = np.zeros((Nx, Nx), dtype = complex)
    for i, xi in enumerate(grid_x):
        for j, xj in enumerate(grid_y):
            psi_n[i, j] = (2*a/np.pi)**0.25*(2*b/np.pi)**0.25*np.exp(-a*(xi+3)**2-b*xj**2)*np.exp(1j*ky*(xi+3))  # Initial condition
    psi_n[0, :] = psi_n[-1, :] = 0  # Boundary condition
    psi_n[:, 0] = psi_n[:, -1] = 0  # Boundary condition

    # Converting psi_n into a 1D for calculations
    psi_n = psi_n.flatten()


    # defining vector which contains potential for each space index
    # x_1 = centre of barrier(units of dx)
    # x_2 = width of barrier(units of dx and even number)
    # y_1 = width of barrier in y (units of dx)
    x_1 = np.round(Nx/2.0,0)-1
    x_2 = 16
    y_1 = 16
    offset = 18
    
    # Loops through each space index and inserts corresponding potential
    v = np.zeros((Nx, Nx))
    for i in range(Nx):
        for j in range(Nx):
            if j > x_1 - y_1/2.0 and j < x_1 + y_1/2.0:
                if (i > x_1 - offset and i < x_1 - offset + x_2) or (i > x_1 + offset - x_2 and i < x_1 + offset ):
                    v[j,i] = 0
                else:
                    v[j,i] = 2000000000

    v = v.flatten()

    # Setting up matrix A
    array_A = np.zeros((N,N),dtype=complex)
    for s in range(N):
        for r in range(N):
            if s == r:
                array_A[s,r] = complex(1+1j*alpha+1j*alpha) + 1j*dt*v[r]/2
                #print(array_A[s,r])
            elif s == r + 1 or s == r - 1:
                array_A[s,r] = complex(-1j*alpha/2)
            elif s == r + Nx or s == r - Nx:
                array_A[s,r] = complex(-1j*alpha/2)

    # Setting up matrix 
    array_B = np.zeros((N,N),dtype=complex)
    for n in range(N):
        for m in range(N):
            if n == m:
                array_B[n,m] = complex(1-1j*alpha-1j*alpha) - 1j*dt*v[m]/2
                #print(array_B[n,m])
            elif n == m + 1 or n == m - 1:
                array_B[n,m] = complex(1j*alpha/2)
            elif n == m + Nx or n == m - Nx:
                array_B[n,m] = complex(1j*alpha/2)
        
    psi_n1 = psi_n.copy()
    
    for k in range(Nt):
        vector_b = array_B @ psi_n1
        psi_n1 = spsolve(csc_matrix(array_A),vector_b)
        psi_n1 = psi_n1.reshape((Nx, Nx))
        for counter5 in range(Nx-1):
            intensity[counter5] += abs(psi_n1[Nx-1,counter5])**2
        psi_n1 = psi_n1.flatten()
        for q in range(N):
            if q <= Nx: # or q >= (Nx*(Nx-1)):
                psi_n1[q] = 0
            elif ((q%Nx)==0):
                psi_n1[q] = 0
                psi_n1[q-1] = 0
        if k%((Nt-1)/nn) == 0:
            psi_nt1 = np.copy(psi_n1)
            psi_nt1 = np.abs(psi_nt1.reshape((Nx, Nx)))**2
            psi_nt[z] = psi_nt1
            z += 1
        print(k)
    
    # Converting psi_n1 to 2D for plotting
    psi_n1 = psi_n1.reshape((Nx, Nx))

    for countera in range(Nx):
        psi_n1[Nx-1, countera] = intensity[countera]

    return (psi_n1, psi_nt, grid_x, grid_y, intensity, nn)

inputs = list(map(float, input('Last time, spatial frequency (k): ').split()))

psi_n1, psi_nt, grid_x, grid_y, intensity, noofframes = wavepacket(inputs[0], inputs[1])

fig = plt.figure()
ax = fig.add_subplot(projection='3d')
x, y = np.meshgrid(grid_x, grid_y)

prob_densityx = simpson(np.abs(psi_n1)**2, x=grid_x)
prob_densityy = simpson(np.abs(psi_n1)**2, x=grid_y)

print(f'Normalisation for wave in the x direction = {prob_densityx:.14f}')
print(f'Normalisation for wave in the y direction = {prob_densityy:.14f}')

ax.plot_surface(x, y, np.abs(psi_n1)**2, cmap=cm.coolwarm, linewidth=0)
ax.set_title("Double Slit")
ax.set_xlabel('Position (x)')
ax.set_ylabel('Position (y)')
ax.set_zlabel('|Ψ|²')
ax.set_zlim(0, 0.85)
plt.savefig("double_slit.pdf")


fig2 = plt.figure()
plt.plot(grid_x,intensity)
plt.title("Double Slit Intensity Pattern")
plt.ylabel("Cumulative intensity at the screen")
plt.xlabel("x")
plt.xlim(-4, 4)
plt.show()
plt.savefig('intensity_double_slit.pdf')