#! /usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import simpson

# Script to solve 1D time-dependent Schrodinger equation numerically with potential barrier
def wavepacket(t, k, v0):
    '''
        Solving the 1D time-dependent Schrodinger equation with potential using the 
        Crank-Nicolson numerical method. This results in a large, complex matrix which 
        is then computed with the Thomas Algorithm.
        
        Variables:
        t = Time period (s)
        k = Wave number
        v0 = Potential
        
        Outputs:
        grid = Spatial grid
        psi_n1 = Wave function
        p = Centre of potential barrier (units of dx)
        b = width of potential barrier (units of dx)
        dx = Grid spacing
    '''
    # Setting parameters
    
    # Setting values for space and time step
    dt = 0.001
    dx = 0.15
    a = 1
	
	# Upper limit is given as 5+dx since arange generates a half-open interval
    times = np.arange(0, t+dt, dt)
    grid = np.arange(-5, 5+dx, dx)
    Nx = len(grid)
    Nt = len(times)
	
    interior_Nx = Nx-2
    alpha = dt/(2*dx**2)
   
	# defining vector which contains potential for each space index
	# p = centre of barrier(units of dx)
	# b = width of barrier(units of dx and even number)
    p = 37
    b= 8
    
    # Loops through each space index and inserts corresponding potential
    v = np.zeros(interior_Nx)
    for i in range(interior_Nx):
        if i < p - b/2 or i > p + b/2:
            v[i] = 0
        else:
            v[i] = v0
			
	# Setting values for matrix_a
    lowerA = np.full(interior_Nx-1, -1j*alpha/2, dtype=complex)
    diagA = np.full(interior_Nx, 1+1j*alpha, dtype=complex) + 1j*dt*v/2
    upperA = np.full(interior_Nx-1, -1j*alpha/2, dtype=complex)
	
    # Setting values for matrix_b
    lowerB = np.full(interior_Nx-1, 1j*alpha/2, dtype=complex)
    diagB = np.full(interior_Nx, 1-1j*alpha, dtype=complex) - 1j*dt*v/2
    upperB = np.full(interior_Nx-1, 1j*alpha/2, dtype=complex)   
	
    psi_n = (2*a/np.pi)**0.25*np.exp(-a*(grid+3)**2)*np.exp(1j*k*(grid+3))  # Initial condition
    psi_n[0] = psi_n[-1] = 0  # Boundary condition

    for i in range(Nt):
        psi_n1 = np.zeros(Nx, dtype=complex)
        rhs = diagB*psi_n[1:-1]
        rhs[1:] += lowerB*psi_n[1:-2]
        rhs[:-1] += upperB*psi_n[2:-1]
		
    	# To reset matices every time
        A = lowerA.copy()
        B = diagA.copy()
        C = upperA.copy()
    	
    	# Performing the Thomas Algorithm
    	# Forward elimination
        for j in range(1, interior_Nx):
            ratio = A[j-1]/B[j-1]
            B[j] = B[j] - ratio*C[j-1]
            rhs[j] = rhs[j] - ratio*rhs[j-1]
    		
    		# Saftey check:
            if abs(B[j-1]) < 1e-12:
                print("Zero pivot encountered — Thomas algorithm fails.")
    			
    		# Back substitution
            interior_psi = np.zeros(interior_Nx, dtype=complex)
            interior_psi[-1] = rhs[-1]/B[-1]  # Computes last unknown
            for n in range(interior_Nx-2, -1, -1):
                interior_psi[n] = (rhs[n] - C[n]*interior_psi[n+1])/B[n]
                
            psi_n1[0] = psi_n1[-1] = 0
            psi_n1[1:-1] = interior_psi
            psi_n = psi_n1.copy()
        
    return(grid, psi_n1, p, b, dx)

T_values = list(map(float, input('Enter 9 time periods (t): ').split()))
inputs = list(map(float, input('Enter wave number (k) and barrier potential (v0): ').split()))

# 2D plot with 9 different time periods
fig = plt. figure()
fig.suptitle('1D probability density with a potential barrier')
for m in range(len(T_values)):
	grid, psi_n1, p, b, dx = wavepacket(T_values[m], inputs[0], inputs[1])
	
	prob_density = simpson(np.abs(psi_n1)**2, x=grid)
	print('Normalisation (t ='+str(round(T_values[m], 3))+f'): {prob_density:.14f}')
	
	plt.subplot(3, 3, m+1)
	plt.plot(grid, np.abs(psi_n1)**2)
	ax = plt.gca()
	rect1 = plt.Rectangle(((p-b/2)*dx-5, 0), b*dx, 1, edgecolor='orange', facecolor='none')
	ax.add_patch(rect1)
	plt.xlabel('Position (x)')
	plt.ylabel(r'$(|\Psi|)^2$')
	plt.title('t ='+str(round(T_values[m], 3)))

plt.tight_layout()
plt.savefig('1D_potential.pdf')
plt.show()