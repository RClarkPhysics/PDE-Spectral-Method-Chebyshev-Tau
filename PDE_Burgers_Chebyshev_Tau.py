#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 26 11:13:46 2024

@author: randallclark
"""

"""
The goal of this script is to solve the Burgers equation using Spectral Methods, specifically the chebyshev tau method with the boundary conditions
where u(-1,t) = u(1,t) = 0

The Burgers equation:
∂u/∂t + u∂u/∂x - v∂^2u/∂x^2 = 0

To solve for u(t) numerically, we expand u(t) into a chebyshev polynomial expansion. This expansion is a sum of time dependent coefficeints times
the chebyshev polynomials, which form an orthogonal basis.
uN(t) = SUM_{k=0 to N} u_k(t) * T(_kx)

We must take advantage of the orthogonality equation:
    INTEGRAL_{from -1 to +1} T_k(x)T_k'(x)dx/(1-x^2)^1/2 = c_k*pi/2 if k'=k, 0 else
    c_k = 2 if k=0, 1 else
    
Starting with the initial conditions, we multiply by  T_k(x)*(1-x^2)^1/2 and integrate both sides of the uN(t) equation:
    u_k(t=0) = 2/(pi*c_k)  INT_{-1 to +1} u(x,t=0) * T_k(x)dx/(1-x^2)^1/2
    
A careful choice of letting u(x,t=0) = 1-x^2 = 1/2*T_0(x) - 1/2*T_2(x), giving us our coefficients (all other coeffients other than k=0,2 are 0)

Plugging our uN definition into the Burgers equation then multiply by  T_k(x)*(1-x^2)^1/2 and integrate both sides we are able to single out an equation
for each k, but note that we drop last two (largest k terms)to enforce the boundary equation.

The boundary equation is enforced simply by (Use the last two u_k's to enforce these terms):
    SUM_{k=0 to N} u_k = 0
    SUM_{k=0 to N} (-1)^k * u_k = 0

The remainig N-1 equations come from the Burger's equation, plugging in the UN equation, and integrating over orthogonality equation gives these equations:
du_k/dt = v*u_k^(2) - SUM_i SUM_j u_i*u_j*1/2*C_ijk

u_k^(2) is a special term that refers to the 2nd derivative of the u function coefficient and is easily calculated in chebyshev literature using the
following equation:
    u_k^(2) = 1/c_k SUM_{p = k+2 to N for even p+k} p(p^2-k^2)u_p
    
C_ijk is a 3D matrix with values 1 if i+j=k or |i-j|=k, 2 if both conditions are true, or 0 if none of the conditions are true

Using scipy ODEINT, we integrate the calcualted du_k/dt terms, then add them all together according to the uN equation to get back our
approximate solution to u(x,t)
"""


import numpy as np
from scipy.integrate import odeint

class PDE_Burgers():
    '''
    Generate Burgers data on convection/diffusion flow
    '''
    def __init__(self, N=20,v=1):
        """Initialize the PDE_Burgers object
        Args:
            N (int)         : Cuttoff number for the spectral expansion order (k will range from -N to +N)
            v (float)       : The diffusion term
        """
        self.N = N
        self.v = v
        self.C = self.calc_C()
        self.Ini = self.InitialCondition()
    
    """
    -------------------------------------User Tools--------------------------------------------------------------------------------------
    """
    def Generate_u(self,t):
        Solver = odeint(self.integrate_uk,self.Ini,t)
        return Solver
    
    #Given a set of u_k values at time t, build u(x)
    def build_uN(self,u,N):
        #First calculate, the two missing dimensions that don't get integrated
        u_full = np.zeros((self.N+1))
        u_full[0:self.N-1] = u
        u_full[self.N-1:] = self.u_Boundary_Condition(u)
        
        u_poly = np.polynomial.chebyshev.Chebyshev(u_full)
        return u_poly
    
    def sample_uN(self,uN,L):
        Y = np.zeros(L)
        X = np.linspace(-1,1,L)
        for i in range(L):
            Y[i] = uN(X[i])
        return Y

    """
    -------------------------------------Tool Code--------------------------------------------------------------------------------------
    """
    #Our initial condition is easy, it is simply a linear combination of
    #two Chebyshev polynomials 1-x^2 = 1/2*T_0(x) - 1/2*T_2(x)
    def InitialCondition(self):
        #Remember to drop the last two dimensions for the sake of the integration
        #N must be greater than 2 and even
        Ini = np.zeros((self.N-1))
        Ini[0] = 1/2
        Ini[2] = -1/2
        return Ini

    """
    -------------------------------------Integration Code--------------------------------------------------------------------------------------
    """
    def integrate_uk(self,u,t):
        #The integration follows the following format:
        #For k = 0,1,..,N-2:
        # du_k/dt = -SUM_i(SUM_j(u_i*u_j*C_ijk*1/2)) + v*u^(2)_k
        #C is 1 if i+j = k or |i-j| = k, else it is 0
        
        #See the 2nd derivative function to see how u^(2) is calculated
        
        #for k = N-1
        #u_{N-1} = SUM_{k odd}u_k/2
        #for k = N
        #u_{N} = SUM_{k even}u_k/2
        
        #This means, the u integrate_uk imports is missing the last two dimensions, we'll add those in first
        u_full = np.zeros((self.N+1))
        u_full[0:self.N-1] = u
        u_full[self.N-1:] = self.u_Boundary_Condition(u)
        
        dukdt = []
        
        for k in range(self.N-1):
            dukdt.append(self.calc_dukdt(u_full,k))
        return dukdt

    def calc_C(self):
        #Return the C matrix that is a N+1 by N+1 by N+1 matrix
        #The C matrix has the following rules, for row i, column j, and kth third dimension
        #If i+j = k or |i-j| = k, then the value of the matrix is 1, else 0
        C = np.zeros((self.N+1,self.N+1,self.N+1))
        for i in range(self.N+1):
            for j in range(self.N+1):
                for k in range(self.N+1):
                    if i + j == k:
                        C[i,j,k] = 1
                    if abs(i-j) == k:
                        C[i,j,k] += 1
        return C    
    
    def u_Boundary_Condition(self,u):
        #This u is assumed to be from k=0 to k = N-2
        #This also implicitly requires N to be even
        a = 0
        b = 0
        for k in range(u.shape[0]):
            if k%2 == 0: #even
                b += u[k]
            if k%2 == 1: #odd
                a += u[k]
        return -a, -b
    
    def calc_dukdt(self,u,k):
        #dukdt is a sum of two terms, the term with two summations of C, and the term with the 2nd derivative
        #Improvement for the future: This is wasteful programming to loop over all possible terms instead of
        #Iterating only over terms where C is 1/2 through a smarter algorithm.
        Term1 = 0
        for i in range(self.N+1):
            for j in range(self.N+1):
                Term1 += u[i]*u[j]*self.C[i,j,k]/2
        
        Term2 = self.v*self.calc_uhat_Term2(u,k)
        
        return Term2-Term1
    
    def calc_uhat_Term2(self,u,k):
        #According to Chebyshev Polynomial literature, derivatives of u (which is the base u we expand into sums
        #of chebyshev polynomials) can be simply passed onto to the coefficients (i.e. u_k^(1) for a first
        #derivative of the kth u, u_k^(2) is the 2nd derivative of the kth u). There is a nice formula for
        #The 2nd derivative of the kth u
        #u_k^(2) = 1/c_k * SUM_{p = k+2, p+k even} p*(p^2-k^2)*u_p
        
        result = 0
        if k == 0 :
            c = 2
        else:
            c = 1
        
        #Sum from k+2,k+4,..,N (or N-1 if k is odd)
        if k%2 == 0:
            for p in range(k+2,self.N+2,2):
                result += p*(p**2-k**2)*u[p]
        else:
            for p in range(k+2,self.N+1,2):
                result += p*(p**2-k**2)*u[p]
        
        return result/c

























