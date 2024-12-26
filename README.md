# PDE-Spectral-Method-Chebyshev-Tau

The goal of this script is to solve the Burgers equation using Spectral Methods, specifically the chebyshev tau method with the boundary conditions
where:

u(-1,t) = u(1,t) = 0

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

Plugging our uN definition into the Burgers equation then multiply by  T_k(x)*(1-x^2)^1/2 and integrate both sides we are able to single out an equation for each k, but note that we drop last two (largest k terms)to enforce the boundary equation.

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


https://github.com/user-attachments/assets/53fa8c45-b982-4784-a6fd-0b4f4a9b6522


