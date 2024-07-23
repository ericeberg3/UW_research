import numpy as np
from findiff import FinDiff
from scipy.integrate import odeint
from scipy.integrate import ode, complex_ode
from scipy.optimize import minimize
import scipy
import matplotlib.pyplot as plt
from time import perf_counter

def tg2(y,z,alpha,N,U,Upp,all_z,c):
    ind = np.argmin(np.abs(all_z-z))
    coeff = alpha**2 - (N**2 - (U[ind]-c) * Upp[ind])/(U[ind]-c)**2
    return [y[1], coeff*y[0]]

def rootode(tg2, ic2, z, arglist): return odeint(tg2, [ic2,1], z,args=( arglist ))

def tg_solve(alpha, z, N, H, U, ic, c_test):
    dz = z[1] - z[0]
    d_dz = FinDiff(0, dz, 1)
    d2_dz2 = FinDiff(0, dz, 2)
    d2U_dz2 = d2_dz2(U)
    d2mat = np.array(d2_dz2.matrix(U.shape).toarray())

    f = lambda ic2: rootode(tg2, ic2, z, (alpha,N,U,d2U_dz2,z,c_test))[-1, 0]
    ic_sol = scipy.optimize.bisect(f, -100, 100)
    
    sol = odeint(tg2, [ic_sol, ic[1]], z,args=( (alpha,N,U,d2U_dz2,z,c_test) ))

    return odeint(tg2, [ic_sol,ic[1]], z,args=( (alpha,N,U,d2U_dz2,z,c_test) )), [ic_sol, ic[1]]