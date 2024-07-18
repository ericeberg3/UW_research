import numpy as np
from findiff import FinDiff
from scipy.integrate import odeint
from scipy.integrate import ode, complex_ode
from scipy.optimize import minimize
import matplotlib.pyplot as plt
from time import perf_counter

def tg(z,y,params):
    alpha,N,U,Upp,all_z,c = params
    ind = np.argmin(np.abs(all_z-z))
    coeff = alpha**2 - (N**2 - (U[ind]-c)*Upp[ind])/(U[ind]-c)**2
    return [y[1], coeff*y[0]]

def tg_solve(alpha, z, N, H, U, ic, c_test):
    dz = z[1] - z[0]

    d_dz = FinDiff(0, dz, 1)
    d2_dz2 = FinDiff(0, dz, 2)
    d2U_dz2 = d2_dz2(U)
    d2mat = np.array(d2_dz2.matrix(U.shape).toarray())

    # Test a single solve using the ode interface
    sol = []
    sol.append(ic)
    r = ode(tg)
    r.set_initial_value(ic, z[0])
    r.set_f_params( (alpha,N,U,d2U_dz2,z,c_test) )
    # for zz in z:
    for zz in z[1:]:
        sol.append(r.integrate(zz))
    sol=np.array(sol)
    return sol
