
import numpy as np
import pandas as pd
import scipy.stats as scs
import helperfuns
from helperfuns import *
from numpy.lib.scimath import sqrt as csqrt
import cmath


def get_r_hat(K,pc,delta,R):
    a = pc + K*(delta-R)/(1+delta)
    b = (delta-R)*(K*(1+R)/(1+delta)-1) - R*pc
    c = -R*(delta-R)*(1 - K/(1+delta))
    
    
    discrim = b**2 - 4*a*c
    rpos = 0
    rneg = 0
    if discrim >= 0:
        rpos = (-b + np.sqrt(b**2 - 4*a*c))/(2*a) 
        rneg = (-b - np.sqrt(b**2 - 4*a*c))/(2*a)
        if R < delta:
            rpos_ans = rpos if rpos < 1 and rpos >= delta else np.nan
            rneg_ans=np.nan
        if R > delta:
            rpos_ans = rpos if rpos > 0 and rpos <= delta else np.nan
            rneg_ans = rneg if rneg > 0 and rneg <= delta else np.nan
        if R== delta:
            rpos_ans = R if R > 0 else np.nan
            rneg_ans = np.nan
    else:
        rpos_ans = np.nan
        rneg_ans = np.nan

    return(np.array([rpos_ans,rneg_ans]))
# Finds the nonzero r equilibrium if it exists
# inputs must be arrays with the same dimensions
def get_r_hat_v(K, pc, delta, R):
    
    K,pc,delta,R = [np.array(item) for item in [K,pc,delta,R]]
    #rpos = np.zeros(K.shape)
    #rneg = np.zeros(K.shape)
    
    a = pc + K*(delta-R)/(1+delta)
    b = (delta-R)*(K*(1+R)/(1+delta)-1) - R*pc
    c = -R*(delta-R)*(1 - K/(1+delta))
    discrim = b**2 - 4*a*c
    
    rpos = np.array((-b + csqrt(b**2 - 4*a*c))/(2*a))
    rneg = (-b - csqrt(b**2 - 4*a*c))/(2*a)
    
    rpos_ans = np.zeros(rpos.shape)
    rneg_ans = np.zeros(rneg.shape)
    
    mask1a = ((discrim>=0) & (rpos >0))& ((rpos <=delta)& (R>delta))
    mask1b = ((discrim>=0) & (rpos >0))& ((rpos >=delta)& (R<delta))
    rpos_ans[mask1a] = rpos[mask1a].real # these are real numbers anyways with 0 in the imaginary part, 
    # but this gets rid of the complex casting warning
    rpos_ans[mask1b] = rpos[mask1b].real
    
    mask2a = ((discrim>=0) & (rneg >0))& ((rneg <=delta)&(R>delta))
    mask2b = ((discrim>=0) & (rneg >0))& ((rneg >=delta)&(R<delta))
    rneg_ans[mask2a]=rneg[mask2a].real 
    rneg_ans[mask2b]=rneg[mask2b].real
    
    
    return(rpos_ans, rneg_ans)
# Finds the u_r equilibrium for the r that solves r = 1 - beta*N*L, if it exists
# works if delta \neq R
def get_u_hat(r,delta,R):
    # note need r > 0\ 
    if r==R:
        print('we have a problem. r = R.')
        # either N_p = 0 or R = delta
        if R == delta:
            return(np.nan)
    if R == delta:
        print('we have a problem. R = delta.')
    W = 1 + R + (r-R)
    L = (delta-R)/(r-R)
    u_r = L*(1+r)/(1+delta)
    
    return(u_r)

def get_N_hat(r,delta,R,beta):
    N = (1-r)*(r-R)/(beta*(delta-R))
    return(N)

# this should be 1 + delta if r > 0
#def get_W_hat(r,delta,R)




