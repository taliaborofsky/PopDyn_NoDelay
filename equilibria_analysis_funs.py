import numpy as np
import scipy.stats as scs
from helperfuns import *

def custmax_s_2(mu, R, δ, smax = 3):
    def get_Nβ(s):
        K = Kfun(s,scs.norm(mu))
        pc = pcfun(s,scs.norm(mu))
        r = get_r_hat_1(K,pc,δ,R)
        Nβ = (1-r)*(r-R)/(δ-R)
        return Nβ
    # gives 2 results...from searching up and from searching down
    
    
    def get_best(smin,smax = 3,nsteps = 1000, reverse = False, go_smaller = True):
        list_iterate = np.linspace(smax, smin) if reverse \
            else np.linspace(smin, smax, nsteps) 

        besty = 0
        for s in list_iterate:
            testy = get_Nβ(s)
            if np.isnan(testy):
                continue
            elif testy >= besty:
                besty = testy
                bestx = s
            else:
                # stopping condition
                if go_smaller:
                    bestx, besty = get_best(smin = bestx, 
                                            smax = s, 
                                            go_smaller = False,
                                           nsteps = 100)
                else:
                    break
        return(bestx, besty)
    
    
    
    maxs_0, maxNβ_0 = get_best(smin=0, smax = smax)
    maxs_1, maxNβ_1 = get_best(smin=maxs_0 + 0.001, smax = smax, reverse = True)
    
    tolerance = 1e-4
    if maxNβ_1 > maxNβ_0 + tolerance:
        maxs_0 = np.nan
        maxNβ_0 = np.nan
    elif maxNβ_0 > maxNβ_1 + tolerance or np.abs(maxs_1 - maxs_0)<1e-3:
        maxs_1 = np.nan
        maxNβ_1 = np.nan
        


    return maxs_0, maxs_1, maxNβ_0, maxNβ_1
            
                

                

def get_r_hat_1(K,pc,delta,R):
    # for R < delta
    
    a = pc + K*(delta-R)/(1+delta)
    b = (delta-R)*(K*(1+R)/(1+delta)-1) - R*pc
    c = -R*(delta-R)*(1 - K/(1+delta))
    
    # check if there's an equilibrium
    Q_r_1 = pc*(1-R) + (delta-R)*(1+R)*(2*K/(1+delta)-1)
    if Q_r_1 >=0:
        rpos = (-b + np.sqrt(b**2 - 4*a*c))/(2*a) 
        return(rpos)
    else:
        return(np.nan)
    
get_r_hat_vec1 = np.frompyfunc(get_r_hat_1,4,1)       
