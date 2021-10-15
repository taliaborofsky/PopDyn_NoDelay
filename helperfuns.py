import numpy as np
import pandas as pd
import scipy.stats as scs

# Helper functions
def Kfun(s, norm = scs.norm(0.2,1)):
    # Finds K, the probability of socially learning
    #input: d real positive numbe or -1, and a normmal curve
    #output K
    K = norm.cdf(s) - norm.cdf(-s)
    return(K)
# find pc given s and the normal curve
def pcfun(s, norm = scs.norm(0.2,1)):
    # Finds pc, the probability of individual learning correctly
    #input: d real positive numbe or -1, and a normmal curve
    #output pc
    pc = 1 - norm.cdf(s)
    return(pc)

# find pw given s and the normal curve
def pwfun(s,norm = scs.norm(0.2,1)):
    # Finds pw, the probability of individual learning incorrectly
    #input: d real positive numbe or -1, and a normmal curve
    #output pw
    pw = norm.cdf(-s)
    return(pw)


# The resource relative density recursion
def r_fun(r, N, u,beta, learn_r_u,learn_r_x, eta = 1):
    # The recursion for resource $i$
    x = 1 - u
    
    tot_learn_r = u*learn_r_u + x*learn_r_x
    r_next = r*(1 + eta - beta*N*tot_learn_r)/(1+eta*r)
    if r_next < 0:
        r_next = 0
        print('CP extinct')
    return(r_next)

# population size of predators in the next generation 
def PopSize(N_t,W,delta):
    # W is mean population fitness
    # delta is the death rate
    N_tplus1 = N_t*(W-delta)
    return(N_tplus1)



def NextGen(u_r,u_R,x_r,x_R,N, r,R,beta,delta,K,pc,dk,dpc):
    print(uvec)
    u_r,u_R = uvec
    u = u_r + u_R; x = 1 - u;
    x_r,x_R = xvec
    p_r = u_r + x_r
    
    learn_r_u = K*p_r + pc*(r/(r+R)) # no longer relevant - if r + R > 0 else K*p_r
    learn_r_x = (K+dk)*p_r + (pc+dpc)*(r/(r+R)) # no longer relevant - if r + R > 0 else (K+dk)*p_r
    
    Wu_r = u*learn_r_u*(1+r)
    Wu_R = u*(1-learn_r_u)*(1+R)
    #Wx_r = x*learn_r_x*(1+r)
    #Wx_R = x*(1-learn_r_x)*(1+R)
    W = Wu_r + Wu_R #+ Wx_r + Wx_R
    # need to check this with my eqn
    u_r = Wu_r/W
    u_R = Wu_R/W
    uvec_new = [u_r,u_R]
    #uvec_new = np.array([Wu_r,Wu_R])/W
    xvec_new = [0,0] # np.array([Wx_r,Wx_R])/W
    
    tot_learn_r = u*learn_r_u + x*learn_r_x
    r_next = r*(1 + eta - beta*N*tot_learn_r)/(1+eta*r)
    r_new = r_next if r_next > 0 else 0
    
    N_new = PopSize(N,W,delta)
    return(uvec_new, xvec_new,W,N_new, r_new)

def calculate_eq_r(K,pc,delta,R):
    # Calculates the nonzero r equilibrium
    
    if R>delta:
        return(np.NaN)
    
    a = pc + K*(delta-R)/(1+delta)
    b = (delta-R)*(K*(1+R)/(1+delta) -1) - R*pc
    c = -R*(delta-R)*(1-K/(1+delta))
    if b**2 - 4*a*c > 0:
        return((-b + np.sqrt(b**2 - 4*a*c))/(2*a))
    else:
        return(np.NaN)

def GetXsteps(param_grid, tsteps):
    u_r_init = param_grid.u_r_init.values
    r_init = param_grid.r_init.values
    
def iterate_row(row,tsteps):
    # iterates from the initial point. tsteps generations
    # row could also be a dataframe
    u_r = row.u_r_init.values
    u_R = 1 - u_r
    r = row.r_init.values
    n = r.size
    x_r = np.zeros(n)
    x_R = np.zeros(n)
    N = row.N_init.values
    R = row.R.values
    beta = row.beta.values; delta = row.delta.values
    K = row.K.values; pc = row.pc.values
    NextGenV = np.frompyfunc(NextGen,13,5)
    for t in range(1,tsteps):
        uvec_new, xvec_new,W,N_new, r_new = NextGenV(u_r,u_R,x_r,x_R,N,r,R,beta,delta,K,pc,0,0)
        uvec, xvec, N, r = uvec_new, xvec_new, N_new,r_new
    row.u_r_eq = uvec[0]
    row.r_eq = r
    row.N_eq = N
    row.W_eq = W
    return(row)
    
