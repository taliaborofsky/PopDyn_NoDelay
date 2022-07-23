import numpy as np
import scipy.stats as scs
from helperfuns import *

"""
Find the s value that maximizes N at the E2 equilibrium
inputs: params = [mu, r, delta]
outputs: maxs_0, maxs_1, maxNβ_0, maxNβ_1
"""
def custmax_s_2(params):
    smax = 3
    mu, R, δ = params
    def get_Nβ(s):
        K = Kfun(s,scs.norm(mu))
        pc = pcfun(s,scs.norm(mu))
        r = get_r_hat_1(K,pc,δ,R)
        Nβ = (1-r)*(r-R)/(δ-R)
        return Nβ
    # gives 2 results...from searching up and from searching down
    
    
    def get_best(smin,smax = 3,nsteps = 5000, reverse = False, go_smaller = True):
        list_iterate = np.linspace(smax, smin, nsteps) if reverse \
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
"""
Plot a line with a color corresponding to a third variable (c)
The first two points can have arrows
@inputs: x- the x-axis. y- the y-axis. c- the third variable.
@ output: CS3, the scatterplot which is cleared but is used for a colorbar
"""          
                
def plot_colourline(x,y,c,minc=None,maxc=None, ax=None, cmapAnchor = True, cmap = None, add_arrow = True,hl=20,hw=20):
    if minc == None:
        minc = np.min(c)
        maxc = np.max(c)
        #ptpc = np.ptp(c)
    c_orig = c  
    if cmap == None:
        c = cm.viridis((c-minc)/(maxc-minc))
    else:
        c = cmap((c-minc)/(maxc-minc))

    if ax == None:
        ax = plt.gca()
    # set up a scatter plot and clear it just so I have something to feed the colorbar thing
    if cmapAnchor:
        CS3 = ax.scatter(x,y,c=c)
        ax.cla()
    else:
        CS3 = None
    lw = 3.0
    
    # now plot the arrows
    for i in np.arange(len(x)-1):
        
        ax.plot([x[i],x[i+1]], [y[i],y[i+1]], c=c[i],linewidth=lw, zorder = -1)
    if add_arrow==True:

        ax.quiver(x[0],y[0],x[1] - x[0],y[1] - y[0],
                  scale_units='xy', angles='xy', scale=1,color=c[1],
                  width=lw/500, headwidth=hw, headlength=hl)
        ax.quiver(x[2],y[2],x[3] - x[2],y[3] - y[2],
                  scale_units='xy', angles='xy', scale=1,color=c[1],
                  width=lw/500, headwidth=hw, headlength=hl)

        
    return(CS3)

"""
gets the values of r, N_p, and u_r at the E2 equilibria 
@inputs: parameters K, pc, delta, R
@ outputs: vectors of 2 vectors with the equilibrium data
"""
def get_equilibrium_data(K,pc,delta,R):
    rhats = get_r_hat(K,pc,delta,R)
#     lambdamat = [[],[]]
#     magsmat = [[],[]]
#     stabilities = [0,0]
    Nhats = [0,0]
    uhats = [0,0]
    for i,r in enumerate(rhats):
        if np.isnan(r) == False:
            N = get_N_hat(r,delta,R,beta)
            u = get_u_hat(r,delta,R)
#             lambdas = get_Jstar_lambdas(r,u,N,K,pc,beta,delta,R)
#             mags = np.abs(lambdas)
            
            Nhats[i] = N
            uhats[i]=u
    return(rhats,Nhats,uhats)
#             lambdamat[i] = lambdas
#             magsmat[i] = mags
#             stabilities[i] = max(mags)<1
#     return(rhats,Nhats, uhats, lambdamat,magsmat,stabilities)

"""
Given current values of all the variables, outputs the variables values at the next generation and W, the mean population fitness
@output: u_r,u_R, x_r,x_R,W,N_new, r_new
"""
def NextGen(u_r,u_R,x_r,x_R,N, r,R,beta,delta,K,pc,dk,dpc):
#    u_r,u_R = uvec
    u = u_r + u_R; x = 1 - u;
#    x_r,x_R = xvec
    p_r = u_r + x_r
    eta = 1
    
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
    return(u_r,u_R, x_r,x_R,W,N_new, r_new)  


"""
plots a few trajectories in the given axis (ax) with the given parameters (R, delta, mu,, s, beta) for tsteps generations, where the x-axis is r, the y-axis is N_p, and the color of the lines is u_r, for which we use the colormap (cmap)
"""
def plot_Nru_traj(ax, R,delta,mu,s,beta,tsteps, cmap, minu=0, maxu=1, Nvals = [100,100,100,100],hl=20,hw=20, 
                  rvals = [0.2, 0.8, 0.5, 0.3], uvals = [0.1, 0.5, 0.2, 0.9]):
    
    
    K = Kfun(s,scs.norm(mu))
    pc = pcfun(s,scs.norm(mu))
    

    
    # get trajectories
    x_r=0; x_R=0
    trajs = [get_trajectory(u_r, 1 - u_r, x_r,x_R,N,r,R,beta,delta,K,pc,tsteps) for u_r, r,N in zip(uvals, rvals,Nvals)]
    u_r_vecs = [traj[0] for traj in trajs]
    N_vecs = [traj[4] for traj in trajs]
    r_vecs = [traj[-2] for traj in trajs]
    # plot
    
    # plot trajectories
    for i in range(0,len(uvals)):
        r_vec = r_vecs[i]
        N_vec = N_vecs[i]
        u_r_vec = u_r_vecs[i]
        cs = plot_colourline(r_vec, N_vec, u_r_vec, 
                             minc=minu, maxc=maxu, ax=ax, 
                             cmapAnchor=False, cmap = cmap,
                             hl=hl, hw=hw)
    
    # plot equilibria
    rhats, Nhats, uhats = get_equilibrium_data(K,pc,delta,R)
    
    
    for i,r in enumerate(rhats):
        if np.isnan(r)==False:
            uhat = uhats[i]
            Nhat = Nhats[i]
            #ax.autoscale(False)
            fcolor = cmap((uhat-minu)/(maxu-minu))
            ax.scatter(r,Nhat,s=500, marker='*', edgecolor= 'white', 
                       facecolor= fcolor,
                       linewidth = 2, zorder = 1)
        
            
    return(trajs,rhats,Nhats,uhats)

#     rhats,Nhats, uhats, lambdamat,magsmat,stabilities = get_equilibrium_data(K,pc,delta,R)
    
#     for i,r in enumerate(rhats):
#         if np.isnan(r)==False:
#             uhat = uhats[i]
#             Nhat = Nhats[i]
#             stability = stabilities[i]
#             ucolor = cmap((uhat-minu)/(maxu-minu))
#             face_color = ucolor if stability==1 else 'none'
#             #ax.scatter(r,Nhat,color=cmap((uhat-minu)/(maxu-minu)), s=500, marker='*')
#             ax.scatter(r,Nhat,facecolors = face_color, edgecolors=ucolor, s=500, marker='*')
    
#     return(trajs,rhats,lambdamat,magsmat)


"""
Sets up the trajectory plots... 
nrow x ncols grid of subplots
makes a colorbar
xlabel --> can be 'r' or 'u'
y-axis is always N
"""
def format_traj_plots_with_cbar(nrows, ncols, R, δ, β, xlabel = 'r'):
    cmap = mpl.colors.LinearSegmentedColormap.from_list('mycolors',['blue','red'])
    #make colorbar anchor
    a = np.linspace(0,1,20)
    b = np.linspace(0,1,20)
    np.outer(a,b)
    CS = plt.imshow(np.outer(a,b), cmap = cmap)
    plt.clf()
    fig, axs = plt.subplots(2,2, figsize = [12,12])
    
    
    # set x and y labels
    lab_r = 'Relative density of CP, $r$'
    lab_u = 'Frequency of CP behavior, $u_r$'
    lab_N = 'Predator Pop. Size, $N_p$'
    
    if xlabel == 'r':
        fig.supxlabel(lab_r, fontsize = 20)
    else:
        fig.supxlabel(lab_u, fontsize = 20)

        
    fig.supylabel(lab_N, fontsize = 20)

        
    # title
    ts = r'$R=$%.2f, $\delta=$%.1f, $\beta =$ %.1f' %(R,delta,beta)
    fig.suptitle(ts, fontsize = 20)
    
    # colorbar
    fig.subplots_adjust(right=0.8)
    cbar_ax = fig.add_axes([0.85,0.15,0.05,0.7])
    cb = fig.colorbar(CS, cax = cbar_ax)
    if xlabel == 'r':
        cb.ax.set_title(r'$u_r$', fontsize=20)
    else:
        cb.ax.set_title(r'$r$', fontsize = 20)
        
    # x and y limits

    plt.setp(axs, xlim=[0,1], ylim=[0,180])

    


    return(fig, axs, cmap)

"""
Function to find rhat using parameters K, pi_C, delta, R
# Only works for R < delta
uses an if/else function to check that there is an E2 equilibrium (if Q_r(1) >= 0)
"""            
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

"""
Finds N β
given r, delta, R
already vectorized
Used by cust_max_s2
"""
def get_Nβ_r(r, δ, R):
        Nβ = (1-r)*(r-R)/(δ-R)
        return Nβ
    
"""
Finds u_r_hat
given r, R, delta
already vectorized
Used by cust_max_s2
"""    
def u_r_hat(r,R,δ):
    return (δ-R)*(1+r)/((r-R)*(1+δ))


"""
Find the L (CP learning probability) that maximizes predator population size
...Called \tilde{L}
Note \tilde{L} = 2(R-δ)/(R-1), and R < 1
Need R < δ to make sense
"""
def get_tilde_L(δ, R):
    return  2*(R-δ)/(R-1)
"""
Find the prey density if the learning rate is optimal, i.e. L = \tilde{L}
"""
def get_tilde_r(R):
    R = np.array(R)
    return (1+R)/2
  
"""
Find the $u_r$ that corresponds to the optimal learning rate...
u = \tilde{L}*(1+\tilde{r})/(1+delta)
"""
def get_tilde_u(L, δ, R):
    r = get_tilde_r(R)
    return L*(1 + r)/( 1 + δ)

"""
For given mu, delta, R, find the s = \tilde{s} that is needed to generate Ltilde
The inputs need to be scalar
"""
def find_s_tilde(list_params):
    mu, R, δ = list_params
    L = get_tilde_L(δ,R)
    r = get_tilde_r(R)
    u = get_tilde_u(L, δ, R)
    # L, r, u are all np.array
    norm = scs.norm(mu)
    for s in np.linspace(0,4,1000):
        K = Kfun(s, norm)
        pc = pcfun(s,norm)
        L_definition = K*u + pc*r/(r+R)
        if np.isclose(L_definition, L, rtol = 1e-4, atol = 1e-4): 
            return s
    print ('did not find s')
    
"""
For given mu, delta, R, find the s = \tilde{s} that is needed to generate Ltilde
The inputs need to be scalar
"""
def find_s_tilde_v(list_params):
    mu, R, δ = list_params
    L = np.array(get_tilde_L(δ,R))
    r = get_tilde_r(R)
    u = get_tilde_u(L, δ, R)
    
    # make an empty vector to store optimal s values
    mu = np.array(mu); δ = np.array(δ); R = np.array(R)
    shapes = [mu.shape, δ.shape, R.shape]
    sizes = np.array([mu.size, δ.size, R.size])
    argmax = sizes.argmax()
    shape = shapes[argmax]
    svec = np.zeros(shape)-1
    norm = scs.norm(mu)
    for s in np.linspace(0,4,int(1e06)):
        K = Kfun(s, norm)
        pc = pcfun(s,norm)
        L_definition = K*u + pc*r/(r+R)
        mask = np.isclose(L_definition, L, rtol = 1e-5, atol = 1e-5)
        svec[mask] = s
    
    return(svec)
            