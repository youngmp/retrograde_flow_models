"""
fit the model to data
parameters to be fit: d_f, d_p, epsilon
where F(r,0) + P(r,0) = g(r) and
F(r,0) = I_control - P(r,0), P(r,0) = epsilon

start with linear PDE for simplicity.

Use initial and final data to start. Will add timed-dependent data eventually.

steady-state is roughly 2 hours based on radius range [10,25].
120 minutes to reach steady-state. simulate out to 240 minutes
then compare I[-1,:] to I[int(TN/2),:]
"""


import pde

import time as tt
import warnings
import os
import argparse

import numpy as np
from scipy.optimize import least_squares
from scipy.optimize import basinhopping, dual_annealing, differential_evolution
from scipy.interpolate import interp1d

np.seterr(all='warn')

def cost_fn(x,p,par_names=None,ss_condition=True,psource=False):
    """
    function for use in least squares.
    x is combination or subset of eps,df,dp.
    par_names: list of variables in order of x
    returns L2 norm.
    """
    assert(len(x) == len(par_names))
    
    for i,val in enumerate(x):
        setattr(p,par_names[i],val)

    TN = int(p.T/p.dt)

    p._run_euler()
    y = p.y

    if False:
        import matplotlib.pyplot as plt
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.plot(y[:,0])
        ax.plot(y[:,int(TN/4)])
        ax.plot(y[:,int(TN/2)])
        ax.plot(y[:,-1])
        plt.show()
        plt.close()
        tt.sleep(1)

    # get solution
    fsol = y[:p.N,:]
    psol = y[p.N:,:]

    I = fsol + psol

    #'2h', '4h', '30min', '1h', '24h', 'control', '8h30'
    err = 0

    for hour in p.data_avg.keys():

        # convert hour to index
        if hour == 'control':
            pass # ignore initial condition (trivial)
        else:

            time = float(hour[:-1])
            minute = time*60
            idx = int(minute/p.dt)

            # restrict solution to observed positions
            I_fn = interp1d(p.r,I[:,idx])
            I_cut = I_fn(p.data_avg[hour][:,0])

            data = p.data_avg[hour][:,1]
            err += np.linalg.norm(data[1:-1]-I_cut[1:-1])**2

    err_old = err
    err = np.log10(err)
    
    if ss_condition:
        if 1e6*np.linalg.norm(I[:,int(1200/p.dt)]-I[:,int(1440/p.dt)])**2 > 1e-10:
            err = 1e5

    if p.model == 't1f':
        if np.sum(p._dp_spatial()<0)>0:
            err = 1e5

    #err_log = np.log10(err)
    if np.isnan(err):
        err = 1e5

    stdout = [err,err_old,p.eps,p.df]
    s1 = 'log(err)={:.4f},err_old={:.6f}, eps={:.4f}, '\
        +'df={:.4f}'

    if not(p.dp_nonconstant):
        stdout.append(p.dp)
        s1 += ', dp={:.4f}'

    if psource:
        stdout.append(p.psource)
        s1 += ', psource={:.4f}'

    if p.model[:-1] == 't2':
        stdout.append(p.dp1);stdout.append(p.dp2)
        s1 += ', dp1={:.4f}, dp2={:.4f}'

    if p.model[:-1] == 'jamming':
        stdout.append(p.imax)
        s1 += ', imax={:.4f}'

    if not(p.u_nonconstant):
        stdout.append(getattr(p,'us'+str(0)))
        s1 += ', us={:.2f}'

    #print(s1,stdout)
    print(s1.format(*stdout))

    return err

def get_data_residuals(p,par_names=['eps','df','dp'],
                       bounds=[(0,1),(0,100),(0,100)],
                       init=[.001,1,1],
                       parfix={},ss_condition=False,
                       psource=False,seed=0,
                       method='annealing'):

    """
    fit sim to data

    p: parameter object
    par_names: list of parameter names to be used in optimization
    bounds: tuple of lo-hi bounds for each parameter
    init: initial guess
    parfix: dict of parameters to fix at this value.
    """

    d_class = pde.Data()

    print(par_names,bounds)
    assert(len(par_names) == len(bounds))
    assert(len(init) == len(bounds))

    for key in parfix:
        setattr(p,key,parfix[key])
    
    args = (p,par_names,ss_condition,psource)

    if method == 'annealing':
        
        res = dual_annealing(cost_fn,bounds=bounds,args=args,
                             visit=2.7,restart_temp_ratio=1e-06,
                             initial_temp=6e4,accept=-5,seed=seed,
                             maxiter=5000,maxfun=1e8)
                             #local_search_options={'method':'Nelder-Mead','bounds':bounds})
                             #local_search_options={'nan_policy':'omit'})
        
    elif method == 'de':
        res = differential_evolution(cost_fn,bounds=bounds,args=args,tol=1e-4)
        
    elif method == 'bh':
        minimizer_kwargs = {"method": "Powell",'bounds':bounds,'args':args}
        res = basinhopping(cost_fn,init,minimizer_kwargs=minimizer_kwargs)
        
    # defaults: initial_temp=5230, restart_temp_ratio=2e-05, visit=2.62, accept=-5.0,maxiter=1000

    return res
    

def main():

    parser = argparse.ArgumentParser(description='run model fitting for retrograde flow',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('-p','--show-plots',dest='plots',action='store_true',
                        help='If true, display plots',default=False)

    parser.add_argument('-q','--quiet',dest='quiet',action='store_true',
                        help='If true, suppress all print statements',default=False)
    
    parser.add_argument('-s','--model',dest='model',
                        help='Choose model. see code for models',
                         default='-1',type=str)

    parser.add_argument('--seed',dest='seed',
                        help='choose seed for dual annealing',
                        default=0,type=int)

    parser.add_argument('--steady-state-condition',dest='ss_condition',
                        action='store_true',
                        help='Choose whether or not to force steady-state condition',
                        default=True)

    parser.add_argument('--umax',dest='umax',
                        help='set max velocity for spatial velocity',
                        default=2,type=float)

    parser.add_argument('--dmax',dest='dmax',
                        help='set max value for dp and df',
                        default=5,type=float)

    parser.add_argument('--psource',dest='psource',
                        action='store_true',
                        help='Include source for P in search',
                        default=False)

    parser.add_argument('--interp-o',dest='interp_o',
                        help='set order for function interpolation',
                        default=1,type=int)

    parser.add_argument('-r','--recompute',dest='recompute',action='store_true',
                        help='If true, recompute optimization data')

    parser.add_argument('--method',dest='method',type=str,
                        default='annealing',
                        help='select method. annealing, bh (basin hopping), or de (differential evolution)')


    args = parser.parse_args()
    #print(args)

    # 1440 minutes in 24 h.    
    p = pde.PDEModel(T=1500,dt=.05,order=1,N=50,model=args.model)
    
    #print(args)
    
    if args.model == 't1a':
        # original model fitting eps, df, dp
        par_names = ['eps','df','dp','us0']
        bounds = [(0,1),(0,args.dmax),(0,args.dmax),(0,args.umax)]
        init = [0.1,0,1,0.16]
        parfix = {}
    
    elif args.model == 't1b':
        # purely reversible trapping
        # dp = 0        
        par_names=['eps','df','us0']
        bounds = [(0,1),(0,args.dmax),(0,args.umax)]
        init = [0.001,1,0.16]
        parfix = {'dp':0}

    elif args.model == 't1c':
        # non-dynamical trapping, pure transport
        # df = dp = 0
        par_names=['eps','us0']
        bounds = [(0,1),(0,args.umax)]
        init = [0.001,0.16]
        parfix = {'df':0,'dp':0}

    elif args.model == 't1d':
        # irreversible trapping
        # df = 0
        par_names=['eps','dp','us0']
        bounds = [(0,1),(0,args.dmax),(0,args.umax)]
        init = [0,1,0.16]
        parfix = {'df':0}

    elif args.model == 't1e':
        par_names=['eps','dp']
        bounds = [(0,.02),(0,args.dmax)]
        init = [0,.01]
        parfix = {'df':0,'us0':0}
    
    elif args.model == 't1f':
        par_names=['eps','us0']
        bounds = [(0,1),(0,args.umax)]
        init = [0,.01]
        parfix = {'df':0}

    elif args.model == 't2a':
        par_names = ['eps','dp1','dp2','df','us0']
        bounds = [(0,1),(0,args.dmax),(0,args.dmax),(0,args.dmax),(0,args.umax)]
        init = [0,1,1,1,0.16]
        parfix = {}

    elif args.model == 't2b':
        par_names = ['eps','df','us0']
        bounds = [(0,1),(0,args.dmax),(0,args.umax)]
        init = [0,1,0.16]
        parfix = {'dp1':0,'dp2':0}

    elif args.model == 't2c':
        par_names = ['eps','us0']
        bounds = [(0,1),(0,args.umax)]
        init = [0,0.16]
        parfix = {'dp1':0,'dp2':0,'df':0}

    elif args.model == 't2d':
        par_names = ['eps','dp1','dp2','us0']
        bounds = [(0,1),(0,args.dmax),(0,args.dmax),(0,args.umax)]
        init = [0,1,1,0.16]
        parfix = {'df':0}

    elif args.model == 'jamminga':
        par_names = ['eps','imax','us0','dp','df']
        bounds = [(0,1),(0,1),(0,args.umax),(0,args.dmax),(0,args.dmax)]
        init = [0,1,0.2,1,1]
        parfix = {}

    elif args.model == 'jammingb':
        par_names = ['eps','imax','us0','df']
        bounds = [(0,1),(0,1),(0,args.umax),(0,args.dmax)]
        init = [0,1,0.2,1]
        parfix = {'dp':0}

    elif args.model == 'jammingc':
        par_names = ['eps','imax','us0']
        bounds = [(0,1),(0,1),(0,args.umax)]
        init = [0,1,0.2]
        parfix = {'dp':0,'df':0}

    elif args.model == 'jammingd':
        par_names = ['eps','imax','us0','dp']
        bounds = [(0,1),(0,1),(0,args.umax),(0,args.dmax)]
        init = [0,1,0.2,1]
        parfix = {'df':0}

    else:
        raise ValueError('invalid model choice')

    fname_pre = p.data_dir+args.model+'_residuals'
    fname_pre += '_umax='+str(args.umax)
    fname_pre += '_dmax='+str(args.dmax)
    fname_pre += '_seed='+str(args.seed)

    if args.method == 'de' or args.method == 'bh':
        fname_pre += '_method='+args.method
    
    
    if args.psource:
        par_names.append('psource')
        bounds.append((0,1))
        init.append(0)
        fname_pre += '_psource'

    if args.ss_condition:
        fname_pre += '_ss'
    

    fname = fname_pre + '.txt'
    fname_err = fname_pre+'_err.txt'
    
    file_not_found = not(os.path.isfile(fname))\
                     and not(os.path.isfile(fname_err))

    if args.recompute or file_not_found:
        res = get_data_residuals(p,par_names=par_names,
                                 bounds=bounds,
                                 init=init,
                                 parfix=parfix,
                                 ss_condition=args.ss_condition,
                                 psource=args.psource,
                                 seed=args.seed,
                                 method=args.method)
        
        np.savetxt(fname,res.x)
        np.savetxt(fname_err,[res.fun])
        
        res_arr = res.x
        res_fun = res.fun
    else:
        
        res_arr = np.loadtxt(fname)
        res_fun = np.loadtxt(fname_err)

    #p.__init__(T=200,dt=0.01)
    for i, val in enumerate(res_arr):
        setattr(p,par_names[i],val)


    if args.seed == 0:
        print(par_names)

    print(str(args.seed)+'\t',end="")
    for i in range(len(par_names)):
        print(str(res_arr[i])+'\t',end="")
    print(res_fun)
    #print()
    #print(par_names,res_arr,'. err =',res_fun)
    if args.plots:
        
        p.T = 200
        p.dt = .005
        print(int(p.T/p.dt))
        p = pde.run_euler(p)
        pde.plot_sim(p)  
      
        plt.show()

if __name__ == "__main__":
    main()
