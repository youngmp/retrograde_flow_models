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
import os
import argparse

import numpy as np
from scipy.optimize import least_squares
from scipy.optimize import basinhopping, dual_annealing

def cost_fn(x,p,par_names=None,ss_condition=False,psource=False,
            scenario=None):
    """
    function for use in least squares.
    x is combination or subset of eps,df,dp.
    par_names: list of variables in order of x
    returns L2 norm.
    """
    assert(len(x) == len(par_names))

    #p.__init__()
    # update parameter values
    #eps,d_f,d_p = x
    for i,val in enumerate(x):
        setattr(p,par_names[i],val)

    TN = int(p.T/p.dt)

    
    p._run_euler(scenario)
    y = p.y

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

            # weight
            if time >= 2 or time <=9:
                w = 1
            
            data = p.data_avg_fns[hour](p.r)
            err += w*np.linalg.norm(data-I[:,idx])

    if ss_condition:
        if np.linalg.norm(I[:,int(1000/p.dt)]-I[:,int(1440/p.dt)]) > 1e-7:
            err = 1000

    stdout = [err,p.eps,p.df,p.dp,p.u(0),p.imax]
    s1 = 'err={:.4f}, eps={:.4f}, '\
        +'d_f={:.4f}, dp={:.4f}, uval={:.4f}, '\
        +'imax={:.4f}'

    if psource:
        stdout.append(p.psource)
        s1 += ', psource={:.4f}'

    if p.order == 2:
        stdout.append(p.D)
        s1 += ', D={:.4f}'

    
    print(s1.format(*stdout))
    return err

def get_data_residuals(p,par_names=['eps','df','dp'],
                       bounds=[(0,1),(0,100),(0,100)],
                       init=[.001,1,1],
                       parfix={},ss_condition=False,
                       psource=False,
                       scenario=None):
    
    """
    fit sim to data

    p: parameter object
    par_names: list of parameter names to be used in optimization
    bounds: tuple of lo-hi bounds for each parameter
    init: initial guess
    parfix: dict of parameters to fix at this value.
    """

    d_class = pde.Data()

    assert(len(par_names) == len(bounds))
    assert(len(init) == len(bounds))

    for key in parfix:
        setattr(p,key,parfix[key])
    
    args = (p,par_names,ss_condition,psource,
            scenario)

    minimizer_kwargs = {"method": "Powell",
                        'bounds':bounds,
                        'args':args}
    
    #res = basinhopping(cost_fn,init,minimizer_kwargs=minimizer_kwargs)
    res = dual_annealing(cost_fn,bounds=bounds,args=args)

    return res
    

def main():

    parser = argparse.ArgumentParser(description='run model fitting for retrograde flow',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('-p','--show-plots',dest='plots',action='store_true',
                        help='If true, display plots',default=False)

    parser.add_argument('-q','--quiet',dest='quiet',action='store_true',
                        help='If true, suppress all print statements',default=False)
    parser.add_argument('-s','--scenario',dest='scenario',
                        help='Choose scenario. see code for scenarios',
                        default='-1',type=str)

    parser.add_argument('--steady-state-condition',dest='ss_condition',
                        action='store_true',
                        help='Choose whether or not to force steady-state condition',
                        default=False)

    parser.add_argument('--psource',dest='psource',
                        action='store_true',
                        help='Include source for P in search',
                        default=False)

    parser.add_argument('-r','--recompute',dest='recompute',action='store_true',
                        help='If true, recompute optimization data')

    args = parser.parse_args()
    print(args)
    
    if type(args.scenario) is int:
        order = 1; dt = 0.01
    else:
        order = 2; dt = 0.01
    
    # 1440 minutes in 24 h
    p = pde.PDEModel(T=1500,dt=dt,order=order)

    if args.scenario == '-3':
        # uval = 0.16, search only eps full exchange
        fname_pre = p.data_dir+'scenario-3_residuals'

        par_names = ['eps','df','dp']
        bounds = [(0,1),(0,10),(0,10)]
        init = [0.1,1,1]
        parfix = {'uval':0.16}

    if args.scenario == '-2':
        # uval = 0.16, search only eps, no exchange
        fname_pre = p.data_dir+'scenario-2_residuals'

        par_names = ['eps']
        bounds = [(0,1)]
        init = [0.1]
        parfix = {'uval':0.16,'df':0,'dp':0}
    
    if args.scenario == '-1':
        # original model fitting eps, df, dp
        fname_pre = p.data_dir+'scenario-1_residuals'

        par_names = ['eps','df','dp','uval']
        bounds = [(0,1),(0,1),(0,10),(0,2)]
        init = [0.1,0,1,.15]
        parfix = {}

    if args.scenario == '0':
        # purely reversible trapping
        # dp = 0
        fname_pre = p.data_dir+'scenario0_residuals'

        par_names=['eps','df','uval']
        bounds = [(0,1),(0,100),(0,2)]
        init = [0.001,1,0.16]
        parfix = {'dp':0}

    elif args.scenario == '1':
        # non-dynamical trapping, pure transport
        # df = dp = 0
        fname_pre = p.data_dir+'scenario1_residuals'

        par_names=['eps','uval']
        bounds = [(0,1),(0,2)]
        init = [0.001,0.16]
        parfix = {'df':0,'dp':0}

    elif args.scenario == '2':
        # irreversible trapping
        # df = 0
        fname_pre = p.data_dir+'scenario2_residuals'

        par_names=['eps','dp','uval']
        bounds = [(0,1),(0,100),(0,2)]
        init = [0.001,1,0.16]
        parfix = {'df':0}

    elif args.scenario == '3':
        pass
    
    elif args.scenario == '4':
        fname_pre = p.data_dir+'scenario4_residuals'

        par_names=['eps','dp','uval','imax']
        bounds = [(0,1),(0,100),(0,2),(0,1000)]
        init = [0.001,1,0.16,100]
        parfix = {'df':0}

    elif args.scenario == 'a':
        fname_pre = p.data_dir+'scenario_a_residuals'
        par_names = ['eps','df','dp','uval','D']
        bounds = [(0,1),(0,10),(0,10),(0,1),(0,1)]
        init = [0.1,1,1,0.16,0]
        parfix = {}

    
    elif args.scenario == 'b':
        fname_pre = p.data_dir+'scenario_b_residuals'
        par_names = ['eps','df','dp']
        bounds = [(0,1),(0,10),(0,10)]
        init = [0.1,1,1]
        parfix = {'D':.04,'uval':0.16}

        
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
                                 scenario=args.scenario)
        
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

    print(par_names,res_arr,'. err =',res_fun)
    if args.plots:
        
        p.T = 200
        p.dt = .005
        print(int(p.T/p.dt))
        p = pde.run_euler(p)
        pde.plot_sim(p)  
      
        plt.show()

if __name__ == "__main__":
    main()
