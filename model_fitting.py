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

import mol
import os
import argparse

import numpy as np
from scipy.optimize import least_squares
from scipy.optimize import basinhopping, dual_annealing

def cost_fn(x,p,par_names=None):
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
    
    p = mol.run_euler(p)
    y = p.y

    fsol = y[:p.N,:]
    psol = y[p.N:,:]

    I = fsol + psol
    #'2h', '4h', '30min', '1h', '24h', 'control', '8h30'

    #print('p.r call',p.r)
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
            if time >= 0.5 or time <=4:
                w = 5
            
            data = p.data_avg_fns[hour](p.r)
            err += w*np.linalg.norm(data-I[:,idx])
        
    stdout = (err,p.eps,p.df,p.dp,p.u(0))
    s1 = 'err={:.2f},eps={:.2f},'\
        +'d_f={:.2f}, dp={:.2f}, u_val={:.2f}'
    print(s1.format(*stdout))
    return err


def get_data_residuals(p,par_names=['eps','df','dp'],
                       bounds=[(0,1),(0,100),(0,100)],
                       init=[.001,1,1],
                       parfix={}):
    
    """
    fit sim to data

    p: parameter object
    par_names: list of parameter names to be used in optimization
    bounds: tuple of lo-hi bounds for each parameter
    init: initial guess
    parfix: dict of parameters to fix at this value.
    """

    d_class = mol.Data()
    
    data_avg, data_rep = d_class._build_data_dict()

    # fit gaussian.
    
    pars_control = d_class._load_gaussian_pars(d_class.data_dir,data_avg,
                                               'control',n_gauss=6)
    pars_steadys = d_class._load_gaussian_pars(d_class.data_dir,data_avg,
                                               '24h',n_gauss=6)
    
    
    control_fn = d_class.control_fn
    steadys_fn = d_class.steadys_fn

    assert(len(par_names) == len(bounds))
    assert(len(init) == len(bounds))

    for key in parfix:
        setattr(p,key,parfix[key])
    
    args = (p,par_names)

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
                        help='If true, display plots')
    parser.add_argument('-v','--save-residuals',dest='save_residuals',action='store_true',
                        help='If true, save residuals from optimization')

    parser.add_argument('-q','--quiet',dest='quiet',action='store_true',
                        help='If true, suppress all print statements')
    parser.add_argument('-s','--scenario',dest='scenario',
                        help='Choose scenario. see code for scenarios',type=int,
                        default=-1)

    parser.add_argument('-r','--recompute',dest='recompute',action='store_true',
                        help='If true, recompute optimization data')
    
    parser.set_defaults(plots= False,save_residuals=False,quiet=False)
    args = parser.parse_args()
    print(args)

    # 1440 minutes in 24 h
    p = mol.Params(T=1500,dt=0.01,L=25)

    
    if args.scenario == -1:
        # original model fitting eps, df, dp

        fname = p.data_dir+'scenario-1_residuals.txt'

        par_names=['eps','df','dp','u_val']
        bounds = [(0,1),(0,1),(0,10),(0,2)]
        init = [0.1,0,1,.15]
        parfix = {}

    if args.scenario == 0:
        # purely reversible trapping
        # dp = 0
        fname = p.data_dir+'scenario0_residuals.txt'

        par_names=['eps','df','u_val']
        bounds = [(0,1),(0,100),(0,2)]
        init = [0.001,1,0.16]
        parfix = {'dp':0}

    elif args.scenario == 1:
        # non-dynamical trapping, pure transport
        # df = dp = 0
        fname = p.data_dir+'scenario1_residuals.txt'

        par_names=['eps','u_val']
        bounds = [(0,1),(0,2)]
        init = [0.001,0.16]
        parfix = {'df':0,'dp':0}

    elif args.scenario == 2:
        # irreversible trapping
        # df = 0
        fname = p.data_dir+'scenario2_residuals.txt'

        par_names=['eps','dp','u_val']
        bounds = [(0,1),(0,100),(0,2)]
        init = [0.001,1,0.16]
        parfix = {'df':0}

    elif args.scenario == 3:
        pass
    
    elif args.scenario == 4:
        pass

    
    file_not_found = not(os.path.isfile(fname))
    
    if args.recompute or file_not_found:
        res = get_data_residuals(p,par_names=par_names,
                                 bounds=bounds,
                                 init=init,
                                 parfix=parfix)
        
        np.savetxt(fname,res.x)
        res_arr = res.x
        
    else:
        
        res_arr = np.loadtxt(fname)

    #p.__init__(T=200,dt=0.01)
    for i, val in enumerate(res_arr):
        setattr(p,par_names[i],val)

    print(par_names,res_arr)
    if args.plots:

        
        
        p.T = 200
        p.dt = .002
        print(int(p.T/p.dt))
        p = mol.run_euler(p)
        mol.plot_sim(p)
        
        plt.show()

if __name__ == "__main__":
    main()
