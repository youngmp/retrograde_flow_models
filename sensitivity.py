"""
script for running SALib examples. for efast, sensitivity analysis
"""

import model_fitting as mf
import pde

import os
import argparse

from SALib.sample import saltelli, fast_sampler
from SALib.analyze import sobol, fast
from SALib.test_functions import Ishigami
import numpy as np

def load_rss_data(ne=100,nd=100,maxe=0.01,mind=0.006,maxd=0.02,exp=False,
                  recompute=False,ss=True):

    eps_vals = np.linspace(0,maxe,ne)

    assert(mind is not int)
    assert(maxd is not int)
    
    if exp:
        assert(mind<0)
        assert(maxd<0)
        dp_vals  = 10**(np.linspace(mind,maxd,nd))
    else:
        dp_vals = np.linspace(mind,maxd,nd)

    #maxd = dp_vals[-1]

    pars = (ne,nd,maxe,mind,maxd,ss,exp)
    fname = 'data/Z_ne={}_nd={}_maxe={}_mind={}_maxd={}_ss={}_exp={}.txt'.format(*pars)
    
    file_not_found = not(os.path.isfile(fname))
    if file_not_found:
        print('Unable to find file', fname)
    
    if recompute or file_not_found:
        EPS,DP = np.meshgrid(eps_vals,dp_vals)
        Z = np.zeros_like(EPS)

        p = pde.PDEModel(T=1500,dt=.05,order=1,N=50,u_nonconstant=True,Nvel=1,df=0)

        par_names = ['eps','dp']

        for i in range(np.shape(EPS)[0]):
            for j in range(np.shape(EPS)[1]):

                x = [EPS[i,j],DP[i,j]]
                exp = mf.cost_fn(x,p,par_names,ss_condition=ss,scenario='t1e')

                if np.isinf(exp):
                    exp = -0.69

                # note 0 == nan, -0.69 == infty. if ss, then 1e5 is fail.
                Z[i,j] = exp

        np.savetxt(fname,Z)
    else:
        Z = np.loadtxt(fname)

    return eps_vals,dp_vals,Z
    

def main():
    
    parser = argparse.ArgumentParser(description='run sensitivity analysis for retrograde flow',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    
    parser.add_argument('-r','--recompute',dest='recompute',action='store_true',
                        help='If true, recompute sensitivity analysis')

    parser.add_argument('-n','--samples',dest='samples',
                        help='set number of samples',default=2048,type=int)

    parser.add_argument('--test',dest='test',
                        help='slect test. sobol, efast. ',default='sobol')

    parser.add_argument('--ss',dest='ss',action='store_true',
                        help='enable steady-state condition',default=True)

    #parser.add_argument('--me',dest='ss',action='store_true',
    #                    help='enable steady-state condition',default=True)
    
    args = parser.parse_args()

    #generate_plot(args.recompute)
    #eps_vals,dp_vals,Z = load_rss_data(recompute=args.recompute,exp=True,ne=50,nd=50,maxe=0.4,mind=-2.5,maxd=-1.,ss=False)


    problem = {
        'num_vars': 2,
        'names': ['eps', 'dp'],
        'bounds': [[0, .05],
                   [0, .05]]
    }

    recompute = True

    fname_pre = 'data/sensitivity_ss='+str(args.ss)
    
    if args.test == 'sobol':
        fname = fname_pre+'_samples='+str(args.samples)+'_sobol.txt'
    elif args.test == 'efast':
        fname = fname_pre+'_samples='+str(args.samples)+'_efast.txt'


        
    file_not_found = not(os.path.isfile(fname))

    if args.recompute or file_not_found:

        if args.test == 'sobol':
            param_values = saltelli.sample(problem, args.samples)
            
        elif args.test == 'efast':
            param_values = fast_sampler.sample(problem, args.samples,M=4)
            
        Y = np.zeros([param_values.shape[0]])
        print(np.shape(Y))

        p = pde.PDEModel(T=1500,dt=.05,order=1,N=50,
                         u_nonconstant=True,
                         Nvel=1,df=0)

        #def cost_fn(x,p,par_names=None,ss_condition=False,psource=False,
        #            scenario=None,uconst=True):

        for i, p_vals in enumerate(param_values):
            eps, dp = p_vals

            par_names = problem['names']

            x = [eps,dp]
            #exp = mf.cost_fn(x,p,par_names,ss_condition=False,scenario='t1e')
            exp = mf.cost_fn(x,p,par_names,ss_condition=args.ss,scenario='t1e')

            if exp == 1e5:
                Y[i] = exp
            else:
                Y[i] = np.exp(exp)

            print(i,p_vals,Y[i])

            #Y[i] = evaluate_model(X)
        np.savetxt(fname,Y)

    else:
        Y = np.loadtxt(fname)
    #Y = Ishigami.evaluate(param_values)

    #print(Y)

    if args.test == 'sobol':
        Si = sobol.analyze(problem,Y)
    elif args.test == 'efast':
        Si = fast.analyze(problem,Y)

    print(Si.keys())
    print('S1',Si['S1'],Si['S1_conf'])
    print('ST',Si['ST'],Si['ST_conf'])


if __name__ == '__main__':
    main()
