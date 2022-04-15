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

def generate_plot(recompute=True):
    import matplotlib.pyplot as plt
    fig, axs = plt.subplots(nrows=1,ncols=1,figsize=(4,4))

    ne = 20;nd = 20
    maxe = .5;maxd = 1
    
    eps_vals = np.linspace(0,maxe,ne)
    dp_vals  = np.linspace(0,maxd,nd)

    fname = 'data/Z_ne={}_nd={}_maxe={}_maxd={}.txt'.format(ne,nd,maxe,maxd)
    file_not_found = not(os.path.isfile(fname))
    
    if recompute or file_not_found:
        EPS,DP = np.meshgrid(eps_vals,dp_vals)
        Z = np.zeros_like(EPS)
        #DP = np.meshgrid(eps_vals,dp_vals)

        p = pde.PDEModel(T=1500,dt=.05,order=1,N=50,
                         u_nonconstant=True,
                         Nvel=1,df=0)

        par_names = ['eps','dp']

        for i in range(np.shape(EPS)[0]):
            print(i)
            for j in range(np.shape(EPS)[1]):

                x = [EPS[i,j],DP[i,j]]
                exp = mf.cost_fn(x,p,par_names,ss_condition=True,scenario='t1e')

                if exp == 1e5:
                    exp = 1
                elif exp == 0:
                    exp = np.nan

                Z[i,j] = exp

        np.savetxt(fname,Z)
    else:
        Z = np.loadtxt(fname)
    
    
    axs.imshow(Z,extent=[eps_vals[0],eps_vals[-1],dp_vals[0],dp_vals[-1]])
    axs.set_ylabel('dp')
    axs.set_xlabel('eps')
    plt.show()

def main():
    
    parser = argparse.ArgumentParser(description='run sensitivity analysis for retrograde flow',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    
    parser.add_argument('-r','--recompute',dest='recompute',action='store_true',
                        help='If true, recompute sensitivity analysis')

    parser.add_argument('-n','--samples',dest='samples',
                        help='set number of samples',default=2048,type=int)

    parser.add_argument('--test',dest='test',
                        help='slect test. sobol, efast. ',default='sobol')

    args = parser.parse_args()

    
    #generate_plot()

    
    problem = {
        'num_vars': 2,
        'names': ['eps', 'dp'],
        'bounds': [[0, .05],
                   [0, .05]]
    }

    recompute = True

    fname_pre = 'data/sensitivity'
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
            exp = mf.cost_fn(x,p,par_names,ss_condition=False,scenario='t1e')

            Y[i] = exp

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
