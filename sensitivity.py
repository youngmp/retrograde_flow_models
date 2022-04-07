"""
script for running SALib examples. for efast, sensitivity analysis
"""

import model_fitting as mf
import pde

import os
import argparse

from SALib.sample import saltelli
from SALib.analyze import sobol
from SALib.test_functions import Ishigami
import numpy as np

def main():
    
    parser = argparse.ArgumentParser(description='run sensitivity analysis for retrograde flow',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    
    parser.add_argument('-r','--recompute',dest='recompute',action='store_true',
                        help='If true, recompute sensitivity analysis')

    parser.add_argument('-n','--samples',dest='samples',
                        help='set number of samples',default=1024,type=int)

    args = parser.parse_args()
    
    problem = {
        'num_vars': 2,
        'names': ['eps', 'dp'],
        'bounds': [[0, .05],
                   [0, .05]]
    }

    recompute = True

    fname = 'data/slib.txt'
    file_not_found = not(os.path.isfile(fname))

    if args.recompute or file_not_found:

        param_values = saltelli.sample(problem, args.samples)

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

    Si = sobol.analyze(problem,Y)

    print('S1',Si['S1'])
    print('ST',Si['ST'])

if __name__ == '__main__':
    main()
