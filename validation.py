"""
run validation on representative cell data
"""

import lib
import pde

import numpy as np
import matplotlib.pyplot as plt


def main():

    # load best parset for t1e
    _,seed = lib.lowest_error_seed()
    pars = lib.load_pars('t1e',seed)
    #pars.update({'L':25})
    print(pars)


    # load rep. data
    p = pde.PDEModel(**pars)

    # run t1e sim
    p._run_euler('t1e',rep=False)

    F = p.y[:p.N,:]
    P = p.y[p.N:,:]

    I = F+P
    # plot rep. data against t1e sol
    #fig = plt.figure()
    #ax = fig.add_subplot(111)

    fig,axs = plt.subplots(nrows=3,ncols=5,figsize=(8,6),
                           gridspec_kw={'wspace':0.1,'hspace':0},
                           sharey='row')

    keys_list = ['control', '2h', '4h','8.5h', '24h']    
    
    for i,hour in enumerate(keys_list):
        #data = p.data_rep['control']

        data = p.data_rep_fns[hour](p.r)
        

        if hour == 'control':
            hour = '0h'
            idx = 0
        else:
            time = float(hour[:-1])
            minute = time*60
            idx = int(minute/p.dt)

        axs[0,i].plot(p.r[1:-1],I[1:-1,idx],color='k')
        axs[0,i].plot(p.r[1:-1],data[1:-1],label='Data',c='tab:green',dashes=(3,1))
        
        #axs.plot(p.r)
        #ax.plot(p.data_rep['control'][:,0],p.data_rep['control'][:,1])

    plt.show()

if __name__ == "__main__":
    main()
