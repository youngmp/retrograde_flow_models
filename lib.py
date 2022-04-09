"""
library functions

"""

import numpy as np

def data_dir():
    return 'data/'

def get_parameter_fname(model,seed,err=False):
    """
    get file name for parameters
    err: return filename for error
    """
    fname_pre = 'data/'+model+'_residuals'

    if model == 't1a':
        fname_pre+='_umax=3.0_dmax=5.0'
    elif model == 't1b':
        fname_pre+='_umax=2.0_dmax=20.0'
    elif model == 't1c':
        fname_pre+='_umax=1.0_dmax=5.0'
        
    elif model == 't1d':
        fname_pre+='_umax=4.0_dmax=5.0'
    elif model == 't1e':
        fname_pre+='_umax=2_dmax=5'
    elif model == 't2a':
        fname_pre+='_umax=2.0_dmax=20.0'

    elif model == 't2b':
        fname_pre+='_umax=2.0_dmax=20.0'
    elif model == 't2c':
        fname_pre+='_umax=2.0_dmax=5.0'
    elif model == 't2d':
        fname_pre+='_umax=2.0_dmax=100.0'

    elif model == 'jamminga':
        fname_pre+='_umax=2.0_dmax=5.0'
    elif model == 'jammingb':
        fname_pre+='_umax=2.0_dmax=5.0'
    elif model == 'jammingc':
        fname_pre+='_umax=2.0_dmax=5.0'
    elif model == 'jammingd':
        fname_pre+='_umax=2.0_dmax=5.0'

    if err:
        fname = fname_pre + '_seed='+str(seed)+'_ss_err.txt'
    else:
        fname = fname_pre + '_seed='+str(seed)+'_ss.txt'

    return fname


def lowest_error_seed(model='t1e'):
    """
    given a model, search over all seeds to find seed with lowest error
    for now, seeds go from 0 to 9.
    return min err and seed
    """
    
    err = 10
    min_seed = 10
    
    for i in range(10):
        fname = get_parameter_fname(model,i,err=True)
        
        err_model = np.loadtxt(fname)

        if err_model < err:
            err = err_model
            min_seed = i

    return err, min_seed


def load_pars(model,seed):
    """
    load residuals found from annealing
    use zero seed for now
    """
    
    fname_pre = 'data/'+model+'_residuals'

    #_umax=1_seed='+str(seed)+'_ss.txt'
    

    pars = {'T':1500,'dt':0.02,'order':1,'N':50}
    
    scenario = model[-1]
    
    if model[:-1] == 't1':
        pars.update({'eps':0,'dp':0,'df':0,'us0':0})

        if scenario == 'a':
            par_names=['eps','df','dp','us0']
            
        elif scenario == 'b':
            par_names=['eps','df','us0']

        elif scenario == 'c':
            par_names = ['eps','us0']

        elif scenario == 'd':
            par_names = ['eps','dp','us0']

        elif scenario == 'e':
            par_names = ['eps','dp'];pars['u_nonconstant']=True
        
    elif model[:-1] == 't2':
        pars.update({'eps':0,'dp1':0,'dp2':0,'df':0,'us0':0})

        if scenario == 'a':
            par_names = ['eps','dp1','dp2','df','us0']

        elif scenario == 'b':
            par_names = ['eps','df','us0']

        elif scenario == 'c':
            par_names = ['eps','us0']

        elif scenario == 'd':
            par_names = ['eps','dp1','dp2','us0']

    elif model[:-1] == 'jamming':
        pars.update({'eps':0,'imax':0,'us0':0,'dp':0,'df':0})
        #pars['dt']=0.01
        #pars['N']=100

        if scenario == 'a':
            par_names = ['eps','imax','us0','dp','df']

        elif scenario == 'b':
            par_names = ['eps','imax','us0','df'];fname_pre+='_umax=2.0_dmax=5.0'

        elif scenario == 'c':
            par_names = ['eps','imax','us0'];fname_pre+='_umax=2.0_dmax=5.0'

        elif scenario == 'd':
            par_names = ['eps','imax','us0','dp'];fname_pre+='_umax=2.0_dmax=5.0'

    
    #fname = fname_pre+'seed='+str(seed)+'_ss.txt'
    res = np.loadtxt(get_parameter_fname(model,seed))
    
    for i,key in enumerate(par_names):
        pars[key] = res[i]
        #print(i,key,res[i],model,seed)

    #print('pars from load_pars',pars,model)
    return pars
