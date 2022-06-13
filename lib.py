"""
library functions

"""

import numpy as np

def data_dir():
    return 'data/'

def get_parameter_fname(model,seed,err=False,method=''):
    """
    get file name for parameters
    err: return filename for error
    """
    
    fname_pre = 'data/'+model+'_residuals'

    if model == 't1a':
        fname_pre+='_umax=4.0_dmax=20.0'
    elif model == 't1b':
        fname_pre+='_umax=4.0_dmax=20.0'
    elif model == 't1c':
        fname_pre+='_umax=4.0_dmax=20.0'
        
    elif model == 't1d':
        fname_pre+='_umax=4.0_dmax=20.0'
    elif model == 't1e':
        #fname_pre+='_umax=2_dmax=5'
        fname_pre+='_umax=4.0_dmax=20.0'
    elif model == 't1f':
        fname_pre+='_umax=4.0_dmax=20.0'
        
    elif model == 't2a':
        fname_pre+='_umax=4.0_dmax=20.0'

    elif model == 't2b':
        fname_pre+='_umax=4.0_dmax=20.0'
    elif model == 't2c':
        fname_pre+='_umax=4.0_dmax=20.0'
    elif model == 't2d':
        fname_pre+='_umax=4.0_dmax=200.0'

    elif model == 'jamminga':
        fname_pre+='_umax=4.0_dmax=20.0'
    elif model == 'jammingb':
        fname_pre+='_umax=4.0_dmax=20.0'
    elif model == 'jammingc':
        fname_pre+='_umax=4.0_dmax=20.0'
    elif model == 'jammingd':
        fname_pre+='_umax=4.0_dmax=20.0'

    fname_pre += '_seed='+str(seed)

    if method == '':
        pass
    else:
        fname_pre += '_method='+str(method)

    if err:
        fname = fname_pre + '_ss_err.txt'
    else:
        fname = fname_pre + '_ss.txt'

    return fname


def lowest_error_seed(model='t1e',method='',seeds=10,exclude=None):
    """
    given a model, search over all seeds to find seed with lowest error
    return min err and seed
    """
    if exclude is None:
        exclude = []
    
    err = 10
    min_seed = 0


    for i in range(seeds):
        if not(i in exclude):
            fname = get_parameter_fname(model,i,err=True,method=method)

            err_model = np.loadtxt(fname)

            if err_model < err:
                err = err_model
                min_seed = i

    return err, min_seed


def load_pars(model,seed,method='',return_names=False):
    """
    load residuals found from annealing
    use zero seed for now
    """
    
    fname_pre = 'data/'+model+'_residuals'
    pars = {'T':1500,'dt':0.05,'order':1,'N':50,'model':model}
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
            par_names = ['eps','dp']

        elif scenario == 'f':
            par_names = ['eps','us0']

        
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
            par_names = ['eps','imax','us0','df']

        elif scenario == 'c':
            par_names = ['eps','imax','us0']

        elif scenario == 'd':
            par_names = ['eps','imax','us0','dp']

    res = np.loadtxt(get_parameter_fname(model,seed,False,method))
    
    for i,key in enumerate(par_names):
        pars[key] = res[i]

    if return_names:
        return pars, par_names
    else:
        return pars

def get_seeds_below_threshold(model,threshold=0):
    """
    load parameter set with error below threshold
    assume seeds go from 0-99, threshold in log(rss), method de
    """

    seedlist = []
    
    for seed in range(100):
        fname = get_parameter_fname(model,seed,err=True,method='de')

        err_model = np.loadtxt(fname)

        if err_model < threshold:
            seedlist.append(seed)
            
    return seedlist
