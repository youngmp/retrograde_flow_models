"""
AIC model selection stuff.

AIC (large number of observations k < n/40):
$\text{AIC}_i = n\log(\text{RSS}_i/n) + 2k_i$,

where $n$ is the total number of observations, $k_i$ is the number of estimated parameters for model $i$, $\text{RSS}_i$ is the minimum value of the objective/cost function.

In our case, we need to use the corrected AIC (small number of observations k > n/40):
$\text{Aic}_i = n\log(\text{RSS}_i/n) + 2 k_i n/(n-k_i-1)$,

In our problem, n = (the number of spatial points in data) * (number to time observations).
"""

import pde
import lib

import numpy as np
import os


models = ['t1a','t1b','t1c','t1d','t1e','t1f',
          't2a','t2b','t2c','t2d',
          'jamminga','jammingb','jammingc','jammingd']

#models = ['t1a','t1b','t1c','t1d','t1e','t1f']

p = pde.Data()

# count number of observations
n = 0
for key in p.data_avg.keys():
    n += len(p.data_avg[key][:,0])
    print(len(p.data_avg[key][:,0]),key)

print('number of observations',n)

aic_data = {}
aic_min = 100

# compile fnames
#for model in models:

#    fname_list.append(lib.get_parameter_fname(model))
    #if os.path.isfile(fname) and os.path.isfile(fname):
    #    fname_list.append(fname)
    #    fname_err_list.append(fname_err)
    #    model_names.append(str(model)+'_'+str(order))

for i in range(len(models)):

    if models[i] == 't1e':
        seeds = 10
    elif models[i] == 't1f':
        seeds = 1
    else:
        seeds = 100
        
    err, min_seed = lib.lowest_error_seed(models[i],seeds=seeds,method='de')
    #print(models[i],min_seed)
    residuals = np.loadtxt(lib.get_parameter_fname(models[i],min_seed,method='de'))

    #print(models[i],'err',err,min_seed)
    err = np.exp(err)
    

    k = len(residuals)
    aic = n*np.log(err/n) + 2*k#*n/(n-k-1)

    # compile aic data

    aic_data[models[i]] = {}
    aic_data[models[i]]['res'] = residuals
    aic_data[models[i]]['err'] = err
    aic_data[models[i]]['k'] = k+1
    aic_data[models[i]]['aic'] = aic

    if aic < aic_min:
        aic_min = aic

aic_data['min'] = aic_min

# compute delta
for m in models:
    aic_data[m]['delta'] = aic_data[m]['aic'] - aic_data['min']
    #print(m,aic_data[m]['delta'])

# compute weights
for m in models:
    norm = 0
    for r in models:
        delta_r = aic_data[r]['delta']
        norm += np.exp(-delta_r/2)
        #print(norm)
    
    delta_i = aic_data[m]['delta']
    aic_data[m]['w'] = np.exp(-delta_i/2)/norm
    aic_data[m]['w_unnorm'] = np.exp(-delta_i/2)
    
print('{:10s} {:10s} {:10s} {:10s}'.format('scenario','k_i','aic_i','w_i'))
for m in models:
    k = aic_data[m]['k']
    aic = aic_data[m]['aic']
    delta = aic_data[m]['w']
    s = '{:<10s} {:<10.0f} {:<10.4f} {:<10.4f}'
    print(s.format(m,k,aic,delta))
    #print('add +1 to k')



print()
print('RSS: ')

for m in models:
    err = aic_data[m]['err']
    print(m,'\t',err)
    #print('add +1 to k')


"""    
print('evidence ratios: ')

w_best = aic_data['t1e']['w_unnorm']
for m in models:
    w = aic_data[m]['w_unnorm']
    #print(m,aic_data[m]['delta'])
    print(m,w_best,w,w_best/w,np.exp(-.5*803))
    #print('add +1 to k')

"""
