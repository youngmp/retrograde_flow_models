"""
AIC model selection stuff.

AIC (large number of observations k < n/40):
$\text{AIC}_i = n\log(\text{RSS}_i/n) + 2k_i$,

where $n$ is the total number of observations, $k_i$ is the number of estimated parameters for model $i$, $\text{RSS}_i$ is the minimum value of the objective/cost function.

In our case, we need to use the corrected AIC (small number of observations k > n/40):
$\text{AICc}_i = n\log(\text{RSS}_i/n) + 2 k_i n/(n-k_i-1)$,

In our problem, n = 7 observations.
"""

import numpy as np
import os
import pde

scenario_numbers = [-1,0,1,2,4]

p = pde.PDEModel()

n = 219*7 # number of observations

aicc_data = {}
aicc_min = 100

fname_list = []
fname_err_list = []
model_names = []
# compile fnames
for m in scenario_numbers:
    for order in [1]:
        fname_pre = p.data_dir+'scenario'+str(m)+'_residuals'
        fname_pre += '_order='+str(order)
        fname_pre += '_ss'
        fname = fname_pre + '.txt'
        fname_err = fname_pre + '_err.txt'

        print(fname,os.path.isfile(fname))
        if os.path.isfile(fname) and os.path.isfile(fname):
            fname_list.append(fname)
            fname_err_list.append(fname_err)
            model_names.append(str(m)+'_'+str(order))

for i in range(len(fname_list)):

    residuals = np.loadtxt(fname_list[i])
    err = np.loadtxt(fname_err_list[i])
    k = len(residuals)
    aicc = n*np.log(err/n) + 2*k*n/(n-k-1)

    m = model_names[i]
    # compile aic data

    aicc_data[m] = {}
    aicc_data[m]['res'] = residuals
    aicc_data[m]['err'] = err
    aicc_data[m]['k'] = k+1
    aicc_data[m]['aicc'] = aicc

    if aicc < aicc_min:
        aicc_min = aicc


aicc_data['min'] = aicc_min

# compute delta
for m in model_names:    
    aicc_data[m]['delta'] = aicc_data[m]['aicc'] - aicc_data['min']

# compute weights
for m in model_names:
    norm = 0
    for r in model_names:
        delta_r = aicc_data[r]['delta']
        norm += np.exp(-delta_r/2)

    
    delta_i = aicc_data[m]['delta']
    
    aicc_data[m]['w'] = np.exp(-delta_i/2)/norm
    
print('{:10s} {:10s} {:10s} {:10s}'.format('scenario','k_i','aicc_i','w_i'))
for m in model_names:
    k = aicc_data[m]['k']
    aicc = aicc_data[m]['aicc']
    delta = aicc_data[m]['w']
    s = '{:<10s} {:<10.0f} {:<10.4f} {:<10.4f}'
    print(s.format(m,k,aicc,delta))
    #print('add +1 to k')
