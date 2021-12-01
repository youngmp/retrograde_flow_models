"""
fit the model to data
parameters to be fit: d_f, d_p, epsilon
where F(r,0) + P(r,0) = g(r) and
F(r,0) = I_control - P(r,0), P(r,0) = epsilon

start with linear PDE for simplicity.

Use initial and final data to start. Will add timed-dependent data eventually.
"""

import mol

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import least_squares
from scipy.optimize import basinhopping

def get_solution(p=None,control_fn=None):

    y0 = np.zeros(2*p.N)
    y0[:p.N] = control_fn(p.r)*p.eps
    y0[p.N:] = control_fn(p.r)*(1-p.eps)
    
    t,y = mol.run_euler(y0,p)

    return t,y

def cost_fn(x,control_fn,steadys_fn,p):
    """
    function for use in least squares.
    x is eps,d_f,d_p.
    returns L2 norm.
    """
    
    eps,d_f,d_p = x
    p.eps = eps
    p.d_f = d_f
    p.d_p = d_p
    
    _,y = get_solution(p,control_fn=control_fn)

    fsol = y[:p.N,:]
    psol = y[p.N:,:]

    I = fsol + psol

    #print('p.r call',p.r)
    err1 = np.linalg.norm(steadys_fn(p.r)-I[:,-1])
    err2 = np.linalg.norm(I[:,-1] - I[:,-int(p.TN/5)])

    if False:
        fig = plt.figure()
        ax1 = fig.add_subplot(211)
        ax2 = fig.add_subplot(212)
        
        ax1.plot(I[:,-1])
        ax1.plot(fsol[:,-1])
        ax1.plot(psol[:,-1])
        ax1.plot(steadys_fn(p.r))

        ax2.plot(I[:,0])
        ax2.plot(fsol[:,0])
        ax2.plot(psol[:,0])
        ax2.plot(control_fn(p.r))

        plt.show()
        
    
    print(err1,err2,p.eps,p.d_f,p.d_p)
    return err1+err2

def main():
    
    funs = mol.build_data_dict(return_interp=True)
    data_avg, data_rep, control_fn, steadys_fn = funs

    p = mol.Params(T=600,dt=0.01,L=25)
    #print(cost_fn((.1,2,1),control_fn,steadys_fn,p))
    init = [.001,1,1]
    #init = [1.78128872e-02, 4.58942054e+01, 2.05296340e-01]
    
    args = (control_fn,steadys_fn,p)
    #res = least_squares(cost_fn,init,args=args,
    #                    bounds=([[0,0,0],[np.inf,np.inf,np.inf]]))

    minimizer_kwargs = {"method": "Powell",
                        'bounds':[(0,1),(0,100),(0,100)],
                        'args':args}
    res = basinhopping(cost_fn,init,minimizer_kwargs=minimizer_kwargs)

    print('residuals',res.x)
    print('cost',res.cost)
    print('optimality',res.optimality)

    eps,d_f,d_p = res.x
    p.eps = eps
    p.d_f = d_f
    p.d_p = d_p
    
    t,y = get_solution(p=p,control_fn=control_fn)
    
    fsol = y[:p.N,:]
    psol = y[p.N:,:]

    ratio = p.d_p/(p.d_f+p.d_p)
    
    fig,axs = plt.subplots(nrows=2,ncols=2,sharey='row')

    axs[0,0].imshow(fsol[::-1,:],aspect='auto',
                    extent=(t[0],t[-1],p.r[0],p.r[-1]))

    axs[0,1].imshow(psol[::-1,:],aspect='auto',
                    extent=(t[0],t[-1],p.r[0],p.r[-1]))

    axs[1,0].plot(p.r,fsol[:,0],label='Initial F')
    axs[1,0].plot(p.r,fsol[:,-1],label='Final F')
    #axs[1,0].plot(p.r,steadys_fn(p.r)*(1-ratio),label='Final*dp/(df+dp)')
    
    axs[1,1].plot(p.r,psol[:,0],label='Initial P')
    axs[1,1].plot(p.r,psol[:,-1],label='Final P')
    #axs[1,1].plot(p.r,steadys_fn(p.r)*ratio,label='Final*df/(df+dp)')

    axs[0,0].set_xlabel('t')
    axs[0,1].set_xlabel('t')
    axs[1,0].set_xlabel('r')
    axs[1,1].set_xlabel('r')

    axs[0,0].set_ylabel('r')
    axs[0,1].set_ylabel('r')
    axs[1,0].set_ylabel('Normalized Intensity')
    axs[1,1].set_ylabel('Normalized Intensity')

    axs[0,0].set_title('F')
    axs[0,1].set_title('P')

    axs[1,0].set_title('F')
    axs[1,1].set_title('P')

    

    #axs[2,0].set_title('Final F')
    #axs[2,1].set_title('Final P')

    axs[1,0].legend()
    axs[1,1].legend()

    plt.tight_layout()
    plt.show()



if __name__ == "__main__":
    main()
