"""
method of lines implementation of one of Steph's PDEs

see page 4 in Steph's draft

\begin{align*}
\pa F/\pa t &= d_p P - d_f F\\
\pa P/\pa t &= \nabla \cdot (uP) - a_d P - a_k P + d_k K + d_d D- d_p P + d_f F\\
\pa K/\pa t &= \nabla \cdot ((u-u_k)K) + a_k P-(c_k + d_k) K + c_d D\\
\pa D/\pa t &= \nabla \cdot ((u + u_d)D) + a_d P + c_k K - (c_d+d_d)D
\end{align*}

u: speed of retrograde flow induced by actiin. u_r = \SI{0.15}{mu m/min} when constant.
u_k: speed of MT-dependent transport driven mainly by kinesin. u_k = \SI{0.35}{mu m/s} when constant.
u_d: speed of MT-dependent transport driven mainly by dynein. u_d = \SI{0.3}{mu m/s} when constant.


Transforming into 1D coordinates then restricting to 1d:

\pa F/\pa t = T(F,P)
\pa P/\pa t = \frac{1}{r}\pa/\pa r (r u(r) P) - T(F,P)

T(F,P) = d_p F - d_f F

let's try no-flux at the origin? before doubling up on the domain.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

from scipy.integrate import solve_ivp
from scipy.interpolate import interp1d

class Params(object):
    """
    class for holding and moving parameters
    """

    def __init__(self,u=None,N=200,L0=10,L=30,eps=0,
                 df=1,dp=1,
                 T=300,dt=0.001):
        """
        ur: speed of retrograde flow as a function of r
        N: domain mesh size
        L: domain radius in um
        """

        if u == None:
            self.u = self._u_constant
        else:
            self.u = u

        self.eps = eps
        self.N = N
        self.L = L
        self.L0 = L0
        
        self.r, self.dr = np.linspace(L0,L,N,retstep=True)
        self.dp = dp
        self.df = df

        self.T = T
        self.dt = dt
        self.TN = int(T/dt)

        # dummy array to speed up integration
        self.du = np.zeros(2*N)
        

    def _u_constant(self,r):
        """
        speed of retrograde flow induced by actin
        r: domain position.
        constant case.
        """
        
        return 0.15
        
        

def rhs(t,y,pars):
    """
    t: float, time.
    y: 2*N array
    p: object of parameters
    """
    
    r = pars.r
    dr = pars.dr
    u = pars.u
    f = y[:pars.N]
    p = y[pars.N:]

    out = pars.du

    #print(y[0],y[1])
    drp = (r[1:]*u(r[1:])*p[1:]-r[:-1]*u(r[:-1])*p[:-1])/(r[:-1]*dr)
    #print(drp[0])
    tfp = pars.dp*p[:-1] - pars.df*f[:-1]
    
    out[:pars.N-1] = tfp
    out[pars.N:-1] = drp - tfp
    
    #drp = (r[:-1]*u(r[:-1])*p[:-1]-r[1:]*u(r[:-1])*p[:-1])/(r[1:]*dr)
    #tfp = pars.dp*p[1:-1] - pars.df*f[1:-1]
    #out[1:pars.N-1] = tfp
    #out[pars.N+1:-1] = drp - tfp

    #drp = (r[2:]*u(r[2:])*p[2:]-r[:-2]*u(r[:-2])*p[:-2])/(2*r[1:-1]*dr)
    #tfp = pars.dp*p[1:-1] - pars.df*f[1:-1]
    #out[1:pars.N-1] = tfp
    #out[pars.N+1:-1] = drp - tfp

    #out[pars.N] = 0
    #out[-1] = 0

    return out


def u_dyn(c,t):
    return c*t

def build_data_dict(fname='patrons20180327dynamiqueNZ_reformatted.xlsx',
                    plot_data=False,return_interp=False):
    
    # load intensity data
    data_raw = pd.read_excel(fname,engine = 'openpyxl',header=[0,1,2])

    #print(data.head())
    #print(data_rep)
    #print(data['24h']['5c'][['radius','intensity']])
    #print(data['24h']['5c'][''])

    # get set of primary headers (to remove duplicates)
    list_hours = set()
    for col in data_raw.columns:
        #print(col[0])
        list_hours.add(col[0])

    # collect data in dict
    data_avg = {} # for average data
    data_rep = {} # for rep. data
    for hour in list_hours:

        # get column names
        # ('13c', 'radius'), ('13c', 'intensity'), ('rep', 'radius'), ...
        cols = data_raw[hour].columns
        #print(hour,col,col[0])

        # save rep. data
        rep_data = data_raw[hour]['rep']
        rep_data.dropna(subset=['radius'],inplace=True)
        #print(rep_data)
        x1 = rep_data['radius']
        y1 = rep_data['intensity']
        #print(x1)
        z1 = np.zeros((len(x1),2))
        z1[:,0] = x1
        z1[:,1] = y1
        data_rep[hour] = z1

        # save avg data
        avg_cell_number = cols[0][0]
        avg_data = data_raw[hour][avg_cell_number]
        avg_data.dropna(inplace=True)
        #print(hour,avg_cell_number,cols,cols[0])
        x2 = avg_data['radius']
        y2 = avg_data['intensity']
        z2 = np.zeros((len(x2),2))
        z2[:,0] = x2
        z2[:,1] = y2
        data_avg[hour] = z2
    
    # plot data if you want
    if plot_data:
        list_hours_subset = ['control','24h']
        #print(data_rep.keys(),data_avg.keys())
        fig, axs = plt.subplots(nrows=1, ncols=2,figsize=(8,3))

        for hour in list_hours_subset:
            axs[0].plot(data_rep[hour][:,0],data_rep[hour][:,1],label=hour)
            axs[1].plot(data_avg[hour][:,0],data_avg[hour][:,1],label=hour)

        axs[0].set_title('Representative')
        axs[1].set_title('Average')

        axs[0].set_xlabel('r (um)')
        axs[1].set_xlabel('r (um)')
        
        axs[0].set_ylabel('Intensity')
        axs[1].set_ylabel('Intensity')
        
        axs[0].legend()
        axs[1].legend()

        plt.tight_layout()
        plt.show()


    if return_interp:
                    
        # create interpolating functions out of the data for use in simulations

        x = data_avg['control'][:,0]
        y = data_avg['control'][:,1]
        y = y[:-1]/np.sum(y[:-1]*np.diff(x)) # normal
        y = np.append(y,y[-1])    
        control_fn = interp1d(x,y,kind='quadratic')

        x = data_avg['24h'][:,0]
        y = data_avg['24h'][:,1]
        y = y[:-1]/np.sum(y[:-1]*np.diff(x)) # normal
        y = np.append(y,y[-1])
        steadys_fn = interp1d(x,y,kind='quadratic')

        return data_avg, data_rep, control_fn, steadys_fn
    
    else:
        return data_avg, data_rep


def run_euler(y0,p):
    """
    t: time array
    y0: initial condition
    """

    t = np.linspace(0,p.T,p.TN)
    y = np.zeros((2*p.N,p.TN))

    y[:,0] = y0
    
    for i in range(p.TN-1):
        y[:,i+1] = y[:,i] + p.dt*rhs(t[i],y[:,i],pars=p)
        y[p.N-1,i+1] = y[p.N-2,i+1]

    return t,y

def main():    
        
    funs = build_data_dict(return_interp=True)
    data_avg, data_rep, control_fn, steadys_fn = funs

    # basin-hopping parameters
    p_bh = {'eps':8.37055184e-03,
            'df':7.44798956,
            'dp':9.04261998,
            'L':25,
            'T':300,'dt':0.01}

    # least squares parameters
    p_ls = {'eps':1.78128872e-02,
            'df':4.58942054e+01,
            'dp':2.05296340e-01,
            'L':25,
            'T':300,'dt':0.01}

    # "fits" but doesn't capture steady-state.
    p_bh_2 = {'eps':0.9438153,
              'df':1.1886299,
              'dp':6.97629,
              'L':25,
              'T':600,'dt':0.01}


    p_bh_2 = {'eps':0.42816316,
              'df':2.12893668,
              'dp':1.4024988,
              'L':25,
              'T':600,'dt':0.01}

    np.random.seed(0)
    p = Params(**p_bh_2)
  
    y0 = np.zeros(2*p.N)
    y0[:p.N] = control_fn(p.r)*p.eps
    y0[p.N:] = control_fn(p.r)*(1-p.eps)
    
    t,y = run_euler(y0,p)
    fsol = y[:p.N,:]
    psol = y[p.N:,:]

    I = fsol + psol
    err = np.linalg.norm(control_fn(p.r)-I[:,-1])
    print('err',err)
    #print(psol[-1,:])
    
    #sol = solve_ivp(rhs,(t[0],t[-1]),y0,args=(p,),
    #                t_eval=t)
    #y = sol.y

    #fig,axs = plt.subplots(nrows=3,ncols=2,sharey='row')

    nrows = 3
    ncols = 2
    
    fig = plt.figure()
    gs = GridSpec(nrows,ncols,figure=fig)
    axs = []
    
    for i in range(nrows):
        axs.append([])
        for j in range(ncols):
            axs[i].append(fig.add_subplot(gs[i,j]))

    axs[0][0].imshow(fsol[::-1,:],aspect='auto',
                    extent=(t[0],t[-1],p.r[0],p.r[-1]))

    axs[0][1].imshow(psol[::-1,:],aspect='auto',
                    extent=(t[0],t[-1],p.r[0],p.r[-1]))

    #axs[1,0].plot(t,np.sum(y[:p.N,:],axis=0)*p.dr)
    #axs[1,1].plot(t,np.sum(y[p.N:,:],axis=0)*p.dr)
    
    #for i in range(10):
    #    axs[1,0].plot(fsol[int(p.N*i/10),:],color=str(int(i)/12))    
    #axs[1,0].set_title('10 solutions over time')

    #for i in range(10):
    #    axs[1,1].plot(psol[int(p.N*i/10),:],color=str(int(i)/12))
    #axs[1,1].set_title('10 solutions over time')

    axs[1][0].plot(p.r,y0[:p.N],label='Initial F')
    axs[1][0].plot(p.r,fsol[:,-1],label='Final F')
    #axs[1,0].plot(p.r,steadys_fn(p.r),label='Final/2')
    
    axs[1][1].plot(p.r,y0[p.N:],label='Initial P')
    axs[1][1].plot(p.r,psol[:,-1],label='Final P')
    #axs[1,1].plot(p.r,steadys_fn(p.r)/2,label='Final/2')

    axs[2][0].plot(p.r,steadys_fn(p.r),label='Final (data)')
    axs[2][0].plot(p.r,fsol[:,-1]+psol[:,-1],label='F+P (PDE)')
    
    axs[0][0].set_xlabel('t')
    axs[0][1].set_xlabel('t')
    axs[1][0].set_xlabel('r')
    axs[1][1].set_xlabel('r')

    axs[0][0].set_ylabel('r')
    axs[0][1].set_ylabel('r')
    axs[1][0].set_ylabel('Normalized Intensity')
    axs[1][1].set_ylabel('Normalized Intensity')

    axs[0][0].set_title('F')
    axs[0][1].set_title('P')

    axs[1][0].set_title('F')
    axs[1][1].set_title('P')

    

    #axs[2,0].set_title('Final F')
    #axs[2,1].set_title('Final P')

    axs[1][0].legend()
    axs[1][1].legend()

    plt.tight_layout()
    plt.show()
    
    

if __name__ == "__main__":
    main()
