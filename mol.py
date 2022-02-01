
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

F: density of static/immobile insoluble vimentin
P: density of mobile insoluble vimentin

Transforming into 1D coordinates then restricting to 1d:

\pa F/\pa t = T(F,P)
\pa P/\pa t = \frac{1}{r}\pa/\pa r (r u(r) P) - T(F,P)

T(F,P) = d_p F - d_f F

let's try no-flux at the origin? before doubling up on the domain.
"""

import os

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from matplotlib.gridspec import GridSpec
from scipy.optimize import least_squares as lsq
from scipy.integrate import solve_ivp
from scipy.interpolate import interp1d


class Data:
    """
    class for loading/saving/manipulating data for
    use in other functions
    """

    def __init__(self,recompute=False,
                 data_dir='./data/',
                 L0=10,
                 L=30):

        self.L = L
        self.L0 = L0
        self.recompute = recompute
        self.data_dir = data_dir
                
        if not(os.path.isdir(self.data_dir)):
            print('created data directory at',self.data_dir)
            os.mkdir(self.data_dir)

        
        data_avg, data_rep = self._build_data_dict(L0=self.L0,L=self.L)

        self.data_avg_fns = self._build_data_fns(data_avg)

        self.data_avg = data_avg
        self.data_rep = data_rep

        # generate functions
        print('data keys',data_avg.keys())

        pars_control = self._load_gaussian_pars(self.data_dir,data_avg,'control',n_gauss=7)
        pars_steadys = self._load_gaussian_pars(self.data_dir,data_avg,'24h',n_gauss=7)
                
        self.control_fn = CallableGaussian(pars_control)
        self.steadys_fn = CallableGaussian(pars_steadys)
        #p.steadys_fn = steadys_fn
        
        if False:
            #plot_data(data_rep,data_avg)

            fn_ctrl = get_1d_interp(data_avg['control'])

            
            x_data = data_avg['control'][:,0]
            y_data = data_avg['control'][:,1]
            t = 'Gaussian Fit VS Control'
            #plot_gaussian_fit(x_data,y_data,pars_control,t=t)
            plot_gaussian_fit(x_data,fn_ctrl(x_data),pars_control,t=t)
            plt.show()


    @staticmethod
    def _load_gaussian_pars(data_dir,data,time,n_gauss=6,recompute=False):
        """
        time: str. time of data
        """

        fname = data_dir + time + str(n_gauss) + '.txt'
        file_not_found = not(os.path.isfile(fname))
        
        if recompute or file_not_found:

            # fit gaussian.
            x_data = data[time][:,0]
            y_data = data[time][:,1]
            pars = get_gaussian_res(x_data,y_data,time,
                                    n_gauss=n_gauss)

            np.savetxt(fname,pars)

            return pars

        else:
            return np.loadtxt(fname)
        
    @staticmethod
    def _build_data_dict(fname='patrons20180327dynamiqueNZ_reformatted.xlsx',
                         L0=10,L=30):

        # load intensity data
        data_raw = pd.read_excel(fname,engine = 'openpyxl',header=[0,1,2])

        # get set of primary headers (to remove duplicates)
        list_hours = set()
        for col in data_raw.columns:
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
            x1 = rep_data['radius']; y1 = rep_data['intensity']

            mask = (x1>=L0)*(x1<=L)
            n = np.sum(y1[mask]*np.diff(x1[mask],prepend=0))
            
            z1 = np.zeros((len(x1[mask]),2))
            z1[:,0] = x1[mask]; z1[:,1] = y1[mask]/n
            data_rep[hour] = z1

            # save avg data
            avg_cell_number = cols[0][0]
            avg_data = data_raw[hour][avg_cell_number]
            avg_data.dropna(inplace=True)
            
            x2 = avg_data['radius']; y2 = avg_data['intensity']
            mask = (x2>=L0)*(x2<=L)

            n = np.sum(y2[mask]*np.diff(x2[mask],prepend=0))
            z2 = np.zeros((len(x2[mask]),2))
            z2[:,0] = x2[mask]; z2[:,1] = y2[mask]/n
            data_avg[hour] = z2

        return data_avg, data_rep

    @staticmethod
    def _build_data_fns(data):
        """
        build 1d interpolation of data for use in cost function
        data is either data_avg or data_rep, the dicts constructed in 
        _build_data_dict
        """

        data_fn = {}

        for hour in data.keys():

            fn = get_1d_interp(data[hour])

            data_fn[hour] = fn

        return data_fn
        
class Params(Data):
    """
    class for holding and moving parameters
    """

    def __init__(self,u=None,N=200,eps=0,
                 df=1,dp=1,uval=0.16,
                 T=300,dt=0.001):
        """
        ur: speed of retrograde flow as a function of r
        N: domain mesh size
        L: domain radius in um
        """
        super().__init__()

        #self.data_dir = data_dir

        self.uval = uval
        
        self.u = self._u_constant

        self.eps = eps
        self.N = N
        
        self.r, self.dr = np.linspace(self.L0,self.L,N,retstep=True)
        self.dp = dp
        self.df = df

        self.T = T
        self.dt = dt
        #self.TN = int(T/dt)

        # dummy array to speed up integration
        self.du = np.zeros(2*N)
    

        
    def _u_constant(self,r):
        """
        speed of retrograde flow induced by actin
        r: domain position.
        constant case.
        """
        
        return self.uval

class CallableGaussian(object):
    """
    create gaussian object to initialize with different parameters
    """
    def __init__(self,pars):
        """
        pars: list of parameters
        """
        self.pars = pars

    def __call__(self,x):
        return g_approx(x,self.pars)

def parsets(scenario='default',method=''):

    
    if scenario == 'default':
        if method == 'annealing':
            pars = {'eps':0.00000000e+00,
                    'df':4.80607607e-04,
                    'dp':4.28388798e-01,
                    'uval':1.79645389e+00,
                    'T':1500,'dt':0.01
            }
    
    elif scenario == 0:
        if method == 'annealing':
            pars = {'eps':6.79309767e-03,
                    'df':6.65433710e+01,
                    'dp':0,
                    'uval':5.65823966e-03,
                    'T':1500,'dt':0.01}
            
    elif scenario == 1:
        if method == 'annealing':
            pars = {'eps':0.18585833,
                    'df':0,
                    'dp':0,
                    'uval':0.0062434,
                    'T':1500,'dt':0.01}

    elif scenario == 2:
        if method == 'annealing':
            pars = {'eps':0,
                    'df':0,
                    'dp':0.38812839,
                    'uval':2,
                    'T':1500,'dt':0.01}

    return pars

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

    # manually update last P derivative
    #out[-1] = 0

    return out


def u_dyn(c,t):
    return c*t

def get_1d_interp(data,kind='linear'):
    """
    get linear interpolation of data
    data: must be data_avg['time'] where 'time' is chosen
    """

    x = data[:,0]
    y = data[:,1]
    #y = y[:-1]/np.sum(y[:-1]*np.diff(x)) # normal
    #y = np.append(y,y[-1])    
    fn = interp1d(x,y,kind=kind,
                  bounds_error=False,
                  fill_value=0)

    return fn

def plot_data(data_rep,data_avg):
    """
    plotting function
    """
    

    # plot data if you want
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

    return fig

def gauss_test_fn(x,mu,sig,amp=1):
    """
    use gaussian initial distribution
    for testing upwinding integrator
    """
    return amp*np.exp(-0.5*(x-mu)**2/sig**2)
    

def run_euler(p):
    """
    t: time array
    y0: initial condition
    """

    y0 = np.zeros(2*p.N)
    y0[:p.N] = p.control_fn(p.r)*p.eps
    y0[p.N:] = p.control_fn(p.r)*(1-p.eps)
    #y0[p.N:] = 100#gauss_test_fn(p.r,20,1)*100

    TN = int(p.T/p.dt)
    t = np.linspace(0,p.T,TN)
    y = np.zeros((2*p.N,TN))

    y[:,0] = y0
    
    for i in range(TN-1):
        y[-1,i] = 0
        y[:,i+1] = y[:,i] + p.dt*rhs(t[i],y[:,i],pars=p)
        #y[p.N-1,i+1] = y[p.N-2,i+1]

    p.t = t
    p.y = y
    p.y0 = y0

    return p

def get_ana(p,t,option='i',scenario='1'):
    """
    analytical solution.
    given time t return solution on domain r.

    option: scenario.
    i == 0:
    i == 1: non-dynamical anchoring, T(F,P) = 0
    i == 2: 

    t: scalar
    r: array for domain
    control_fn: 1d function for control initial condition
    p: 1d function for P(r,0)
    
    ###### for now assume that p_fn is control_fn*eps

    I = F + P
    """

    p_fn = p.control_fn
    
    r = p.r
    u = p.u(r)

    if scenario == '1':

        if option == 'i':
            #print(u,t)#,p_fn(r+u*t))
            return p_fn(r)*p.eps + (r+u*t)*p_fn(r+u*t)/r
        elif option == 'f':
            return p_fn(r)*p.eps
        elif option == 'p':
            return (r+u*t)*p_fn(r+u*t)/r
        else:
            raise ValueError('analytical solution invalid option',str(option))


def add_plot_sim_labels(axs):

    
    
    axs[0][0].set_xlabel('t')
    axs[0][1].set_xlabel('t')
    axs[1][0].set_xlabel('r')
    
    axs[1][1].set_xlabel('r')
    axs[2][0].set_xlabel('r')
    axs[2][1].set_xlabel('r')

    axs[0][0].set_ylabel('r')
    axs[0][1].set_ylabel('r')
    axs[1][0].set_ylabel('Norm. Intensity')
    
    axs[1][1].set_ylabel('Norm. Intensity')
    axs[2][0].set_ylabel('Norm. Intensity')
    axs[2][1].set_ylabel('Norm. Intensity')

    axs[0][0].set_title('F (Immobile)')
    axs[0][1].set_title('P (Mobile)')

    axs[1][0].set_title('F')
    axs[1][1].set_title('P')

    

    #axs[2][0].set_title()

    axs[1][0].legend()
    axs[1][1].legend()

    axs[2][0].legend()
    axs[2][1].legend()

    #axs[-1][-1].axes.xaxis.set_visible(False)
    #axs[-1][-1].axes.yaxis.set_visible(False)

    #axs[-1][-1].axis('off')

    #ax[-1][-1].text(x,y,'$\varepsilon={:.2g}$'.format())
    

def plot_sim(p):
    """
    plot first and last simulation data
    compared to data
    """
    
    
    fsol = p.y[:p.N,:]
    psol = p.y[p.N:,:]

    I = fsol + psol
    
    nrows = 3
    ncols = 2
    
    fig = plt.figure(figsize=(7,7))
    gs = GridSpec(nrows,ncols,figure=fig)
    
    axs = []
    
    for i in range(nrows):
        axs.append([])
        for j in range(ncols):
            axs[i].append(fig.add_subplot(gs[i,j]))
    
    axs[0][0].imshow(fsol[::-1,:],aspect='auto',
                    extent=(p.t[0],p.t[-1],p.r[0],p.r[-1]))

    axs[0][1].imshow(psol[::-1,:],aspect='auto',
                    extent=(p.t[0],p.t[-1],p.r[0],p.r[-1]))

    axs[1][0].plot(p.r,p.y0[:p.N],label='Initial F')
    axs[1][0].plot(p.r,fsol[:,-1],label='Final F')
    #axs[1,0].plot(p.r,steadys_fn(p.r),label='Final/2')
    
    axs[1][1].plot(p.r,p.y0[p.N:],label='Initial P')
    axs[1][1].plot(p.r,psol[:,-1],label='Final P')
    #axs[1,1].plot(p.r,steadys_fn(p.r)/2,label='Final/2')

    axs[2][0].plot(p.r,p.control_fn(p.r),label='Initial (data)')
    axs[2][0].plot(p.r,fsol[:,0]+psol[:,0],label='Initial F+P (PDE)')

    axs[2][1].plot(p.r,p.steadys_fn(p.r),label='Final (data)')
    axs[2][1].plot(p.r,fsol[:,-1]+psol[:,-1],label='Final F+P (PDE)')

    axs = add_plot_sim_labels(axs)
    
    plt.tight_layout()

    return fig

def plot_sim_intermediate(p):
    """
    plot simulation data
    including intermediate comparisons
    """
    
    F = p.y[:p.N,:]
    P = p.y[p.N:,:]

    I = F + P
    
    nrows = 3
    ncols = 2

    fig,axs = plt.subplots(nrows=3,ncols=7,figsize=(15,5),sharey='row')

    # keys list sorted manually for now.
    keys_list = ['control', '0.5h', '1h', '2h', '4h', '8.5h', '24h']

    assert(len(keys_list) == len(p.data_avg.keys()))
    
    for i,hour in enumerate(keys_list):
        if hour == 'control':
            idx = 0
        else:
            time = float(hour[:-1])
            minute = time*60
            idx = int(minute/p.dt)

        data = p.data_avg_fns[hour](p.r)
        axs[0,i].plot(I[:,idx],label='PDE')
        axs[0,i].plot(data,label='Data')
        
        axs[1,i].plot(F[:,idx])
        axs[2,i].plot(P[:,idx])

        axs[0,i].set_title('I '+str(hour))
        axs[1,i].set_title('F '+str(hour))
        axs[2,i].set_title('P '+str(hour))

        axs[0,i].legend()
    
    plt.tight_layout()

    return fig


def construct_ana_imshow_data(p,nsol=5,option='f'):
    """
    construct imshow data using analytical solution
    """

    TN = int(p.T/p.dt)
    data = np.zeros((p.N,TN))
    
    for i in range(TN):        
        data[:,i] = get_ana(p,p.t[i],option=option)

    return data

def plot_ana(p):


    fig, axs = plt.subplots(nrows=3,ncols=2)

    TN = int(p.T/p.dt)

    nsol = 2
    # plot analytical solution
    p_dat = np.zeros((TN,p.N))

    fsol = construct_ana_imshow_data(p,option='f')
    psol = construct_ana_imshow_data(p,option='p')

    axs[1,0].plot(p.r,fsol[:,0],label='Initial F')
    axs[1,0].plot(p.r,fsol[:,-1],label='Final F')

    axs[1,1].plot(p.r,psol[:,0],label='Initial P')
    axs[1,1].plot(p.r,psol[:,-1],label='Final P')

    # plot imshow data of solutions
    fsol = construct_ana_imshow_data(p,option='f')
    psol = construct_ana_imshow_data(p,option='p')

    axs[0,0].imshow(fsol[::-1,:],aspect='auto',
                    extent=(p.t[0],p.t[-1],p.r[0],p.r[-1]))
    
    axs[0,1].imshow(psol[::-1,:],aspect='auto',
                    extent=(p.t[0],p.t[-1],p.r[0],p.r[-1]))


    axs[0,0].set_title('F analytical')
    axs[0,1].set_title('P analytical')

    axs[1,0].set_title('F analytical')
    axs[1,1].set_title('P analytical')
    
    axs[2,0].set_title('F+P analytical')

    axs[1,0].legend()
    axs[1,1].legend()
    axs[2,0].legend()

    plt.tight_layout()

    return fig
    
def plot_gaussian_fit(x_data,y_data,pars,t=''):
    fig = plt.figure(figsize=(6,3))
    ax = fig.add_subplot(111)
    
    ax.plot(x_data,g_approx(x_data,pars),label='Approx.')
    ax.plot(x_data,y_data,label='Data')
    ax.set_title(t)

    ax.set_xlabel('r')
    ax.set_ylabel('Norm. Intensity')
    ax.legend()

    plt.tight_layout()

    return fig

def g_approx(r,pars):
    """
    gaussian approximation
    length of pars determines how many gaussians to use.
    """
    assert(len(pars)%3 == 0)

    tot = 0
    for i in range(int(len(pars)/3)):
        a = pars[3*i]
        b = pars[3*i+1]
        c = pars[3*i+2]
        tot += a*np.exp(-((r-b)/c)**2)

    return tot
    
def cost_fn(x,data_x,data_y):
    """
    cost function for fitting gaussians to data
    """

    r = data_x
    g = g_approx(r,x)
    
    err =  np.linalg.norm(g-data_y)

    return err

def get_gaussian_res(x_data,y_data,time,n_gauss=3):
    """
    x_data: x values of data to be fitted with gaussians
    y_data: y values of data to be fitted with gaussians
    time: string of time. 'control', '1h', etc.
    see keys() of data.
    n_gauss: number of gaussians to use
    """
    
    #par_init = [1,0,1,1,18,1,1,25,1]
    #par_init = [1,0,1,1,10,1,1,15,1,1,20,1]
    #par_init = [1,0,1,1,5,1,1,10,1,1,15,1,1,20,1]
    par_init = np.zeros(3*n_gauss)
    for i in range(int(len(par_init)/3)):
        par_init[3*i] = .1 # magnitude
        par_init[3*i+1] = i*35/int(len(par_init)/3) # shift
        par_init[3*i+2] = 5 # width
        
    res = lsq(cost_fn,par_init,args=(x_data,y_data))

    # from least squares above
    pars = res.x
    return pars


def main():    

    scenario = 2
    method = 'annealing'
    np.random.seed(0)

    pars = parsets(scenario,method)

    p = Params(**pars)

    # get numerical solution
    p = run_euler(p)

    # plot solution
    if True:
        
        plot_sim(p)
        plot_sim_intermediate(p)
        plot_ana(p)
    
    plt.show()

if __name__ == "__main__":
    main()
