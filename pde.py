"""
fd1 = finite difference order 1

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

        """
        if False:
            fn_ctrl = get_1d_interp(data_avg['control'])

            
            x_data = data_avg['control'][:,0]
            y_data = data_avg['control'][:,1]
            t = 'Gaussian Fit VS Control'
            #plot_gaussian_fit(x_data,y_data,pars_control,t=t)
            plot_gaussian_fit(x_data,fn_ctrl(x_data),pars_control,t=t)
            plt.show()
        """

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
                         L0=10,L=30,normed=True):

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

            mask = (x1>=L0)&(x1<=L)
            if normed:
                n = np.sum(y1[mask]*np.diff(x1[mask],prepend=0))
            else:
                n = 1
                
            z1 = np.zeros((len(x1[mask]),2))
            z1[:,0] = x1[mask]; z1[:,1] = y1[mask]/n
            data_rep[hour] = z1

            # save avg data
            avg_cell_number = cols[0][0]
            avg_data = data_raw[hour][avg_cell_number]
            avg_data.dropna(inplace=True)
            
            x2 = avg_data['radius']; y2 = avg_data['intensity']
            mask = (x2>=L0)&(x2<=L)

            if normed:
                n = np.sum(y2[mask]*np.diff(x2[mask],prepend=0))
            else:
                n = 1
            
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
        
class PDEModel(Data):
    """
    class for holding and moving parameters
    """

    def __init__(self,u=None,N=100,eps=0,
                 df=1,dp=1,uval=0.16,
                 uvalf=0,
                 T=300,dt=0.001,psource=0,
                 fsource=0,
                 imax=100,order=1,D=1):
        """
        ur: speed of retrograde flow as a function of r
        N: domain mesh size
        L: domain radius in um
        """
        super().__init__()

        #self.data_dir = data_dir

        self.uval = uval
        self.uvalf = uvalf

        self.order = order

        if order == 1:
            self.rhs = self._fd1
        elif order == 2:
            self.rhs = self._fd2
        elif order == 'test':
            self.rhs = self._fd1a

        self.D = D
        self.imax = imax
        self.u = self._u_constant
        self.psource = psource
        self.fsource = fsource

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

    
    def _fd1(self,t,y,scenario='default'):
        """
        finite diff for 1st order PDE
        t: float, time.
        y: 2*N array
        p: object of parameters
        """

        r = self.r
        dr = self.dr
        u = self.u
        f = y[:self.N]
        p = y[self.N:]

        out = self.du

        drp = (r[1:]*u(r[1:])*p[1:]-r[:-1]*u(r[:-1])*p[:-1])/(r[:-1]*dr)

        if (scenario == 'default') or (scenario == 0)\
           or (scenario == 1) or (scenario == 2)\
           or (scenario == -2) or (scenario == -3):
            tfp = self.dp*p[:-1] - self.df*f[:-1]
        elif scenario == 4:
            tfp = -self.dp*(1-p[:-1]/self.imax)
        else:
            raise ValueError('Unrecognized or unimplemented scenario',scenario)

        out[:self.N-1] = tfp
        out[self.N:-1] = drp - tfp

        return out


    def _fd1a(self,t,y,scenario='default'):
        """
        finite diff for 1st order PDE. assuming aterograde flow
        t: float, time.
        y: 2*N array
        p: object of parameters
        """

        r = self.r
        dr = self.dr
        u = self.u; uu = u(r[0])
        f = y[:self.N]
        p = y[self.N:]

        out = self.du

        drf = -uu*(r[1:]*p[1:]-r[:-1]*p[:-1])/(r[1:]*dr)
        drp =  uu*(r[1:]*p[1:]-r[:-1]*p[:-1])/(r[:-1]*dr)

        tfp = self.dp*p - self.df*f
        
        out[1:self.N] = drf + tfp[1:]
        out[self.N:-1] = drp - tfp[:-1]

        #out[0] += self.fsource/dr
        #out[-1] += f[-1]/dr

        return out

    def _fd2(self,t,y,scenario='default'):
        """
        finite diff for 1st order PDE
        t: float, time.
        y: 2*N array
        p: object of parameters
        """

        r = self.r;dr = self.dr;u = self.u;D = self.D
        f = y[:self.N];p = y[self.N:]

        out = self.du

        tfp = self.dp*p - self.df*f

        # interior derivatives
        drp2_i = D*(p[2:] - 2*p[1:-1] + p[:-2])/dr**2
        drp1_i = (r[1:-1]*u(r[1:-1])+D)*(p[2:]-p[:-2])/(2*r[1:-1]*dr)
        drp0_i = u(r[1:-1])*p[1:-1]/r[1:-1]

        # left endpoint derivative
        p0 = p[1] + 2*u(r[0])*dr*p[0]/D
        drp2_l = D*(p0 - 2*p[0] + p[1])/dr**2
        drp1_l = (r[0]*u(r[0])+D)*(p[1]-p0)/(2*r[0]*dr)
        drp0_l = u(r[0])*p[0]/r[0] - tfp[0]

        # right endpoint derivative
        pn1 = p[-2] - 2*u(r[-1])*dr*p[-1]/D
        drp2_r = D*(pn1 - 2*p[-1] + p[-2])/dr**2
        drp1_r = (r[-1]*u(r[-1])+D)*(pn1-p[-2])/(2*r[-1]*dr)
        drp0_r = u(r[-1])*p[-1]/r[-1] - tfp[-1]

        # update interior derivatives
        out[:self.N] = tfp
        out[self.N+1:-1] = drp2_i + drp1_i + drp0_i - tfp[1:-1]

        # update left d
        out[self.N] = drp2_l + drp1_l + drp0_l

        # update right d
        out[-1] = drp2_r + drp1_r + drp0_r

        return out

    
    def _run_euler(self,scenario):
        """
        t: time array
        y0: initial condition
        """

        y0 = np.zeros(2*self.N)

        if self.order == 'test':
            y0[:self.N] = self.eps*self.control_fn(self.r)
            y0[self.N:] = self.control_fn(self.r)
        else:
            y0[:self.N] = self.control_fn(self.r)*self.eps
            y0[self.N:] = self.control_fn(self.r)*(1-self.eps)

        TN = int(self.T/self.dt)
        t = np.linspace(0,self.T,TN)
        y = np.zeros((2*self.N,TN))

        y[:,0] = y0

        for i in range(TN-1):
            y[-1,i] = self.psource
            if self.order == 'test':
                y[0] = self.fsource
            y[:,i+1] = y[:,i] + self.dt*self.rhs(t[i],y[:,i],scenario=scenario)
            

        self.t = t
        self.y = y
        self.y0 = y0


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




def get_ana(p,t,option='i',scenario=1):
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

    if scenario == 1:

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

    import matplotlib.pyplot as plt
    from matplotlib.gridspec import GridSpec
    
    fsol = p.y[:p.N,:]
    psol = p.y[p.N:,:]

    if p.order == 'test':
        I = psol

    else:
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
    axs[2][0].plot(p.r,I[:,0],label='Initial F+P (PDE)')

    axs[2][1].plot(p.r,p.steadys_fn(p.r),label='Final (data)')
    axs[2][1].plot(p.r,I[:,-1],label='Final F+P (PDE)')

    axs = add_plot_sim_labels(axs)
    
    plt.tight_layout()

    return fig

def plot_sim_intermediate(p):
    """
    plot simulation data
    including intermediate comparisons
    """

    import matplotlib.pyplot as plt
    
    F = p.y[:p.N,:]
    P = p.y[p.N:,:]
    #print(p.order)

    if p.order == 'test':
        I = P

    else:
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


def construct_ana_imshow_data(p,nsol=5,option='f',scenario=1):
    """
    construct imshow data using analytical solution
    """

    TN = int(p.T/p.dt)
    data = np.zeros((p.N,TN))
    
    for i in range(TN):        
        data[:,i] = get_ana(p,p.t[i],option=option,
                            scenario=scenario)

    return data

def plot_ana(p,scenario=1):


    import matplotlib.pyplot as plt
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
    fsol = construct_ana_imshow_data(p,option='f',scenario=scenario)
    psol = construct_ana_imshow_data(p,option='p',scenario=scenario)

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

    import matplotlib.pyplot as plt
    
    from matplotlib.gridspec import GridSpec

    scenario = 'default'
    method = 'annealing'
    np.random.seed(0)

    pars = parsets(scenario,method)
    #uval=, imax=133.1764
    #eps=, d_f=0.0000, dp=0.0000, uval=, imax=
    #eps=0.9067, d_f=0.1768, dp=4.8743, uval=0.1600, imax=100.0000, D=0.0100

    #eps=, d_f=, dp=, uval=, imax=100.0000, D=0.2117
    #ps=, d_f=, dp=, uval=, imax=100.0000, D=0.0400
    #err=0.6189, eps=0.8226, d_f=0.0017, dp=7.4196, uval=0.1600, imax=100.0000, D=0.0400
    #0.4272, eps=0.1676, d_f=0.5134, dp=7.1638, uval=0.0751, imax=100.0000, D=0.0037
    #eps=0.8278, d_f=3.2512, dp=3.6793, uval=0.5342, imax=100.0000, fsource=0.0627, uvalf=0.4949
    #eps=0.0700, d_f=0.4221, dp=5.6531, uval=0.0951, imax=100.0000, D=0.0037; diffusion default
    #eps=0.8874, d_f=0.4221, dp=6.6712, uval=0.0951, imax=100.0000, D=0.0037 diffusion default

    #eps=0.9994, d_f=0.0001, dp=7.0515, uval=0.1600, imax=100.0000, D=0.0400; diffusion fix D, u
    pars = {'eps':0.8874,
            'df':0.4221,
            'dp':6.6712,
            'uval':0.0951,
            #'fsource':0.0627,
            #'uvalf':0,
            'D':0.0037,
            'T':1500,'dt':0.005,
            'order':2,'N':200
            }
    
    p = PDEModel(**pars)

    # get numerical solution
    p._run_euler(scenario)

    # plot solution
    if True:
        
        plot_sim(p)
        plot_sim_intermediate(p)
        #plot_ana(p)
    
    plt.show()

if __name__ == "__main__":
    main()
