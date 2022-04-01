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
import time
import copy

import pandas as pd
import numpy as np

from scipy.optimize import least_squares as lsq
from scipy.integrate import solve_ivp
from scipy.interpolate import interp1d

#import matplotlib.pyplot as plt


class Data:
    """
    class for loading/saving/manipulating data for
    use in other functions
    """

    def __init__(self,recompute=False,
                 data_dir='./data/',
                 L0=10,
                 L=30,normed=True):

        self.L = L
        self.L0 = L0
        self.recompute = recompute
        self.data_dir = data_dir
        self.normed = normed
                
        if not(os.path.isdir(self.data_dir)):
            print('created data directory at',self.data_dir)
            os.mkdir(self.data_dir)

        
        data = self._build_data_dict(L0=self.L0,L=self.L,
                                      normed=self.normed)

        data_avg, data_rep, data_avg_raw, data_rep_raw = data
        
        self.data_avg = data_avg
        self.data_rep = data_rep
        self.data_avg_raw = data_avg_raw
        self.data_rep_raw = data_rep_raw


        # interp1d functions -- linear interp on bounded domain
        self.data_avg_fns = self._build_data_fns(data_avg)


        # generate functions
        print('data keys',data_avg.keys())

        pars_control = self._load_gaussian_pars(self.data_dir,data_avg,'control',
                                                normed=self.normed,n_gauss=11,
                                                recompute=self.recompute)
        pars_steadys = self._load_gaussian_pars(self.data_dir,data_avg,'24h',
                                                normed=self.normed,n_gauss=6,
                                                recompute=self.recompute)

        # gaussian interp fns. -- gaussian interp on R
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

    def _get_gaussian_res(self,x_data,y_data,time,n_gauss=3):
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
            par_init[3*i] = 100 # magnitude
            par_init[3*i+1] = i*35/int(len(par_init)/3) # shift
            par_init[3*i+2] = 5 # width

        res = lsq(cost_fn,par_init,args=(x_data,y_data))

        # from least squares above
        pars = res.x
        return pars

    def _load_gaussian_pars(self,data_dir,data,time,normed,n_gauss=6,recompute=False):
        """
        time: str. time of data
        """

        fname = data_dir + time + str(n_gauss) +'_normed=' + str(normed) + '.txt'
        file_not_found = not(os.path.isfile(fname))
        
        if recompute or file_not_found:

            # fit gaussian.
            x_data = data[time][:,0]
            y_data = data[time][:,1]
            pars = self._get_gaussian_res(x_data,y_data,time,
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
        list_hours = []
        for col in data_raw.columns:
            list_hours.append(col[0])

        list_hours = list(dict.fromkeys(list_hours))

        assert(list_hours[0] == 'control')

        # collect data in dict
        data_avg = {} # for average data
        data_rep = {} # for rep. data
        data_avg_raw = {}
        data_rep_raw = {}

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
                dr1 = np.diff(x1[mask])
                r1 = x1[mask][:-1]
                i1 = y1[mask][:-1]
                n1 = np.sum(i1*2*np.pi*r1*dr1)
            else:
                n1 = 1

            z1 = np.zeros((len(x1[mask]),2))
            z1[:,0] = x1[mask]; z1[:,1] = y1[mask]
            data_rep_raw[hour] = copy.deepcopy(z1)
            #print(z1[:,1][0])
            z1[:,1] /= n1
            data_rep[hour] = z1
            #print(z1[:,1][0])

            # save avg data
            avg_cell_number = cols[0][0]
            avg_data = data_raw[hour][avg_cell_number]
            avg_data.dropna(inplace=True)
            
            x2 = avg_data['radius']; y2 = avg_data['intensity']
            mask = (x2>=L0)&(x2<=L)

            if normed:
                dr2 = np.diff(x2[mask])
                r2 = x2[mask][:-1]
                i2 = y2[mask][:-1]
                n2 = np.sum(i2*2*np.pi*r2*dr2)
            else:
                n2 = 1
            
            z2 = np.zeros((len(x2[mask]),2))
            z2[:,0] = x2[mask]; z2[:,1] = y2[mask]
            data_avg_raw[hour] = copy.deepcopy(z2)
            z2[:,1] /= n2
            data_avg[hour] = z2

        return data_avg, data_rep, data_avg_raw, data_rep_raw

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

    def __init__(self,N=100,eps=0,
                 df=1,dp=1,
                 dp1=None,dp2=None,
                 T=300,dt=0.001,psource=0,
                 imax=1,order=1,D=1,
                 interp_o=1,
                 Nvel=1,us0=0.16,
                 u_nonconstant=False):
        
        """
        uval: scale for velocity.
        N: domain mesh size
        L: domain radius in um

        T: total time
        dt: time step
        psource: Dirichlet right boundary for mobile P

        order: 1st or 2nd order PDE (advection or advection + diffusion)
        D: diffusion constant
        interp_o: kind for linear interpolation. linear, quadratic, etc.

        Nvel: number of velocity points to use in nonconstant discrete velocity
        u_nonconstant: if true, use spatial velocity from PDE
        dp1 and dp2 are parameters for trapping via bundling.
        """
        
        super().__init__()

        #self.data_dir = data_dir

        self.Nvel = Nvel
        self.order = order
        self.interp_o = interp_o
        self.u_nonconstant = u_nonconstant

        if order == 1:
            self.rhs = self._fd1
        elif order == 2:
            self.rhs = self._fd2

        self.D = D
        self.imax = imax
            
        self.psource = psource

        self.eps = eps
        self.N = N
        
        self.r, self.dr = np.linspace(self.L0,self.L,N,retstep=True)
        
        self.dp = dp
        self.df = df

        self.dp1 = dp1
        self.dp2 = dp2

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
        return 1, but it is scaled by uval in code.
        needed to be compatible with user-defined velocities
        """
        
        return np.ones(len(r))

    
    def _fd1(self,t,y,scenario='default'):
        """
        finite diff for 1st order PDE
        t: float, time.
        y: 2*N array
        p: object of parameters
        """

        r = self.r
        dr = self.dr
        #u = self.u
        ur = self.ur
        f = y[:self.N]
        p = y[self.N:]

        out = self.du        

        if scenario[:-1] == 't1' or scenario[:-1] == 'jamming':
            tfp = self.dp*p[:-1] - self.df*f[:-1]
            out[self.N-1] = self.dp*p[-1] - self.df*f[-1]

        elif scenario[:-1] == 't2':
            tfp = self.dp1*p[:-1]*f[:-1] + self.dp2*p[:-1]**2 - self.df*f[:-1]
            out[self.N-1] = self.dp1*p[-1]*f[-1] + self.dp2*p[-1]**2 - self.df*f[-1]

        else:
            raise ValueError('Invalid Scenario', scenario)

        if scenario[:-1] == 'jamming':
            pr = (p[1:]-p[:-1])/dr
            drp = self.ur[1:]*(p[1:]/r[1:]*(1-p[1:]/self.imax) + pr*(1-2*p[1:]/self.imax))
        else:
            drp = (r[1:]*ur[1:]*p[1:]-r[:-1]*ur[:-1]*p[:-1])/(r[:-1]*dr)
            out[-1] = (-r[-1]*ur[-1]*p[-1])/(r[-1]*dr)

        out[:self.N-1] = tfp
        out[self.N:-1] = drp - tfp

        return out


    def _fd2(self,t,y,scenario='default'):
        """
        finite diff for 1st order PDE
        t: float, time.
        y: 2*N array
        p: object of parameters
        """

        r = self.r;dr = self.dr;u = self.u;D = self.D
        f = y[:self.N];p = y[self.N:];ur = self.ur
        
        tfp = self.dp*p - self.df*f
        
        # interior derivatives
        drp2_i = D*(p[2:] - 2*p[1:-1] + p[:-2])/dr**2
        drp1_i = (r[1:-1]*ur[1:-1]+D)*(p[2:]-p[:-2])/(2*r[1:-1]*dr)
        drp0_i = ur[1:-1]*p[1:-1]/r[1:-1]

        # left endpoint derivative
        p0 = p[1] + 2*ur[0]*dr*p[0]/D
        drp2_l = D*(p0 - 2*p[0] + p[1])/dr**2
        drp1_l = (r[0]*ur[0]+D)*(p[1]-p0)/(2*r[0]*dr)
        drp0_l = ur[0]*p[0]/r[0] - tfp[0]

        # right endpoint derivative
        pn1 = p[-2] - 2*ur[-1]*dr*p[-1]/D
        drp2_r = D*(pn1 - 2*p[-1] + p[-2])/dr**2
        drp1_r = (r[-1]*ur[-1]+D)*(pn1-p[-2])/(2*r[-1]*dr)
        drp0_r = ur[-1]*p[-1]/r[-1] - tfp[-1]

        # update interior derivatives
        
        out = self.du
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
        
        if self.Nvel == 1:
            if self.u_nonconstant:
                self.ur = self._s2_vel()
            else:
                self.u = self._u_constant
                self.ur = self.us0*self.u(self.r)
        else:
            us = []
            for i in range(self.Nvel):
                us1 = getattr(self,'us'+str(i))
                us.append(us1)
                
            if self.interp_o == 1:
                kind = 'linear'

            elif self.interp_o == 2:
                kind = 'quadratic'

            self.rs = np.linspace(self.L0,self.L,len(us))
            fn = interp1d(self.rs,us,kind=kind)
            self.ur = fn(self.r)
            
        
        y0 = np.zeros(2*self.N)

        y0[:self.N] = self.control_fn(self.r)*self.eps
        y0[self.N:] = self.control_fn(self.r)*(1-self.eps)


        if False:
            import matplotlib.pyplot as plt
            fig = plt.figure()
            ax = fig.add_subplot(111)
            ax.plot(self.r,y0[:self.N])
            ax.plot(self.r,y0[self.N:])
            #ax.plot(self.r,self.control_fn(self.r))
            ax.set_title('just before run euler)')
            plt.show()

        TN = int(self.T/self.dt)
        t = np.linspace(0,self.T,TN)
        y = np.zeros((2*self.N,TN))

        y[:,0] = y0
        
        
        for i in range(TN-1):
            #if i >= y[-1,i] = self.psource
            y[:,i+1] = y[:,i] + self.dt*self.rhs(t[i],y[:,i],scenario=scenario)


        if False:
            import matplotlib.pyplot as plt
            fig = plt.figure()
            ax = fig.add_subplot(111)
            ax.plot(self.r,self.ur)
            plt.show()
            plt.close()
            time.sleep(2)


        self.t = t
        self.y = y
        self.y0 = y0


    def _s2_vel(self):
        """
        take a look at velocity profile (see page 230 in personal notebook)
        """
        
        import matplotlib.pyplot as plt

        r = self.r
        #F = self.steadys_fn(r)
        f_last = self.data_avg_fns['24h'](r)
        p0 = self.data_avg_fns['control'](r)*(1-self.eps)
        f0 = self.data_avg_fns['control'](r)*self.eps

        fhat = f0-f_last
        
        dr = self.dr

        mu = r*fhat
        
        #c = 2*np.sum(mu)*dr/(self.L**2 - self.L0**2)
        
        #u = np.cumsum(r*(F-c))*dr/mu
        u = self.dp*np.cumsum(mu*(1+p0/fhat))*dr/mu
        
        # shift back 1 to include 0 value
        u = np.append([0],u)
        u = u[:-1]

        return u

    def u(self,r):
        """
        spatially-dependent u

        r: array of domain
        us: array-like of u values
        xs: corresponding array-like of r values
        """

        out = np.zeros_like(r)
        if self.Nvel == 1:
            out[:] = getattr(self,'us'+str(0))

        for i in range(self.Nvel-1):
            us1 = getattr(self,'us'+str(i))
            us2 = getattr(self,'us'+str(i+1))

            rs1 = self.rs[i]
            rs2 = self.rs[i+1]

            mask = (rs1<=r)&(r<=rs2)
            m = (us2-us1)/(rs2-rs1)
            
            out[mask] = m*(r[mask]-rs1) + us1
            
        return out

    
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
        return g_approx(x,self.pars)/100000

def parsets(scenario='default',method=''):
       
    
    if scenario == 'default':
        if method == 'annealing':
            pars = {'eps':3.87011903e-05,
                    'df':4.79557732e-03,
                    'dp':5.40722009e+00,
                    'uval':1.81417393e+00,
                    'T':1500,'dt':0.01
            }
    
    elif scenario == 0:
        if method == 'annealing':
            pars = {'eps':1,
                    'df':0,
                    'dp':0,
                    'uval':1.75861129,
                    'T':1500,'dt':0.01}
            
    elif scenario == 1:
        if method == 'annealing':
            pars = {'eps':1,
                    'df':0,
                    'dp':0,
                    'uval':1.25095189,
                    'T':1500,'dt':0.01}

    elif scenario == 2:
        if method == 'annealing':
            pars = {'eps':0,
                    'df':0,
                    'dp':0.05601278,
                    'uval':0.11115834,
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
                  fill_value='extrapolate')

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
    u = self.uval*p.u(r)

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

    I = fsol + psol

    if False:
        import matplotlib.pyplot as plt
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.set_title('in plot_sim')
        ax.plot(p.r,p.y[:p.N,0])
        ax.plot(p.r,p.y[p.N:,0])
        #ax.plot(p.r,p.control_fn(p.r))
        plt.show()

    
    nrows = 3
    ncols = 2
    
    fig = plt.figure(figsize=(7,7))
    gs = GridSpec(nrows,ncols,figure=fig)
    
    axs = []
    
    for i in range(nrows):
        axs.append([])
        for j in range(ncols):
            axs[i].append(fig.add_subplot(gs[i,j]))
    
    axs[0][0].imshow(fsol[::-1,::100],aspect='auto',
                    extent=(p.t[0],p.t[-1],p.r[0],p.r[-1]))

    axs[0][1].imshow(psol[::-1,::100],aspect='auto',
                    extent=(p.t[0],p.t[-1],p.r[0],p.r[-1]))

    axs[1][0].plot(p.r,p.y0[:p.N],label='Initial F')
    axs[1][0].plot(p.r,fsol[:,-1],label='Final F')
    #axs[1,0].plot(p.r,steadys_fn(p.r),label='Final/2')
    
    axs[1][1].plot(p.r,p.y0[p.N:],label='Initial P')
    axs[1][1].plot(p.r,psol[:,-1],label='Final P')
    #axs[1,1].plot(p.r,steadys_fn(p.r)/2,label='Final/2')

    data_ctrl = p.data_avg_fns['control'](p.r)
    data_ss = p.data_avg_fns['24h'](p.r)
    axs[2][0].plot(p.r[1:-2],data_ctrl[1:-2],label='Initial (data)')
    axs[2][0].plot(p.r,I[:,0],label='Initial F+P (PDE)')

    axs[2][1].plot(p.r[1:-1],data_ss[1:-1],label='Final (data)')
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
        axs[0,i].plot(p.r,I[:,idx],label='PDE')
        axs[0,i].plot(p.r[1:-1],data[1:-1],label='Data')
        
        axs[1,i].plot(p.r,F[:,idx])
        axs[2,i].plot(p.r,P[:,idx])

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

    if False:
        import matplotlib.pyplot as plt
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.plot(100000*data_y)
        plt.show()
        #time.sleep(10)
    
    err =  np.linalg.norm(g-100000*data_y)

    return err


def main():

    import matplotlib.pyplot as plt
    
    from matplotlib.gridspec import GridSpec


    #scenario = 'default'
    #scenario = 'model2a'
    scenario = 't1e'
    method = 'annealing'
    np.random.seed(3)

    #pars = parsets(scenario,method)
        
    #us = [0.00,0.05,0.09,0.13,0.15,0.17,0.21,0.27,0.36,0.45,0.55,0.72,0.64,1.00,1.00,1.00,0.93,0.61,0.07,0.42]
    #pars = {'eps':0.2453,'df':0,'dp':0.0895,'uval':-69,'T':1500}

    # n5 umax 1
    #us = [0.05066241, 0.2695232,  0.57624002, 1., 0.06706248]
    #pars = {'eps':0,'df':0,'dp':0.14370955,'uval':-69,'T':10}
    
    #pars = {'eps':.2502,'df':0,'dp':.0766,'T':1500,'dt':.01,'N':100,'Nvel':10,'u_nonconstant':False,'interp_o':2}
    
    #us = [0., 0.08, 0.14, 0.2, 0.35, 0.51, 0.78, 1, 0.84, 0.55]
    #for i in range(len(us)):
    #    setattr(p,'us'+str(i),us[i])

    #0.40490838
    #=0.0268, d_f=0.0000, dp=0.0328
    pars = {'eps':0.0268,'df':0,'dp':0.0328,'T':1500,'dt':.01,'N':100,'Nvel':1,'u_nonconstant':True}
    #pars = {'eps':0.0020,'dp1':1.9939,'dp2':1.8193,'Nvel':1,'T':1500}

    p = PDEModel(**pars)

    us = [0.01]
    for i in range(len(us)):
        setattr(p,'us'+str(i),us[i])
        
    if False:
        ur = p._s2_vel()
        #mu = 
        #mu = np.exp(np.cumsum((r[1:]*dF+F[1:])/dF)*dr)
        #print(mu)

        fig, axs = plt.subplots(nrows=1,ncols=1,figsize=(5,5))

        axs.plot(p.r,ur)

        """
        # check out derivatives as a function of c
        cs = np.linspace(-.1,1.2,10)
        for c_val in cs:
            up = 1 - c_val/F[1:] + (1/mu)*u
            zero_crossings = np.where(np.diff(np.sign(up)))[0]
            rs_zero = r[1:][zero_crossings]
            
            if len(rs_zero) > 0:
                
                print(c_val,rs_zero)
                axs[1].scatter(c_val*np.ones(len(rs_zero)),rs_zero,
                               s=10,color='tab:red')
        #axs[1].scatter(c,)
        axs[1].set_xlabel('Value of c')
        axs[1].set_ylabel('r')
        axs[1].set_xlim(cs[0],cs[-1])
        axs[1].set_title('Zero derivatives of u(r)')
        """
        
        axs.set_xlabel('r')
        #axs.set_title('u(r) with c='+str(c))        

        plt.tight_layout()
        plt.show()

    

    #time.sleep(10)
    #p.rs = np.linspace(p.L0,p.L,len(us))
    #p.Nvel = len(p.rs)
    
    #for i in range(len(us)):
    #    setattr(p,'us'+str(i),us[i])

    # get numerical solution
    p._run_euler(scenario)


    
    if False:
        
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.plot(p.r,p.ur)
        ax.scatter(p.rs,us)
        ax.set_xlabel('r')
        ax.set_ylabel('velocity')
        ax.set_title('u(r)')

        


    if False:
        import matplotlib.pyplot as plt
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.set_title('just after p._run_euler')
        ax.plot(p.r,p.y[:p.N,0])
        ax.plot(p.r,p.y[p.N:,0])
        #ax.plot(p.r,p.control_fn(p.r))


    # plot solution
    if True:
        
        plot_sim(p)
        plot_sim_intermediate(p)
        
        #plot_ana(p)

    plt.show()

if __name__ == "__main__":
    main()
