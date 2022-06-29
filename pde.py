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

import lib

import os
import time
import time as tt
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
                 L0=10,
                 L=29.5,normed=True):

        self.L = L
        self.L0 = L0
        self.recompute = recompute
        self.data_dir = lib.data_dir()
        self.normed = normed
        
        if not(os.path.isdir(self.data_dir)):
            print('created data directory at',self.data_dir)
            os.mkdir(self.data_dir)

        self._build_data_dict(L0=self.L0,L=self.L,
                              normed=self.normed)
        
        # interp1d functions -- linear interp on bounded domain
        self.data_avg_fns = self._build_data_fns(self.data_avg)
        self.data_rep_fns = self._build_data_fns(self.data_rep,rep=True)

        # generate functions
        #print('data keys',data_avg.keys())

        args = (self.data_dir,self.data_avg,'control')
        kwargs = {'normed':self.normed,'n_gauss':10,'recompute':self.recompute}
        pars_control_avg = self._load_gaussian_pars(*args,**kwargs)
        
        args = (self.data_dir,self.data_rep,'control')
        kwargs['rep'] = True
        pars_control_rep = self._load_gaussian_pars(*args,**kwargs)

        # gaussian interp fns. -- gaussian interp on R
        self.control_fn_avg = CallableGaussian(pars_control_avg)
        self.control_fn_rep = CallableGaussian(pars_control_rep)


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

        res = lsq(cost_fn_gauss,par_init,args=(x_data,y_data))

        # from least squares above
        pars = res.x
        return pars

    def _load_gaussian_pars(self,data_dir,data,time,normed,
                            n_gauss=6,recompute=False,rep=False):
        """
        time: str. time of data
        """

        fname = data_dir + '{}' + time + str(n_gauss) +'_normed=' + str(normed) + '.txt'
        if rep:
            fname = fname.format('rep_')
        else:
            fname = fname.format('avg_')

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
        
    #@staticmethod
    def _build_data_dict(self,fname='data/patrons20180327dynamiqueNZ_reformatted.xlsx',
                         L0=10,L=29.5,normed=True):

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

        data_avg_n = {}
        data_rep_n = {}

        for hour in list_hours:

            # get column names
            # ('13c', 'radius'), ('13c', 'intensity'), ('rep', 'radius'), ...
            cols = data_raw[hour].columns

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
                data_rep_n[hour] = n1
            else:
                n1 = 1
                
            z1 = np.zeros((len(x1[mask]),2))
            z1[:,0] = x1[mask]; z1[:,1] = y1[mask]
            data_rep_raw[hour] = copy.deepcopy(z1)

            z1[:,1] /= n1
            data_rep[hour] = z1

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
                data_avg_n[hour] = n2
            else:
                n2 = 1
            
            z2 = np.zeros((len(x2[mask]),2))
            z2[:,0] = x2[mask]; z2[:,1] = y2[mask]
            data_avg_raw[hour] = copy.deepcopy(z2)
            z2[:,1] /= n2
            data_avg[hour] = z2

        self.data_avg = data_avg
        self.data_rep = data_rep
        self.data_avg_raw = data_avg_raw
        self.data_rep_raw = data_rep_raw

        self.data_avg_n = data_avg_n
        print('avg n fn',self.data_avg_n)
        self.data_rep_n = data_rep_n
        

    @staticmethod
    def _build_data_fns(data,rep=False):#fill_value=0):
        """
        build 1d interpolation of data for use in cost function
        data is either data_avg or data_rep, the dicts constructed in 
        _build_data_dict
        """

        data_fn = {}

        for hour in data.keys():
            #print(data[hour][0,1])
            if rep:
                fn = get_1d_interp(data[hour],fill_value=(data[hour][0,1],0))
            else:
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
                 interp_o=1,L=29.5,L0=10,
                 Nvel=1,us0=0.16,
                 model=None):
        
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

        scenario: str, mechanism + scenario (e.g., t1d, t2a).
        """
        
        super().__init__(L=L,L0=L0)

        #self.data_dir = data_dir

        self.Nvel = Nvel
        self.order = order
        self.interp_o = interp_o
        self.us0 = us0

        self.model = model

        if self.model[-1] == 'e':
            self.u_nonconstant = True
            self.dp_nonconstant = False
            
        elif self.model[-1] == 'f':
            self.u_nonconstant = False
            self.dp_nonconstant = True
            
        else:
            self.u_nonconstant = False
            self.dp_nonconstant = False

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
    
    def _fd1(self,t,y):
        """
        finite diff for 1st order PDE
        t: float, time.
        y: 2*N array
        p: object of parameters
        """

        r = self.r
        dr = self.dr
        ur = self.ur; dpr = self.dpr
        
        f = y[:self.N]
        p = y[self.N:]
        out = self.du

        if self.model[:-1] == 't1' or self.model[:-1] == 'jamming':
            tfp = dpr[:-1]*p[:-1] - self.df*f[:-1]
            out[self.N-1] = dpr[-1]*p[-1] - self.df*f[-1]

        elif self.model[:-1] == 't2':
            tfp = self.dp1*p[:-1]*f[:-1] + self.dp2*p[:-1]**2 - self.df*f[:-1]
            out[self.N-1] = self.dp1*p[-1]*f[-1] + self.dp2*p[-1]**2 - self.df*f[-1]

        else:
            raise ValueError('Invalid Scenario', self.model)

        if self.model[:-1] == 'jamming':
            pr = (p[1:]-p[:-1])/dr
            drp = self.ur[1:]*(p[1:]/r[1:]*(1-p[1:]/self.imax) + pr*(1-2*p[1:]/self.imax))
        else:
            drp = (r[1:]*ur[1:]*p[1:]-r[:-1]*ur[:-1]*p[:-1])/(r[:-1]*dr)
            out[-1] = (-r[-1]*ur[-1]*p[-1])/(r[-1]*dr)

        out[:self.N-1] = tfp
        out[self.N:-1] = drp - tfp

        return out

    
    def _run_euler(self,rep=False):
        """
        t: time array
        y0: initial condition
        """
        
        if self.u_nonconstant:
            self.dpr = self.dp*np.ones(len(self.r))
            self.ur = self._vel_spatial()
            
        elif self.dp_nonconstant:
            self.dpr = self._dp_spatial()
            self.ur = self.us0*np.ones(len(self.r))
            
        else:
            self.dpr = self.dp*np.ones(len(self.r))
            self.ur = self.us0*np.ones(len(self.r))
            
        #print(self.ur)
        y0 = np.zeros(2*self.N)
        
        if rep:
            y0[:self.N] = self.control_fn_rep(self.r)*self.eps
            y0[self.N:] = self.control_fn_rep(self.r)*(1-self.eps)
            
        else:
            y0[:self.N] = self.control_fn_avg(self.r)*self.eps
            y0[self.N:] = self.control_fn_avg(self.r)*(1-self.eps)

        TN = int(self.T/self.dt)
        #print('TN',TN)
        t = np.linspace(0,self.T,TN)
        y = np.zeros((2*self.N,TN))

        y[:,0] = y0
        
        
        for i in range(TN-1):
            #if i >= y[-1,i] = self.psource
            y[:,i+1] = y[:,i] + self.dt*self.rhs(t[i],y[:,i])
            y[-1,i+1] = 0
            #y[-1] = y[-2]


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


    def _vel_spatial(self):
        """
        take a look at velocity profile (see page 230 in personal notebook)
        """
        
        import matplotlib.pyplot as plt

        r = self.r

        f_last = self.data_avg_fns['24h'](r)
        p0 = self.data_avg_fns['control'](r)*(1-self.eps)
        f0 = self.data_avg_fns['control'](r)*self.eps        

        fhat = f0-f_last
        
        dr = self.dr

        mu = r*fhat
        
        #u = np.cumsum(r*(F-c))*dr/mu
        u = self.dp*np.cumsum(mu*(1+p0/fhat))*dr/mu
        
        # shift back 1 to include 0 value
        u = np.append([0],u)
        u = u[:-1]

        return u


    def _dp_spatial(self):
        """
        rate profile
        """
        
        import matplotlib.pyplot as plt

        r = self.r

        f_last = self.data_avg_fns['24h'](r)
        p0 = self.data_avg_fns['control'](r)*(1-self.eps)
        f0 = self.data_avg_fns['control'](r)*self.eps        

        fhat = -(f0-f_last)
        
        dr = self.dr
        #mu = r*fhat
        
        #u = np.cumsum(r*(F-c))*dr/mu
        q = np.cumsum(r*(fhat-p0))*dr
        dp = r*self.us0*fhat/q
        
        # shift back 1 to include 0 value
        return dp

    
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


def cost_fn(x,p,par_names=None,ss_condition=False,psource=False):
    """
    copy of cost function from model_fitting.
    x is combination or subset of eps,df,dp.
    par_names: list of variables in order of x
    returns L2 norm.
    """
    assert(len(x) == len(par_names))
    
    for i,val in enumerate(x):
        setattr(p,par_names[i],val)

    TN = int(p.T/p.dt)

    p._run_euler()
    y = p.y

    if False:
        import matplotlib.pyplot as plt
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.plot(p.r,y[:p.N,0])
        ax.plot(p.r,y[:p.N,int(TN/4)])
        ax.plot(p.r,y[:p.N,int(TN/2)])
        ax.plot(p.r,y[:p.N,-1])
        plt.show()
        plt.close()
        tt.sleep(1)

    # get solution
    fsol = y[:p.N,:]
    psol = y[p.N:,:]

    I = fsol + psol

    #'2h', '4h', '30min', '1h', '24h', 'control', '8h30'
    err = 0

    for hour in p.data_avg.keys():

        # convert hour to index
        if hour == 'control':
            pass # ignore initial condition (trivial)
        else:

            time = float(hour[:-1])
            minute = time*60
            idx = int(minute/p.dt)

            # restrict solution to observed positions
            I_fn = interp1d(p.r,I[:,idx])
            I_cut = I_fn(p.data_avg[hour][:,0])

            data = p.data_avg[hour][:,1]
            err0 = np.linalg.norm(data[1:-1]-I_cut[1:-1])**2
            print(err0)
            err += err0
            

    err = np.log10(err)
    if ss_condition:
        if 1e6*np.linalg.norm(I[:,int(1200/p.dt)]-I[:,int(1440/p.dt)])**2 > 1e-10:
            err = 1e5

    #err_log = np.log10(err)
    if np.isnan(err):
        err = 1e5

    stdout = [err,p.eps,p.df]
    s1 = 'log(err)={:.4f}, eps={:.4f}, '\
        +'df={:.4f}'

    if not(p.dp_nonconstant):
        stdout.append(p.dp)
        s1 += ', dp={:.4f}'

    if psource:
        stdout.append(p.psource)
        s1 += ', psource={:.4f}'

    if p.model[:-1] == 't2':
        stdout.append(p.dp1);stdout.append(p.dp2)
        s1 += ', dp1={:.4f}, dp2={:.4f}'

    if p.model[:-1] == 'jamming':
        stdout.append(p.imax)
        s1 += ', imax={:.4f}'

    if not(p.u_nonconstant):
        stdout.append(getattr(p,'us'+str(0)))
        s1 += ', us={:.2f}'

    #print(s1,stdout)
    print(s1.format(*stdout))

    return err

    
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

def get_1d_interp(data,kind='linear',fill_value='extrapolate'):
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
                  fill_value=fill_value)

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

    p_fn = p.control_fn_avg
    
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
        #ax.plot(p.r,p.control_fn_avg(p.r))
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
    
    axs[1][1].plot(p.r,p.y0[p.N:],label='Initial P')
    axs[1][1].plot(p.r,psol[:,-1],label='Final P')

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
    
def cost_fn_gauss(x,data_x,data_y):
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

    model = 't1f'
    method = 'de'#'annealing'
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
    #0.01041775
    pars = {'eps':0.5,'df':0,'dp':0.01041775,'us0':.067,
            'T':1500,'dt':.05,'model':model}
    #pars = {'eps':0.0020,'dp1':1.9939,'dp2':1.8193,'Nvel':1,'T':1500}

       
    p = PDEModel(**pars)

    # get cost function information
    #err = cost_fn([.02,0.067],p,par_names=['eps','us0'],ss_condition=True)
    #err = cost_fn([0.0,0.010417365875774465],p,par_names=['eps','dp'],ss_condition=True)
    #print(err)
    
            
    if False:
        ur = p._vel_spatial()

        fig, axs = plt.subplots(nrows=1,ncols=1,figsize=(5,5))

        axs.plot(p.r,ur)        
        axs.set_xlabel('r')

        plt.tight_layout()
        plt.show()


    if True:
        dp = p._dp_spatial()

        fig, axs = plt.subplots(nrows=1,ncols=1,figsize=(5,5))

        axs.plot(p.r,dp)        
        axs.set_xlabel('r')

        plt.tight_layout()
        plt.show()
    

    # get numerical solution
    p._run_euler()


    
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
        #ax.plot(p.r,p.control_fn_avg(p.r))


    # plot solution
    if True:
        
        #plot_sim(p)
        plot_sim_intermediate(p)
        
        #plot_ana(p)

    plt.show()

if __name__ == "__main__":
    main()
