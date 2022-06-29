import matplotlib as mpl
from scipy import ndimage

#from matplotlib import rc
#mpl.rc('text',usetex=True)
#mpl.rc('font',family='serif', serif=['Computer Modern Roman'])

#mpl.rcParams.update(pgf_with_latex)
mpl.rcParams['font.family'] = 'serif'
mpl.rcParams['text.usetex'] = True
#mpl.rcParams['text.latex.preview'] = True
mpl.rcParams['pgf.texsystem'] = 'pdflatex'
mpl.rcParams['text.latex.preamble'] = r'\usepackage{amsmath} \usepackage{siunitx} \usepackage{enumitem} \usepackage{xcolor}'

fsizetick = 13
fsizelabel = 13
fsizetitle = 13
#tight_layout_pad =  0

from matplotlib.patches import Rectangle


import model_fitting as mf
import sensitivity as sv
import pde
import lib
import copy
import numpy as np

from scipy.interpolate import interp1d
import os
import multiprocessing

import warnings
warnings.filterwarnings('ignore')
#np.seterr(under=None)


def experiment_figure():
    """
    figure for experimental setup
    """
    import matplotlib.pyplot as plt
    import matplotlib.image as mpimg
    from matplotlib.gridspec import GridSpec

    fig = plt.figure(figsize=(5,5))

    gs = GridSpec(2,2,height_ratios=[2,1])
    
    axs = []
    axs.append(fig.add_subplot(gs[0,0]))
    axs.append(fig.add_subplot(gs[0,1]))
    axs.append(fig.add_subplot(gs[1,:]))
    
    # plot cell images
    img_ctrl = mpimg.imread('i_controlmicro.png')
    img_noco = mpimg.imread('i_nocomicro.png')

    axs[0].imshow(ndimage.rotate(img_ctrl,180),aspect='equal')
    axs[1].imshow(ndimage.rotate(img_noco,180),aspect='equal')

    # show horizontal lines corresponding to data
    axs[0].plot([.5,1-.08],[.5,.5],transform=axs[0].transAxes,
                color='tab:red',ls='--',marker='o')
    axs[1].plot([.5,1-.08],[.5,.5],transform=axs[1].transAxes,
                color='tab:blue',ls='-',marker='o')
    
    
    # plot discarded circular region
    circle1 = plt.Circle((.5,.5),0.14,color='tab:red',alpha=.3,zorder=4,
                         hatch='x',transform=axs[0].transAxes)
    circle2 = plt.Circle((.5,.5),0.14,color='tab:red',alpha=.3,zorder=4,
                         hatch='x',transform=axs[1].transAxes)
    
    axs[0].add_patch(circle1)
    axs[1].add_patch(circle2)
    
    # plot vimentin data
    d = pde.Data(L0=0,L=29.5)
    x = d.data_rep_raw['control'][:,0];y = d.data_rep_raw['control'][:,1]
    axs[2].plot(x,y,label=r'\SI{0}{h}',color='tab:red',ls='--')

    x = d.data_rep_raw['24h'][:,0];y = d.data_rep_raw['24h'][:,1]
    axs[2].plot(x,y,label=r'\SI{24}{h}',color='tab:blue',ls='-')

    # plot discarded region
    axs[2].set_xlim(0,29.5)
    axs[2].axvspan(0, 10, color='tab:red', alpha=0.1,hatch='x')
    axs[2].get_position().x1
    axs[2].text(1/6+.02,.07,"Discarded Region",ha='center',transform=axs[2].transAxes)

    # labels, axis tweaks
    axs[0].set_title(r'(a) \SI{0}{h}',loc='left',size=fsizetitle)
    axs[1].set_title(r'(b) \SI{24}{h}',loc='left',size=fsizetitle)
    axs[2].set_title(r'(c)',x=-.12,size=fsizetitle)

    axs[2].xaxis.get_major_ticks()[0].set_visible(False)
    axs[2].annotate(r'Cell Center',xy=(0, -.01), xycoords='axes fraction', xytext=(0.,-.5), arrowprops=dict(arrowstyle='->', color='k'),ha='center',fontsize=fsizelabel)

    axs[2].annotate(r'Cell Edge',xy=(1, -.01), xycoords='axes fraction', xytext=(1.,-.5), arrowprops=dict(arrowstyle='->', color='k',relpos=(1,0)),ha='right',fontsize=fsizelabel)
    
    axs[2].set_xlabel(r'Radius ($\si{\um}$)',fontsize=fsizelabel)
    axs[2].set_ylabel('Fluor. Intensity\n(a.u.)',fontsize=fsizelabel)

    axs[2].tick_params(axis='both',labelsize=fsizetick)    
    axs[2].legend()

    axs[0].axis('off')
    axs[1].axis('off')

    axs[2].set_xlim(0,29.5)
    
    fig.subplots_adjust(top=.97,right=.98,left=.1,bottom=0.15)
    
    chartBox = axs[2].get_position()
    x_shift = .14
    y_shift = 0
    axs[2].set_position([chartBox.x0+x_shift,chartBox.y0+y_shift,
                         chartBox.width-x_shift,chartBox.height-y_shift])
    
    return fig


def get_y(radii,x,y,maxr):
    """
    get closest y values given radius
    assume domain is [0,30].
    """

    x_list = []
    y_list = []
    
    for i in range(len(radii)):

        idx = np.argmin(np.abs(radii[i]/maxr*29.5-x))
        y_list.append(y[idx])
        x_list.append(x[idx])
    
    return np.array(x_list),np.array(y_list)
    

def experiment_figure2():
    """
    figure for experimental setup
    """
    import matplotlib.pyplot as plt
    import matplotlib.image as mpimg
    import matplotlib
    
    from matplotlib.gridspec import GridSpec

    fig = plt.figure(figsize=(5,5))

    gs = GridSpec(2,2,height_ratios=[2,1])
    
    axs = []
    axs.append(fig.add_subplot(gs[0,0]))
    axs.append(fig.add_subplot(gs[0,1]))
    axs.append(fig.add_subplot(gs[1,:]))
    
    # plot cell images
    img_ctrl = mpimg.imread('i_controlmicro.png')
    img_noco = mpimg.imread('i_nocomicro.png')

    axs[0].imshow(ndimage.rotate(img_ctrl,180),aspect='equal')
    axs[1].imshow(ndimage.rotate(img_noco,180),aspect='equal')

    # show concentric circles corresponding to data
    cmap1 = matplotlib.cm.get_cmap('YlGn');cmap2 = matplotlib.cm.get_cmap('Blues')
    minr1=.19;maxr1=(0.83)/2;x1=0.485;y1=0.5
    minr2=.19;maxr2=(0.83)/2;x2=0.505;y2=0.515
    
    radii1 = np.linspace(minr1,maxr1,3)[:-1];radii2 = np.linspace(minr2,maxr2,3)[:-1]
    thetas = np.linspace(0,2*np.pi,100)
    colors1 = np.linspace(.8,.5,len(radii1));colors2 = np.linspace(.8,.4,len(radii2))

    for i in range(len(radii1)):
        axs[0].plot(radii1[i]*np.cos(thetas)+x1,radii1[i]*np.sin(thetas)+y1,transform=axs[0].transAxes,
                    color=cmap1(colors1[i]),alpha=1,lw=2)

    for i in range(len(radii2)):
        axs[1].plot(radii2[i]*np.cos(thetas)+x2,radii2[i]*np.sin(thetas)+y2,transform=axs[1].transAxes,
                    color=cmap2(colors2[i]),alpha=1,lw=2)

    #axs[0].plot([.5,1-.08],[.5,.5],transform=axs[0].transAxes,
    #            color='tab:red',ls='--',marker='o')
    #axs[1].plot([.5,1-.08],[.5,.5],transform=axs[1].transAxes,
    #            color='tab:blue',ls='-',marker='o')
        
    # plot discarded circular region
    circle1 = plt.Circle((.485,.5),0.14,color='tab:red',alpha=.3,zorder=4,
                         hatch='x',transform=axs[0].transAxes)
    circle2 = plt.Circle((.505,.515),0.14,color='tab:red',alpha=.3,zorder=4,
                         hatch='x',transform=axs[1].transAxes)
    
    axs[0].add_patch(circle1)
    axs[1].add_patch(circle2)
    
    # plot vimentin data
    d = pde.Data(L0=0,L=29.5)
    x = d.data_rep_raw['control'][:,0];y = d.data_rep_raw['control'][:,1]
    axs[2].plot(x,y,label=r'\SI{0}{h}',color='tab:green',ls='--')
    
    x,y = get_y(radii1,x,y,maxr1)
    axs[2].scatter(x,y,color=cmap1(colors1),marker='x',s=100,linewidths=3)

    x = d.data_rep_raw['24h'][:,0];y = d.data_rep_raw['24h'][:,1]
    axs[2].plot(x,y,label=r'\SI{24}{h}',color='tab:blue',ls='-')

    x,y = get_y(radii2,x,y,maxr2)
    axs[2].scatter(x,y,color=cmap2(colors2),marker='x',s=100,linewidths=3)

    # plot discarded region
    axs[2].set_xlim(0,29.5)
    axs[2].axvspan(0, 10, color='tab:red', alpha=0.1,hatch='x')
    axs[2].get_position().x1
    axs[2].text(1/6+.02,.07,"Discarded Region",ha='center',transform=axs[2].transAxes)

    # labels, axis tweaks
    axs[0].set_title(r'(a) \SI{0}{h}',loc='left',size=fsizetitle)
    axs[1].set_title(r'(b) \SI{24}{h}',loc='left',size=fsizetitle)
    axs[2].set_title(r'(c)',x=-.12,size=fsizetitle)

    axs[2].xaxis.get_major_ticks()[0].set_visible(False)
    axs[2].annotate(r'Cell Center',xy=(0, -.01), xycoords='axes fraction', xytext=(0.,-.5), arrowprops=dict(arrowstyle='->', color='k'),ha='center',fontsize=fsizelabel)

    axs[2].annotate(r'Cell Edge',xy=(1, -.01), xycoords='axes fraction', xytext=(1.,-.5), arrowprops=dict(arrowstyle='->', color='k',relpos=(1,0)),ha='right',fontsize=fsizelabel)
    
    axs[2].set_xlabel(r'Radius ($\si{\um}$)',fontsize=fsizelabel)
    axs[2].set_ylabel('Fluor. Intensity\n(a.u.)',fontsize=fsizelabel)

    axs[2].tick_params(axis='both',labelsize=fsizetick)    
    axs[2].legend()

    axs[0].axis('off')
    axs[1].axis('off')

    axs[2].set_xlim(0,29.5)
    
    fig.subplots_adjust(top=.97,right=.98,left=.05,bottom=0.15)
    
    chartBox = axs[2].get_position()
    x_shift = .14
    y_shift = 0
    axs[2].set_position([chartBox.x0+x_shift,chartBox.y0+y_shift,
                         chartBox.width-x_shift,chartBox.height-y_shift])
    
    return fig


def data_figure():
    """
    figure comparing representative data, average data
    and normalized versions of each
    """
    import matplotlib.pyplot as plt

    # load normed data
    d = pde.Data()

    avg_n_dict = copy.deepcopy(d.data_avg_n)
    rep_n_dict = copy.deepcopy(d.data_rep_n)

    d = pde.Data(L0=0,L=29.5)

    # load unnormed data
    #data =  d._build_data_dict(L0=d.L0,L=d.L,normed=True)
    
    #list_hours = ['control','0.5h', '1h','2h','4h','8.5h','24h']
    list_hours = ['control','0.5h', '1h','2h','24h']
    #assert(len(list_hours) == len(d.data_avg.keys()))
    #assert(len(d.data_avg.keys()) == len(d.data_rep.keys()))
    
    #fig = plt.figure()
    fig,axs = plt.subplots(nrows=4,ncols=1,figsize=(5,6))
    
    for i,hour in enumerate(list_hours):

        color = str((1-i/len(list_hours))/1.2)

        if hour == 'control':
            label = r'\SI{0}{h}'
        else:
            
            label = r'\SI{'+hour[:-1]+'}{h}'

        if hour == '24h':
            color = 'tab:blue'
        else:
            color = color

        # raw, rep
        x = d.data_rep_raw[hour][:,0]
        y = d.data_rep_raw[hour][:,1]
        axs[0].plot(x,y,label=label,color=color)

        # raw, avg
        x = d.data_avg_raw[hour][:,0]
        y = d.data_avg_raw[hour][:,1]
        axs[1].plot(x,y,label=label,color=color)
        
        # normed, rep
        x = d.data_rep_raw[hour][:,0]
        y = d.data_rep_raw[hour][:,1]/rep_n_dict[hour]
        #x = d.data_rep[hour][:,0]
        #y = d.data_rep[hour][:,1]
        axs[2].plot(x,y,label=label,color=color)

        # normed, avg
        x = d.data_avg_raw[hour][:,0]
        y = d.data_avg_raw[hour][:,1]/avg_n_dict[hour]
        #x = d.data_avg[hour][:,0]
        #y = d.data_avg[hour][:,1]
        axs[3].plot(x,y,label=label,color=color)

        
    axs[0].legend(labelspacing=-.1,bbox_to_anchor=(.8,.15),framealpha=1)
    
    for i in range(4):
            
        #axs[i,j].set_xlabel(r'Radius ($\si{\um}$)',fontsize=fsizelabel)
        axs[i].set_xlabel(r'Radius $r$ ($\si{\um}$)',fontsize=fsizelabel)
        axs[i].tick_params(axis='both',labelsize=fsizetick)
        
        axs[i].set_xlim(0,29.5)
        axs[i].axvspan(0, 10, color='tab:red', alpha=0.1,hatch='x')
        
        axs[i].get_position().x1
        axs[i].text(1/6,.07,"Discarded Region",ha='center',transform=axs[i].transAxes)

        axs[i].xaxis.labelpad=0


    axs[0].set_title('(a) Representative',loc='left',size=fsizetitle,y=.95)
    axs[1].set_title('(b) Average',loc='left',size=fsizetitle,y=.95)
    axs[2].set_title('(c) Representative (Normalized)',loc='left',size=fsizetitle,y=.95)
    axs[3].set_title(r'\textbf{(d) Average (Normalized)}',loc='left',size=fsizetitle,y=.95)
    
    axs[0].set_ylabel('Fluor. Intensity',fontsize=fsizelabel)
    axs[1].set_ylabel('Fluor. Intensity',fontsize=fsizelabel)
    axs[2].set_ylabel('Norm. Fluor.',fontsize=fsizelabel)
    axs[3].set_ylabel('Norm. Fluor.',fontsize=fsizelabel)

    #plt.tight_layout(pad=tight_layout_pad)
    plt.tight_layout()

    fig.subplots_adjust(top=.96,right=.95,left=.2,bottom=0.07,hspace=.8)
    

    return fig


def data_figure2(lower=10):
    """
    new data figure assuming we don't perform a validation
    figure comparing representative data, average data
    and normalized versions of each
    """
    
    import matplotlib.pyplot as plt
    import matplotlib.ticker as ticker
    from mpl_toolkits.axes_grid1.inset_locator import inset_axes
    
    # load normed data
    d = pde.Data(L0=10,L=29.5)

    avg_n_dict = copy.deepcopy(d.data_avg_n)
    rep_n_dict = copy.deepcopy(d.data_rep_n)

    d = pde.Data(L0=0,L=29.5)

    # load unnormed data
    #data =  d._build_data_dict(L0=d.L0,L=d.L,normed=True)    
    list_hours = ['control','0.5h', '1h','2h','4h','8.5h','24h']
    
    #fig = plt.figure()
    fig,axs = plt.subplots(nrows=2,ncols=1,figsize=(5,4))
    
    for i,hour in enumerate(list_hours):

        color = str((1-i/len(list_hours))/1.3)

        if hour == 'control':
            label = r'\SI{0}{h}'
        else:
            
            label = r'\SI{'+hour[:-1]+'}{h}'

        if hour == '24h':
            color = 'tab:blue'
        else:
            color = color

        # raw, avg
        x0 = d.data_avg_raw[hour][:,0]
        y0 = d.data_avg_raw[hour][:,1]
        
        axs[0].plot(x0[x0>=lower],y0[x0>=lower],label=label,color=color)
        #axs[0].plot(x0,y0,label=label,color=color)
        
        # normed, avg
        x1 = d.data_avg_raw[hour][:,0]
        y1 = d.data_avg_raw[hour][:,1]/avg_n_dict[hour]
        #x1 = d.data_avg[hour][:,0]
        #y1 = d.data_avg[hour][:,1]
        axs[1].plot(x1[x1>=lower],y1[x1>=lower],label=label,color=color)
        #axs[1].plot(x1,y1,label=label,color=color)
        
    #axs[1].legend(labelspacing=-.1,bbox_to_anchor=(.55,.6),framealpha=1,ncol=2)
    axs[0].legend(labelspacing=-.1,bbox_to_anchor=(.805,.35),framealpha=1)
    
    for i in range(2):
        axs[i].set_xlabel(r'Radius $r$ ($\si{\um}$)',fontsize=fsizelabel)
        axs[i].tick_params(axis='both',labelsize=fsizetick)        

        axs[i].xaxis.labelpad=0

        if lower == 0:
            axs[i].set_xlim(0,29.5)
            axs[i].axvspan(0, 10, color='tab:red', alpha=0.1,hatch='x')
            
            axs[i].get_position().x1
            axs[i].text(1/6,.07,"Discarded Region",ha='center',transform=axs[i].transAxes)

    axs[0].set_title('(a) Average',loc='left',size=fsizetitle)
    axs[1].set_title(r'\textbf{(b) Average (Normalized)}',loc='left',size=fsizetitle)
    
    axs[0].set_ylabel('Fluor. Intensity (a.u.)',fontsize=fsizelabel)
    axs[1].set_ylabel('Norm. Fluorescence',fontsize=fsizelabel)

    # rescale to percent
    ticks_y = ticker.FuncFormatter(lambda x, pos: '{0:g}\%'.format(x*100))
    axs[1].yaxis.set_major_formatter(ticks_y)
    
    lim0 = axs[0].set_xlim(lower,29.5)
    lim1 = axs[1].set_xlim(lower,29.5)

    axs[0].set_xticks(list(axs[0].get_xticks())+[29.5])
    axs[1].set_xticks(list(axs[1].get_xticks())+[29.5])

    a0 = axs[0].get_xticks().tolist()
    a1 = axs[1].get_xticks().tolist()
    
    a0[0]=r'$L_0$';a0[-1]=r'$L$'
    a1[0]=r'$L_0$';a1[-1]=r'$L$'

    axs[0].set_xticklabels(a0)
    axs[1].set_xticklabels(a1)

    axs[0].set_xlim(lim0)
    axs[1].set_xlim(lim1)

    plt.tight_layout()
    fig.subplots_adjust(top=.93,right=.96,left=.15,bottom=0.11,hspace=.5)

    # Gaussian plot inset for panel (b)
    axins = inset_axes(axs[1],width="40%", height="40%",loc=1)
    
    d = pde.Data(recompute=False,normed=True)
    x_data = d.data_avg['control'][:,0]
    y_data = d.data_avg['control'][:,1]

    #axins.plot(x_data,y_data,label='Data',lw=4,color='tab:green',ls='--')
    #axins.plot(x_data,d.control_fn_avg(x_data),label='Approx.',lw=2,color='k')
    
    axins.plot(x_data,y_data,label='\SI{0}{h}',lw=3,color=str(1/1.3))
    axins.plot(x_data,d.control_fn_avg(x_data),label='Approx.',lw=1.5,color='tab:green',ls='--')

    #axins.set_xlabel(r'$r$ (\si{\um})',fontsize=fsizelabel,labelpad=-11)
    axins.set_ylabel(r'$\tilde V_0(r)$',fontsize=fsizelabel)
    axins.tick_params(axis='both',labelsize=fsizetick)

    axins.set_xlim(x_data[0],x_data[-1])
    axins.legend(labelspacing=-.1,fontsize=8,frameon=False)

    axins.tick_params(labelleft=False)
    axins.set_xticks([10,29.5])
    axins.set_xticklabels([r'$L_0$',r'$L$'])
    

    return fig


def gaussian_fit():
    """
    figure for fitting Gaussian to data
    """

    import matplotlib.pyplot as plt
    import matplotlib.ticker as mticker
    
    fig, axs = plt.subplots(nrows=1,ncols=2,figsize=(5,2))

    d = pde.Data(recompute=False,normed=True)
    
    x_data = d.data_avg['control'][:,0]
    y_data = d.data_avg['control'][:,1]
    axs[0].plot(x_data,y_data,label='Data',lw=4,color='tab:green',ls='--')
    axs[0].plot(x_data,d.control_fn_avg(x_data),label='Approx.',lw=2,color='k')

    max1 = np.amax(y_data)
    
    x_data = d.data_rep['control'][:,0]
    y_data = d.data_rep['control'][:,1]
    axs[1].plot(x_data,y_data,label='Data',lw=4,color='tab:green',ls='--')
    axs[1].plot(x_data,d.control_fn_rep(x_data),label='Approx.',lw=2,color='k')

    axs[0].set_xlabel(r'$r$ (\si{\um})',fontsize=fsizelabel)
    axs[0].set_ylabel(r'$\tilde V_0(r)$',fontsize=fsizelabel)
    axs[0].tick_params(axis='both',labelsize=fsizetick)

    axs[1].set_xlabel(r'$r$ (\si{\um})',fontsize=fsizelabel)
    axs[1].set_ylabel(r'$\tilde V_0(r)$',fontsize=fsizelabel)
    axs[1].tick_params(axis='both',labelsize=fsizetick)
    
    axs[0].set_xlim(x_data[0],x_data[-1])
    axs[1].set_xlim(x_data[0],x_data[-1])
    axs[0].legend(labelspacing=-.1)

    axs[0].set_title("(a) Average \SI{0}{h}",loc='left')
    axs[1].set_title("(b) Representative \SI{0}{h}",loc='left')

    #f = mticker.ScalarFormatter(useOffset=False, useMathText=True)
    #g = lambda x,pos : '${}$'.format(f._formatSciNotation('%1.10e' % x))
    #axs[0].yaxis.set_major_formatter(mticker.FuncFormatter(g))
    #axs[1].yaxis.set_major_formatter(mticker.FuncFormatter(g))
    #axs[0].ticklabel_format(axis='y',style='sci',scilimits=(0,0),useOffset=False)

    #axs[0].set_ylim(0,max1+max1/10)
    plt.tight_layout()

    fig.subplots_adjust(hspace=.8,top=.88,bottom=.25)

    return fig



def gaussian_fit2():
    """
    figure for fitting Gaussian to data
    same as above but without representative data.
    """

    import matplotlib.pyplot as plt
    
    fig, axs = plt.subplots(nrows=1,ncols=1,figsize=(5,2))

    d = pde.Data(recompute=False,normed=True)
    
    x_data = d.data_avg['control'][:,0]
    y_data = d.data_avg['control'][:,1]
    axs.plot(x_data,y_data,label='Data',lw=4,color='tab:green',ls='--')
    axs.plot(x_data,d.control_fn_avg(x_data),label='Approx.',lw=2,color='k')

    axs.set_xlabel(r'$r$ (\si{\um})',fontsize=fsizelabel)
    axs.set_ylabel(r'$\tilde V_0(r)$',fontsize=fsizelabel)
    axs.tick_params(axis='both',labelsize=fsizetick)
    
    axs.set_xlim(x_data[0],x_data[-1])
    axs.legend(labelspacing=-.1)
    
    # rescale to percent
    import matplotlib.ticker as ticker
    ticks_y = ticker.FuncFormatter(lambda x, pos: '{0:g}\%'.format(x*100))
    axs.yaxis.set_major_formatter(ticks_y)

    #axs.set_title("(a) Average \SI{0}{h}",loc='left')
    plt.tight_layout()

    fig.subplots_adjust(hspace=.8,top=.9,bottom=.25)

    return fig


def solution_schematic():

    import matplotlib.pyplot as plt
    import matplotlib.ticker as ticker
    
    pars = {'eps':.2,'df':0,'dp':.01,'T':1500,'dt':.01,'N':100,'model':'t1e'}

    p = pde.PDEModel(**pars)
    p._run_euler()
        
    #p = PDEModel()
    L0=p.L0; L=p.L
    
    nrows=5;ncols=2
    
    fig,axs = plt.subplots(nrows=nrows,ncols=ncols,figsize=(8,8),sharey='col')

    r = np.linspace(L0,L)
    m = 1/(L0-L)

    # initial
    f1 = p.y[:p.N,0]
    p1 = p.y[p.N:,0]
    
    axs[2,0].plot(p.r,f1,color='tab:blue',lw=2)
    axs[2,1].plot(p.r,p1,color='tab:orange',lw=2)

    # intermediate
    f2 = p.y[:p.N,int(2*60/p.dt)]
    p2 = p.y[p.N:,int(2*60/p.dt)]
    #print(len(p2),len(r))

    axs[3,0].plot(p.r,p.y[:p.N,int(1.2*60/p.dt)],color='tab:blue',lw=2,alpha=.2)
    axs[3,0].plot(p.r,p.y[:p.N,int(1.6*60/p.dt)],color='tab:blue',lw=2,alpha=.5)
    axs[3,0].plot(p.r,f2,color='tab:blue',lw=2)

    axs[3,1].plot(p.r,p.y[p.N:,int(1.6*60/p.dt)],color='tab:orange',lw=2,alpha=.2)
    axs[3,1].plot(p.r,p.y[p.N:,int(1.8*60/p.dt)],color='tab:orange',lw=2,alpha=.5)
    axs[3,1].plot(p.r,p2,color='tab:orange',lw=2)
    
    # steady-state
    f3 = p.y[:p.N,-1]
    p3 = p.y[p.N:,-1]

    axs[4,0].plot(p.r,f3,color='tab:blue',lw=2)
    axs[4,1].plot(p.r,p3,color='tab:orange',lw=2)
    
    #axs[2,0].text((L+L0)/2,2.7,'Initial',ha='center',size=15)
    axs[3,0].text((L+L0)/2,2.7,'',ha='center',size=15)

    plt.text(.12,.94,"(a)",transform=fig.transFigure,size=fsizetitle)
    plt.text(.12,.83,"(b)",transform=fig.transFigure,size=fsizetitle)
    
    axs[2,0].set_title(r'(c) Initial ($t=0$)',loc='left',size=fsizetitle)
    axs[3,0].set_title(r'(d) $t>0$',loc='left',size=fsizetitle)
    axs[4,0].set_title(r'(e) Steady-State ($t\rightarrow \infty$)',loc='left',size=fsizetitle)
    
    axs[2,1].set_title(r'(f) Initial ($t=0$)',loc='left',size=fsizetitle)
    axs[3,1].set_title(r'(g) $t>0$',loc='left',size=fsizetitle)
    axs[4,1].set_title(r'(h) Steady-State  ($t\rightarrow \infty$)',loc='left',size=fsizetitle)
    
    axs[2,0].set_ylabel('$I$',fontsize=fsizelabel)
    axs[3,0].set_ylabel('$I$',fontsize=fsizelabel)
    axs[4,0].set_ylabel('$I$',fontsize=fsizelabel)
    
    axs[2,1].set_ylabel('$M$',fontsize=fsizelabel)
    axs[3,1].set_ylabel('$M$',fontsize=fsizelabel)
    axs[4,1].set_ylabel('$M$',fontsize=fsizelabel)
    

    for i in range(2):
        for j in range(2):
            #axs[j,i].set_xticks([])
            #axs[j,i].set_yticks([])
            axs[j,i].axis('off')
    

    for i in range(2,nrows):
        for j in range(ncols):
            # remove y axis label
            axs[i,j].tick_params(axis='both',labelsize=fsizetick)
            #axs[i,j].tick_params(axis='y',which='both',left=False,labelleft=False)

            # x axis label
            axs[i,j].set_xticks([L0,L])
            #axs[i,j].set_xticklabels([r'$L_0$ (Nuclear Envelope)',r'$L$ (Cell Membrane)'])
            axs[i,j].set_xticklabels([r'$L_0$',r'$L$'])

            # set axis limit
            axs[i,j].set_xlim(L0,L)
            axs[i,j].ticklabel_format(axis='y',style='scientific',scilimits=(0,0))

            axs[i,j].spines['right'].set_visible(False)
            axs[i,j].spines['top'].set_visible(False)
    
    fig.subplots_adjust(top=1,right=.95,left=.12,bottom=0.05,hspace=.8,wspace=1)

    #fig.canvas.draw()
    
    ## absolute coordinate stuff.

    #x0 = (axs[3,0].get_position().x1 + axs[3,0].get_position().x0)
    x0 = axs[3,0].get_position().x1-.05
    y0 = 0.7
    w = (axs[3,1].get_position().x0 - axs[3,0].get_position().x1)+.1
    h = axs[3,0].get_position().y1 - axs[3,0].get_position().y0
    
    ax_i0 = fig.add_axes([x0,y0,w,h])
    #ax_i0.yaxis.set_major_formatter(ScalarFormatter(useMathText=True))
    
    # initial data

    ax_i0.set_title(r'Initial Data $\tilde V_0(r) = I(r,0)+M(r,0)$',size=fsizetitle)
    ax_i0.plot(p.r,p.data_avg_fns['control'](p.r),lw=2,color='k')
    #ax_i0.ticklabel_format(axis='y',style='scientific',scilimits=(0,0))

    # rescale to percent
    ticks_y = ticker.FuncFormatter(lambda x, pos: '{0:g}\%'.format(x*100))
    ax_i0.yaxis.set_major_formatter(ticks_y)

    #bbox=dict(boxstyle="round",fc='tab:blue',alpha=0.4)
    ax_i0.annotate(r'$I(r,0) = \varepsilon\tilde V_0(r)$',xy=(.25,-.6),xycoords='axes fraction',size=15,ha='right',bbox=dict(boxstyle="round",fc='tab:blue',alpha=0.4))
    ax_i0.annotate(r'',xy=(.15, -1), xycoords='axes fraction', xytext=(0.5,-.1), arrowprops=dict(arrowstyle="simple,head_width=1,head_length=1", color='tab:blue'))

    ax_i0.annotate(r'$M(r,0) = (1-\varepsilon)\tilde V_0(r)$',xy=(.75,-.6),xycoords='axes fraction',size=15,ha='left',bbox=dict(boxstyle="round",fc='tab:orange',alpha=0.4))
    ax_i0.annotate(r'',xy=(.85, -1), xycoords='axes fraction', xytext=(.5,-.1), arrowprops=dict(arrowstyle="simple,head_width=1,head_length=1", color='tab:orange'))

    ax_i0.set_xticks([L0,L])
    ax_i0.set_xticklabels([r'$L_0$ (Nuclear Envelope)',r'$L$ (Cell Edge)'])
    ax_i0.set_xlim(L0,L)
    
    # I to M and vice-versa diagram
    c1 = (axs[3,0].get_position().x0+axs[3,0].get_position().x1)/2
    c2 = (axs[3,1].get_position().x0+axs[3,1].get_position().x1)/2
    plt.text(c1, .9, "$I$: Immobile Material", ha="center", va="center", size=15,
             transform=fig.transFigure,bbox=dict(boxstyle="round",fc='tab:blue',alpha=0.4))

    plt.text(c2, .9, "$M$: Material Subject to\n Retrograde Flow", ha="center", va="center", size=15,transform=fig.transFigure,bbox=dict(boxstyle="round",fc='tab:orange',alpha=0.4))

    width = .005
    head_width = 0.02
    head_length = 0.05
    
    b1 = axs[3,0].get_position().x1
    b2 = axs[3,1].get_position().x0
    a2 = plt.arrow(b1+.033,.905,.2,0,width=width,ec='k',fc='k',transform=fig.transFigure,
                   head_width=head_width,head_length=head_length,length_includes_head=True,**{'shape':'right'})
    a2.set_clip_on(False)
    
    a3 = plt.arrow(b2-.043,.895,-.2,0,width=width,ec='k',fc='k',transform=fig.transFigure,
                   head_width=head_width,head_length=head_length,length_includes_head=True,**{'shape':'right'})
    a3.set_clip_on(False)

    x = (axs[3,1].get_position().x0 + axs[3,0].get_position().x1)/2
    
    plt.text(x,.87,r'$\alpha$ (Trap)',size=fsizelabel,ha='center',transform=fig.transFigure)
    plt.text(x,.92,r'$\beta$ (Release)',size=fsizelabel,ha='center',transform=fig.transFigure)

    # arrow from M to I in plots, vice-versa.
    x = axs[3,0].get_position().x1
    y = (axs[3,0].get_position().y0+axs[3,1].get_position().y1)/2
    dx = axs[3,1].get_position().x0 - axs[3,0].get_position().x1
    
    a1 = plt.arrow(x+.02,y+.005,dx*.6,0,width=width,ec='k',fc='k',
                   head_width=head_width,head_length=head_length,transform=fig.transFigure,
                   length_includes_head=True,
                   **{'shape':'right'})
    a1.set_clip_on(False)
    
    x = axs[3,1].get_position().x0
    dx = axs[3,1].get_position().x0 - axs[3,0].get_position().x1

    a1 = plt.arrow(x-.09,y-.005,-dx*.6,0,width=width,ec='k',fc='k',
                   head_width=head_width,head_length=head_length,transform=fig.transFigure,
                   length_includes_head=True,
                   **{'shape':'right'})
    a1.set_clip_on(False)

    x = (axs[3,1].get_position().x0 + axs[3,0].get_position().x1)/2

    plt.text(x-.03,y-.03,r'$\alpha$ (Trap)',size=fsizelabel,ha='center',transform=fig.transFigure)
    plt.text(x-.03,y+.02,r'$\beta$ (Release)',size=fsizelabel,ha='center',transform=fig.transFigure)
    

    #plt.text(x-.03,y+.05,r'(Trapping)',size=fsizelabel,ha='center',transform=fig.transFigure)
    

    # advection label
    arrow_str = 'simple, head_width=.7, tail_width=.15, head_length=1'
    axs[3,1].annotate('',(.25,.4),(.45,.4),ha='left',
                      va='center',
                      xycoords='axes fraction',
                      arrowprops=dict(arrowstyle=arrow_str,
                                      fc='k',
                                      ec='k'))
    
    advection_label = (r'Advection:\begin{itemize}[noitemsep,topsep=0pt,labelsep=.1em,label={}]'
                       r'\setlength{\itemindent}{-1.5em}'
                       r'\item $u$ constant'
                       r'\item $u(r)$ space-dependent'
                       r'\item $u(V)$ qty.-dependent'
                       r'\end{itemize}')
    
    axs[3,1].annotate(advection_label,(.5,.5),(.5,.5),xycoords='axes fraction',va='center')

    curly_str = (r'$'
                 r'\begin{cases}'
                 r'\phantom{1}\\'
                 r'\phantom{1}\\'
                 r'\phantom{1}\\'
                 r'\end{cases}'
                 r'$')

    axs[3,1].annotate(curly_str,(.45,.4),(.45,.4),xycoords='axes fraction',size=10,va='center')
    
    
    # time arrow
    plt.annotate(r'',xy=(.05,.05), xycoords='figure fraction', xytext=(.05, .56), arrowprops=dict(arrowstyle="simple,head_width=1,head_length=1", color='k'))
    
    plt.text(.034,.25,'Time',ha='center',transform=fig.transFigure,size=20,rotation=90)


    # update scientific notation
    fig.canvas.draw()

    offset = ax_i0.get_yaxis().get_offset_text().get_text()
    #offset = ax_i0.yaxis.get_major_formatter().get_offset()
    
    
    print(offset)
    
    ax_i0.yaxis.offsetText.set_visible(False)
    #ax_i0.yaxis.set_label_text(r"$\tilde V_0$" + " (" + offset+")",size=fsizelabel)
    ax_i0.yaxis.set_label_text(r"$\tilde V_0$")
    ax_i0.tick_params(axis='both',labelsize=fsizetick)

    
    for i in range(2,nrows):
        for j in range(ncols):
            # remove y axis label

            offset = axs[i,j].get_yaxis().get_offset_text().get_text()

            axs[i,j].yaxis.offsetText.set_visible(False)
            ylabel = axs[i,j].yaxis.get_label().get_text()
            #print(ylabel.__dict__)

            
            # rescale to percent
            ticks_y = ticker.FuncFormatter(lambda x, pos: '{0:g}\%'.format(x*100))
            axs[i,j].yaxis.set_major_formatter(ticks_y)

            #axs[i,j].yaxis.set_label_text(ylabel + " (" + offset+")")
            axs[i,j].yaxis.set_label_text(ylabel)
            axs[i,j].yaxis.labelpad=-10

            axs[i,j].tick_params(axis='both',labelsize=fsizetick)
            #axs[i,j].tick_params(axis='y',which='both',left=False,labelleft=False)



    return fig

def velocity():

    import matplotlib.pyplot as plt
    from matplotlib.gridspec import GridSpec

    fig = plt.figure(figsize=(5,3.5))
    gs = GridSpec(2, 2,hspace=0.6,wspace=.8)
    #gs = GridSpec(2, 2)
    #gs.update()

    ax1 = fig.add_subplot(gs[0,0])
    ax2 = fig.add_subplot(gs[0,1])
    ax3 = fig.add_subplot(gs[1,0])
    ax4 = fig.add_subplot(gs[1,1])

    axs = [ax1,ax2,ax3,ax4]

    nrows=2;ncols=2
    #fig,axs = plt.subplots(nrows=nrows,ncols=ncols,figsize=(6,4))

    pars = {'eps':1,'df':0,'dp':.01,'T':60,
            'dt':.01,'N':100,'us0':0.2,'model':'t1d'}

    p = pde.PDEModel(**pars)
    p._run_euler()
    V = p.y[:p.N,-1] + p.y[p.N:,-1]
    imax = np.amax(V)

    print(imax)

    p.eps = 0

    axs[0].plot([p.L0,p.L],[0.16,0.16],lw=2,color='k')
    axs[1].plot(p.r,V,lw=2,color='k')
    axs[2].plot(p.r,p._vel_spatial(),lw=2,color='k')
    axs[3].plot(p.r,p.us0*(1-V/imax),lw=2,color='k')

    for i in range(4):
        axs[i].set_xticks([p.L0,p.L])
        axs[i].set_xticklabels([r'$L_0$',r'$L$'])
        axs[i].tick_params(axis='both',labelsize=fsizetick)

    axs[0].set_ylabel(r'Velocity $\bar{u}$',size=fsizelabel)
    axs[1].set_ylabel(r'Intensity $V(r,t)$',size=fsizelabel)
    axs[2].set_ylabel(r'Velocity $u(r)$',size=fsizelabel)
    axs[3].set_ylabel(r'Velocity $u(V(r,t))$',size=fsizelabel)
    
    axs[0].set_title(r'(a) Constant',loc='left',size=fsizetitle)
    axs[1].set_title(r'(b)',loc='left',size=fsizetitle)
    axs[2].set_title(r'(c) Space-Dep.',loc='left',size=fsizetitle)
    axs[3].set_title(r'Qty. Dep.',loc='center',size=fsizetitle)


    fig.subplots_adjust(left=.16,right=.95,top=.93)

    # arrow betwen right plots
    x_start = (axs[1].get_position().x0+axs[1].get_position().x1)/2
    y_start = (axs[1].get_position().y0)-.01
    
    plt.annotate(r'',xytext=(x_start, y_start), xy=(x_start,y_start-.12),
                 ha='center', xycoords='figure fraction',arrowprops=dict(arrowstyle="simple,tail_width=1.3,head_width=2.3,head_length=1", color='k'))

    
    return fig    
    

def best_10_seeds(model):
    """
    found using google docs sheet
    """
    
    if model == 't1a':
        return [79,34,71,9,83,94,36,24,53,40,39]
    elif model == 't1b':
        return [69,34,84,11,8,86,43,45,12,74]
    elif model == 't1c':
        return [21,32,58,74,39,42,62,46,50,56]

    elif model == 't1d':
        return [65,58,28,22,10,27,76,33,20,80]
    elif model == 't1e':
        return [46,51,62,5,24,25,54,60,64,86]
    elif model == 't1f':
        return [87,98,67,47,8,30,25,29,10,53]
    
    elif model == 't2a':
        return [97,92,82,16,32,84,65,39,63,4]
    elif model == 't2b':
        return [95,89,64,41,38,49,90,84,35,23]
    elif model == 't2c':
        return [93,75,48,84,25,61,38,0,1,4]
    
    elif model == 't2d':
        return [93,89,59,22,81,73,84,38,95,32]
    elif model == 'jamminga':
        return [96,26,27,70,87,47,22,17,69,81]
    elif model == 'jammingb':
        return [56,71,26,84,64,30,34,8,41,21]
    
    elif model == 'jammingc':
        return [60,18,58,8,3,68,93,85,12,26]
    elif model == 'jammingd':
        return [22,69,29,45,11,35,25,10,96,1]
    
    else:
        raise ValueError('Invalid model',model)

def solution(model='t1e',rep=False,method=''):
    """
    plot simulation data
    including intermediate comparisons
    """

    import matplotlib.pyplot as plt
    import matplotlib.ticker as ticker

    # best 10 seeds in order from best to worst
    best_seeds = best_10_seeds(model)
    seed = best_seeds[0]
    
    pars = lib.load_pars(model,seed,method)
    print(model,seed,pars)
    
    p = pde.PDEModel(**pars)    
    p._run_euler(rep)

    I = p.y[:p.N,:];M = p.y[p.N:,:];V = I + M
    
    if model == 't1e':
        dp_lo = 0.0093765
        pars['dp'] = dp_lo
        p_lo = pde.PDEModel(**pars)    
        p_lo._run_euler(rep)

        dp_hi = 0.011617
        pars['dp'] = dp_hi
        p_hi = pde.PDEModel(**pars)    
        p_hi._run_euler(rep)
                
    if model == 't1f':
        u_lo = 0.1
        pars['us0'] = u_lo
        p_lo = pde.PDEModel(**pars)    
        p_lo._run_euler(rep)

        u_hi = 0.12
        pars['us0'] = u_hi
        p_hi = pde.PDEModel(**pars)    
        p_hi._run_euler(rep)    

    I_lo = p_lo.y[:p_lo.N,:];M_lo = p_lo.y[p_lo.N:,:];V_lo = I_lo + M_lo
    I_hi = p_hi.y[:p_hi.N,:];M_hi = p_hi.y[p_hi.N:,:];V_hi = I_hi + M_hi

    nrows = 3
    ncols = 2

    if rep:
        ncols = 6
        keys_list = ['control','1h','2h', '4h','8.5h', '24h']
    else:
        ncols = 7
        #keys_list = ['control','2h', '4h','8.5h', '24h']
        #keys_list = ['control','0.5h','1h', '2h', '24h']
        keys_list = ['control','0.5h','1h', '2h','4h','8.5h','24h']
        
    fig,axs = plt.subplots(nrows=3,ncols=ncols,figsize=(4,4),
                           gridspec_kw={'wspace':0.1,'hspace':0},
                           sharey='all')
    
    # plot best solution
    for i,hour in enumerate(keys_list):

        if rep:
            data = p.data_rep_fns[hour](p.r)
        else:
            data = p.data_avg_fns[hour](p.r)
        
        if hour == 'control':
            hour = '0h'
            idx = 0
        else:
            time = float(hour[:-1])
            minute = time*60
            idx = int(minute/p.dt)
        
        axs[0,i].plot(p.r[:-1],V[:-1,idx],color='k',lw=.7)
        axs[0,i].plot(p.r[1:-1],data[1:-1],label='Data',c='tab:green',dashes=(3,1))
        axs[1,i].plot(p.r[:-1],I[:-1,idx],color='tab:blue',lw=.7)
        axs[2,i].plot(p.r[:-1],M[:-1,idx],color='tab:orange',lw=.7)

        # plot fill_between:
        if model == 't1e' or model == 't1f':
            axs[0,i].fill_between(p.r[:-1],V_lo[:-1,idx],V_hi[:-1,idx],color='gray',alpha=.25)
            axs[1,i].fill_between(p.r[:-1],I_lo[:-1,idx],I_hi[:-1,idx],color='tab:blue',alpha=.25)
            axs[2,i].fill_between(p.r[:-1],M_lo[:-1,idx],M_hi[:-1,idx],color='tab:orange',alpha=.25)

        if hour == '24h':
            print(np.linalg.norm(M[:-1,int(20*60/p.dt)]-M[:-1,int(24*60/p.dt)])**2)

        axs[0,i].set_title(r'\SI{'+hour[:-1]+r'}{h}',size=fsizetitle,y=0.95)
        axs[-1,i].set_xticks([p.L0,p.L])

        #axs[-1,i].set_xticklabels([r'$L_0$\hspace{.1in}',r'\hspace{.1in}$L$'],ha='center')
        axs[-1,i].set_xticklabels([r'$L_0$',r'$L$'],ha='center')
        
        axs[-1,i].get_xaxis().majorTicks[0].label1.set_horizontalalignment('left')
        axs[-1,i].get_xaxis().majorTicks[1].label1.set_horizontalalignment('right')
        
        axs[0,i].set_xticks([])
        axs[0,i].set_xticklabels([])

        axs[1,i].set_xticks([])
        axs[1,i].set_xticklabels([])
        
        axs[0,i].tick_params(axis='both',labelsize=fsizetick)
        axs[1,i].tick_params(axis='both',labelsize=fsizetick)
        axs[2,i].tick_params(axis='both',labelsize=fsizetick)

    fn_labels = [r'$V$',r'$I$',r'$M$']
    for i in range(3):
        # go blow to change label size
        #axs[i,0].set_ylabel(fn_labels[i],size=fsizelabel,labelpad=0)
        #axs[i,0].ticklabel_format(axis='y',style='scientific',scilimits=(0,0))
        
        # rescale to percent
        ticks_y = ticker.FuncFormatter(lambda x, pos: '{0:g}\%'.format(x*100))
        axs[i,0].yaxis.set_major_formatter(ticks_y)
        axs[i,0].tick_params(axis='y',labelsize=fsizetick-2)

    if not(rep):
        # plot remaining seeds
        for seed_idx in best_seeds:
            if seed_idx not in [seed]:
                
                pars = lib.load_pars(model,seed_idx,method)
                print(pars)
                pars['dt'] = 0.055
                #pars['N'] = 100

                #fn = lib.get_parameter_fname(model,seed_idx,err=True,method='de')
                #print(seed_idx,pars,np.loadtxt(fn))
                
                p2 = pde.PDEModel(**pars)
                p2._run_euler()

                I = p2.y[:p2.N,:]
                M = p2.y[p2.N:,:]

                V = I + M

                if False:
                    
                    fig2 = plt.figure()
                    axs2 = fig2.add_subplot()

                    for i,hour in enumerate(keys_list):
                        if hour == 'control':
                            hour='0h';idx=0
                        else:
                            time=float(hour[:-1]);minute=time*60;idx=int(minute/p2.dt)

                        #print('time',time,minute)
                        axs2.plot(p2.r[:-1],V[:-1,idx],lw=.7)
                    plt.show()

                for i,hour in enumerate(keys_list):

                    if hour == 'control':
                        hour='0h';idx=0
                    else:
                        time=float(hour[:-1]);minute=time*60;idx=int(minute/p2.dt)
                    
                    axs[0,i].plot(p2.r[:-1],V[:-1,idx],color='gray',zorder=-3,alpha=0.25,lw=.5)
                    #axs[1,i].plot(p2.r[:-1],I[:-1,idx],color='gray',zorder=-3,alpha=0.25,lw=.5)
                    #axs[2,i].plot(p2.r[:-1],M[:-1,idx],color='gray',zorder=-3,alpha=0.25,lw=.5)

    if (model == 't1e' or model == 't1f') and not(rep):
        fig.subplots_adjust(top=.95,right=.95,left=.18,bottom=0.4,hspace=.8,wspace=1)
        
    else:
        fig.set_size_inches(4,3)
        fig.subplots_adjust(top=.9,right=.95,left=.1,bottom=0.1,hspace=.8,wspace=1)

    # move scientific notation to y axis label
    fig.canvas.draw()

    for i in range(3):
        pass
        #offset = axs[i,0].get_yaxis().get_offset_text().get_text()
     
        #axs[i,0].yaxis.offsetText.set_visible(False)
        #axs[i,0].yaxis.set_label_text(fn_labels[i] + " (" + offset+")",size=fsizelabel-2)
        axs[i,0].yaxis.set_label_text(fn_labels[i],size=fsizelabel-2)

    if (model == 't1e' or model == 't1f') and not(rep):
        # subplot for velocity profile
        x0 = (axs[2,1].get_position().x1 + axs[2,1].get_position().x0)/2
        y0 = .07
        w = (axs[2,5].get_position().x1 + axs[2,5].get_position().x0)/2-x0
        h = .2

        ax_u = fig.add_axes([x0,y0,w,h])

        if model == 't1e':
            ax_u.plot(p.r,p._vel_spatial(),lw=.7,color='k')
            ax_u.fill_between(p.r,p_lo._vel_spatial(),p_hi._vel_spatial(),lw=1,color='k',alpha=.25)
            
            # mark max
            max_idx = np.argmax(p._vel_spatial())
            max_val = np.amax(p._vel_spatial())

            ax_u.scatter(p.r[max_idx],max_val,s=40,color='tab:red',zorder=10)
            xticklabels = [r'$L_0$',r'$L^*$',r'$L$'];xticks = [p.L0,p.r[max_idx],p.L]
            ylabel = r'$u(r)$'
            
        if model == 't1f':
            ax_u.plot(p.r,p._dp_spatial(),lw=.7,color='k')
            ax_u.fill_between(p.r,p_lo._dp_spatial(),p_hi._dp_spatial(),lw=1,color='k',alpha=.25)
            
            xticklabels = [r'$L_0$',r'$L$'];xticks = [p.L0,p.L]
            ylabel = r'$d_p(r)$'
            print('eps, us0',p.eps,p.us0)
            
        ax_u.set_xticklabels(xticklabels)
        ax_u.set_ylabel(ylabel,size=fsizelabel)
        ax_u.set_xticks(xticks)
        #ax_u.get_xaxis().majorTicks[0].label1.set_horizontalalignment('left')
        
        ax_u.tick_params(axis='both',labelsize=fsizetick)
        #ax_u.set_ylim(0,.1)

        plt.text(.04,.96,"(a)",transform=fig.transFigure,size=fsizetitle)
        plt.text(.04,.3,"(b)",transform=fig.transFigure,size=fsizetitle)

    #plt.show()
    return fig


def cost_function_t1f(recompute=False,model='t1f'):

    from matplotlib.patches import Rectangle
    #print()
    #print(eps1[0],eps1[-1],dps1[0],dps1[-1])
    #eps_vals2,dp_vals2,Z2 = load_rss_data(recompute=recompute,exp=True,ne=50,nd=50,maxe=.1,mind=-2.5,maxd=-1.,ss=False)

    # get cost function without steady-state assumption
    eps2,dps2,Z_no_ss = sv.load_rss_data(recompute=recompute,ne=50,n2=50,maxe=1,
                                         min2=0.01,max2=0.2,ss=False,model=model)

    
    # get cost function with ss assumption. use this to partition figure
    eps3,dps3,Z3 = sv.load_rss_data(recompute=recompute,ne=50,n2=50,mine=0,maxe=1,
                                    min2=0.01,max2=.2,ss=True,model=model)
    
    
    

    
    
    # limit Z data in eps
    p2_mask = dps2 > 0.025
    eps_mask = eps2 < .8
    p2 = dps2[p2_mask]
    #print(len(p2_mask))
    
    Z_no_ss = Z_no_ss[p2_mask,:]
    Z_no_ss = Z_no_ss[:,eps_mask]
    eps2 = eps2[eps_mask]
    #Z_no_ss = Z_no_ss[:,eps_mask]

    Z3 = Z3[p2_mask,:]
    Z3 = Z3[:,eps_mask]

    boundary_idxs = np.where(np.abs(np.diff(Z3,axis=0))>1e4)
    print(boundary_idxs)
    
    Z_fail_ss = np.zeros_like(Z_no_ss)+np.nan
    Z_u_undef = np.zeros_like(Z_no_ss)+np.nan    
    
    # force 0 values to be nan (cost function returns 0 if solutions return nan)
    #nan_idxs = np.where(Z_no_ss==0)
    #Z_no_ss[nan_idxs] = np.nan; Z_fail_ss[nan_idxs] = np.nan
    Z_no_ss[np.where(Z_no_ss==0)] = np.nan
    Z_no_ss = 10**(Z_no_ss)
    Z_no_ss[np.where(Z_no_ss==np.inf)] = np.nan
    
    nan2_idxs = np.where(np.isnan(Z_no_ss))
    Z_u_undef[np.where(np.isnan(Z_no_ss))] = 0

    
    bdy_idx_x = []
    bdy_idx_y = []

    # force Z data to be above boundary_idxs
    """
    for n in range(len(boundary_idxs[0])):
        i = boundary_idxs[0][n]; j = boundary_idxs[1][n]

        Z_fail_ss[:i,j] = Z_no_ss[:i,j]
        Z_fail_ss[i:,j] = np.nan
        Z_no_ss[:i,j] = np.nan

        #print(i,j)
        # construct boundary line
        bdy_idx_x.append(j)
        bdy_idx_y.append(i)
    """
    
    import matplotlib.pyplot as plt
    
    fig, axs = plt.subplots(nrows=2,ncols=1,figsize=(5,7))

    maxval = np.nanmax([np.nanmax(Z_no_ss),np.nanmax(Z_fail_ss)])
    minval = np.nanmin([np.nanmin(Z_no_ss),np.nanmin(Z_fail_ss)])
    print(np.nanmax(Z_no_ss),)

    #cax1 = axs.pcolormesh(eps2,dps2,Z_u_undef,color='k')
    cax2 = axs[0].pcolormesh(eps2,p2,Z_no_ss,vmin=minval,vmax=maxval)
    #cax3 = axs[0].pcolormesh(eps2,p2,Z_fail_ss,vmin=minval,vmax=maxval,alpha=.9)
    #cax3 = axs.pcolormesh(eps3,dps3,np.diff(Z3,axis=0))
    
    c2 = axs[0].contour(eps2,p2,Z_no_ss,colors='white')

    #sorted_idx = np.argsort(bdy_idx_x)
    #print(sorted_idx,np.array(bdy_idx_x)[sorted_idx])
    #x = eps2[np.array(bdy_idx_x)[sorted_idx]]
    #y = dps2[np.array(bdy_idx_y)[sorted_idx]+4]
    
    #x = np.append(x,eps2[np.array(bdy_idx_x)[sorted_idx][-1]+1])
    #y = np.append(y,dps2[np.array(bdy_idx_y)[sorted_idx][-1]+1])
    
    #axs[0].fill_between(x,np.zeros(len(x))+0.025,y,alpha=.5,color='tab:red',hatch='x')

    # show minimum
    axs[0].scatter(0.55,0.11,color='tab:orange',marker='x',s=100,clip_on=False,lw=4,zorder=10)

    # label failure regions
    #axs[0].text(0.9,0.5,r'$\exists r$ s.t. $d_p(r)<0$',rotation=90,transform=axs[0].transAxes,size=fsizelabel,
    #         ha='center',va='center')
    
    props = dict(boxstyle='round', facecolor='white', alpha=0.8)
    #axs[0].text(0.4, 0.1, 'Steady-State Cond. Fails', transform=axs[0].transAxes, fontsize=12,
    #         ha='center', bbox=props)
    
    cbar = fig.colorbar(cax2,ax=axs[0]); cbar.ax.tick_params(labelsize=fsizetick)
    cbar.formatter.set_powerlimits((0,0))
    #cbar = fig.colorbar(cax3)

    #axs[0].set_yscale('log')
    axs[0].set_ylim(0.025,.2)


    # confidence interval
    eps1,dps1,Z1 = sv.load_rss_data(recompute=recompute,ss=False,
                                    ne=50,n2=50,min2=.05,max2=.15,mine=0.4,maxe=0.65)
    
    eps_lo = .55-.1
    eps_hi = .55+.1
    dp_lo = .08
    dp_hi = .14

    axs[0].add_patch(Rectangle((eps_lo,dp_lo),
                               eps_hi-eps_lo,
                               dp_hi-dp_lo,
                               fc='none',ec='tab:red',zorder=1))

    Z = 10**(Z1)
    n = 1001
    rss0 = np.amin(Z)
    ln_th0 = np.log(Z/n)
    ln_th_ = np.log(rss0/n)

    diff = ln_th0-ln_th_
    
    idxs_dp = (dps1<dp_hi)&(dps1>dp_lo)
    idxs_ep = (eps1<eps_hi)&(eps_lo<eps1)
    diff1 = diff[idxs_dp,:]
    diff2 = diff1[:,idxs_ep]

    cax = axs[1].pcolormesh(eps1[idxs_ep],dps1[idxs_dp],diff2)
    #axs[1].contour(eps1,dps1,diff,colors='white',levels=[5.991/n])
    
    axs[1].contour(eps1,dps1,diff,colors='white',levels=[7.815/n])
    cbar2 = fig.colorbar(cax,ax=axs[1]); cbar2.ax.tick_params(labelsize=fsizetick)

    axs[0].set_xlabel(r'$\varepsilon$',size=fsizelabel)
    axs[1].set_xlabel(r'$\varepsilon$',size=fsizelabel)

    axs[0].set_ylabel(r'$\bar{u}$',size=fsizelabel)
    axs[1].set_ylabel(r'$\bar{u}$',size=fsizelabel)

    axs[0].tick_params(axis='both',labelsize=fsizetick)
    axs[1].tick_params(axis='both',labelsize=fsizetick)

    axs[1].set_xlim(eps_lo,eps_hi)
    axs[1].set_ylim(dp_lo,dp_hi-.00004)    
    
    axs[0].set_title('(a) RSS',loc='left',size=fsizetitle)
    axs[1].set_title('(b) Confidence Interval',loc='left',size=fsizetitle)

    
    plt.tight_layout()
    #fig.subplots_adjust(bottom=0.1)
    return fig

    #plt.show()

def cost_function_t1e(recompute=False):

    #print()
    #print(eps1[0],eps1[-1],dps1[0],dps1[-1])
    #eps_vals2,dp_vals2,Z2 = load_rss_data(recompute=recompute,exp=True,ne=50,nd=50,maxe=.1,mind=-2.5,maxd=-1.,ss=False)

    # get cost function without steady-state assumption
    eps2,dps2,Z_no_ss = sv.load_rss_data_t1e(recompute=recompute,exp=True,
                                             ne=50,nd=50,maxe=0.4,mind=-2.5,maxd=-1.,ss=False)

    # get cost function with ss assumption. use this to partition figure
    eps3,dps3,Z3 = sv.load_rss_data_t1e(recompute=recompute,exp=True,
                                    ne=50,nd=50,maxe=0.4,mind=-2.5,maxd=-1.,ss=True)
    
    boundary_idxs = np.where(np.abs(np.diff(Z3,axis=0))>1e4)
    
    # limit Z data in eps
    eps_mask = eps2 < 0.35
    eps2 = eps2[eps_mask]
    Z_no_ss = Z_no_ss[:,eps_mask]
    
    Z_fail_ss = np.zeros_like(Z_no_ss)
    Z_u_undef = np.zeros_like(Z_no_ss)+np.nan

    # force 0 values to be nan (cost function returns 0 if solutions return nan)
    nan_idxs = np.where(Z_no_ss==0)
    Z_no_ss[nan_idxs] = np.nan; Z_fail_ss[nan_idxs] = np.nan
    Z_no_ss = 10**(Z_no_ss)
    
    nan2_idxs = np.where(np.isnan(Z_no_ss))
    Z_u_undef[np.where(np.isnan(Z_no_ss))] = 0
    
    bdy_idx_x = []
    bdy_idx_y = []

    # force Z data to be above boundary_idxs
    for n in range(len(boundary_idxs[0])):
        i = boundary_idxs[0][n]; j = boundary_idxs[1][n]

        Z_fail_ss[:i,j] = Z_no_ss[:i,j]
        Z_fail_ss[i:,j] = np.nan
        Z_no_ss[:i,j] = np.nan

        #print(i,j)
        # construct boundary line
        bdy_idx_x.append(j)
        bdy_idx_y.append(i)

    import matplotlib.pyplot as plt
    
    fig, axs = plt.subplots(nrows=2,ncols=1,figsize=(5,7))
    
    axs[0].set_yscale('log')
    #axs[1].set_yscale('log')
    totval = np.nansum(Z_no_ss+Z_fail_ss)
    
    maxval = np.amax([np.nanmax(Z_no_ss),np.nanmax(Z_fail_ss)])
    minval = np.amin([np.nanmin(Z_no_ss),np.nanmin(Z_fail_ss)])

    #cax1 = axs.pcolormesh(eps2,dps2,Z_u_undef,color='k')
    cax2 = axs[0].pcolormesh(eps2,dps2,Z_no_ss,vmin=minval,vmax=maxval)
    cax3 = axs[0].pcolormesh(eps2,dps2,Z_fail_ss,vmin=minval,vmax=maxval,alpha=.9)
    #cax3 = axs.pcolormesh(eps3,dps3,np.diff(Z3,axis=0))
    
    c2 = axs[0].contour(eps2,dps2,Z_no_ss,colors='white')
    c3 = axs[0].contour(eps2,dps2,Z_fail_ss,colors='white',levels=c2.levels,alpha=.5)

    sorted_idx = np.argsort(bdy_idx_x)
    #print(sorted_idx,np.array(bdy_idx_x)[sorted_idx])
    x = eps2[np.array(bdy_idx_x)[sorted_idx]]
    y = dps2[np.array(bdy_idx_y)[sorted_idx]]
    
    x = np.append(x,eps2[np.array(bdy_idx_x)[sorted_idx][-1]+1])
    y = np.append(y,dps2[np.array(bdy_idx_y)[sorted_idx][-1]+1])
    #axs.plot(x,y,color='tab:red',lw=2)
    axs[0].fill_between(x,np.zeros(len(x)),y,alpha=.3,color='tab:red',hatch='x')

    # show minimum
    axs[0].scatter(0,0.011,color='tab:orange',marker='x',s=100,clip_on=False,lw=4,zorder=10)

    # label failure regions
    axs[0].text(0.9,0.5,r'$u(r)$ Undefined',rotation=90,transform=axs[0].transAxes,size=fsizelabel,
             ha='center',va='center')
    
    props = dict(boxstyle='round', facecolor='white', alpha=0.8)
    axs[0].text(0.4, 0.1, 'Steady-State Cond. Fails', transform=axs[0].transAxes, fontsize=14,
             ha='center', bbox=props)

    
    cbar = fig.colorbar(cax2,ax=axs[0]); cbar.ax.tick_params(labelsize=fsizetick)
    cbar.formatter.set_powerlimits((0,0))
    #cbar = fig.colorbar(cax3)


    # confidence interval
    eps1,dps1,Z1 = sv.load_rss_data_t1e(recompute=recompute,ss=False,mind=8e-3,maxd=1.3e-2,maxe=0.02)
    
    eps_lo = 0
    eps_hi = .01
    dp_lo = .0085
    dp_hi = .013

    axs[0].add_patch(Rectangle((eps_lo,dp_lo),
                               eps_hi-eps_lo,
                               dp_hi-dp_lo,
                               fc='none',ec='tab:red',zorder=1))

    Z = 10**(Z1)
    n = 1001
    rss0 = np.amin(Z)
    ln_th0 = np.log(Z/n)
    ln_th_ = np.log(rss0/n)

    diff = ln_th0-ln_th_

    idxs_dp = (dps1<dp_hi)&(dps1>dp_lo)
    idxs_ep = eps1<eps_hi
    diff1 = diff[idxs_dp,:]
    diff2 = diff1[:,idxs_ep]

    cax = axs[1].pcolormesh(eps1[idxs_ep],dps1[idxs_dp],diff2)
    #axs[1].contour(eps1,dps1,diff,colors='white',levels=[5.991/n])
    axs[1].contour(eps1,dps1,diff,colors='white',levels=[7.815/n])
    cbar2 = fig.colorbar(cax,ax=axs[1]); cbar2.ax.tick_params(labelsize=fsizetick)
    
    #axs[1].set_yscale('log')
    axs[0].set_xlabel(r'$\varepsilon$',size=fsizelabel)
    axs[1].set_xlabel(r'$\varepsilon$',size=fsizelabel)
    
    axs[0].set_ylabel(r'$d_p$',size=fsizelabel)
    axs[1].set_ylabel(r'$d_p$',size=fsizelabel)
    

    axs[0].tick_params(axis='both',labelsize=fsizetick)
    axs[1].tick_params(axis='both',labelsize=fsizetick)

    axs[0].set_xlim(0,0.35)
    
    axs[1].set_xlim(eps_lo,eps_hi)
    axs[1].set_ylim(dp_lo,dp_hi-.00004)

    axs[0].set_title('(a) RSS',loc='left',size=fsizetitle)
    axs[1].set_title('(b) Confidence Interval',loc='left',size=fsizetitle)
    
    plt.tight_layout()
    #fig.subplots_adjust(bottom=0.1)
    
    return fig
    #plt.show()
    
def rss(p,V):
    """
    return RSS of only the initial function scaled by \ve
    """
    err = 0
    for hour in p.data_avg.keys():

        # convert hour to index
        if hour == 'control':
            pass # ignore initial condition (trivial)
        else:
            # restrict solution to observed positions
            V_fn = interp1d(p.r,V)
            V_cut = V_fn(p.data_avg[hour][:,0])

            data = p.data_avg[hour][:,1]
            err += np.linalg.norm(data[1:-1]-V_cut[1:-1])**2
            
    return err

def get_scatter_params(errs,err):
    """
    return scatter plot parameters given error difference
    """

    
    from matplotlib import cm
    viridis = cm.get_cmap('viridis',12)
    
    err_min=np.amin(errs);err_max=np.amax(errs)        
    errs_range = np.linspace(err_min,err_max,100)

    cmap_val = np.linspace(0,1,100)

    # get min err range
    idx2 = np.argmin(np.abs(err-errs_range))
    color_val = cmap_val[idx2]
    s = 5
    zorder = 0

    #if np.abs(err - err_max) > np.abs(err - err_min):
    #    zorder = 10
    #    s = 10
    #else:
    #    zorder = 2
    #    s = 10
                    
    color = viridis(color_val*.9)
            
    """
    #print(color_val)
    if (err_max - err_min) < 1e-2:
        color = viridis(.0)
        zorder = 2
        s = 10
                
    else:
        # get min err range
        idx2 = np.argmin(np.abs(err-errs_range))
        color_val = cmap_val[idx2]

        if np.abs(err - err_max) > np.abs(err - err_min):
            zorder = 10
            s = 10
        else:
            zorder = 2
            s = 10
                    
        color = viridis(color_val*.8)
    """
    return color, s, zorder
    
def get_errs(model):
    """
    collect all rss given model
    """
    errs = []
    
    max_seed = 100
    if model == 't1e':
        max_seed = 10        
    for seed in range(max_seed):
        fname_err = lib.get_parameter_fname(model,seed,err=True,method='de')
        
        err = np.loadtxt(fname_err)
        errs.append(err)

    return errs

def fmt(x, pos):
    a, b = '{:.2e}'.format(x).split('e')
    b = int(b)
    return r'${} \times 10^{{{}}}$'.format(a, b)


def plot_axs_split(fig,axs,axs_split):
    """
    axs: list of gridspec 
    """

    import matplotlib
    
    models = ['t1d','t2d','jammingd']
    ylabels = [(r'$d_p^*$',r'$\bar{u}^*$'),(r'$d_{p_2}^*/d_{p_1}^*$',r'$\bar{u}^*$'),
               (r'$d_p^*$',r'$V_\text{max}^*/u_\text{max}^*$')]
    ylos = [(-2,-.4),(-.1,-.4),(-2,-.1)]
    yhis = [(22,4.4),(1.1,4.4),(22,1.1)]
    
    for i in range(3):
        axs_split[i].set_zorder(2)
        axs_split[i].set_xlim(-.1,1.1);axs_split[i+3].set_xlim(-.1,1.1)
        
        axs_split[i].set_xticklabels([])
        axs_split[i].tick_params(axis='x',direction='in')

    for i,model in enumerate(models):
        
        # plot all
        xlist = []
        ylist1 = []
        ylist2 = []
        
        errs = np.array(get_errs(model))
        err_min = np.amin(errs);err_max=np.amax(errs)        
        errs_range = np.linspace(err_min,err_max,100)
        seeds_sorted = np.argsort(errs)[::-1]

        for seed in seeds_sorted:
            fname = lib.get_parameter_fname(model,seed,method='de')
            pars = np.loadtxt(fname)
            color, s, zorder = get_scatter_params(errs,errs[seed])

            xlist.append(pars[0])
            
            if i == 0:
                ylist1.append(pars[1])
                ylist2.append(pars[2])

            if i == 1:
                ylist1.append(pars[2]/pars[1])
                ylist2.append(pars[3])

            if i == 2:
                ylist1.append(pars[3])
                ylist2.append(pars[1]/pars[2])

        cbformat = matplotlib.ticker.ScalarFormatter()
        cbformat.set_powerlimits((0,0))

        err_exp = np.round(np.exp(errs[seeds_sorted]),4)
        im1 = axs_split[i].scatter(xlist,ylist1,c=err_exp,s=s,zorder=zorder,alpha=.8)
        im2 = axs_split[i+3].scatter(xlist,ylist2,c=err_exp,s=s,zorder=zorder,alpha=.8)

        #,shrink=1,aspect=1
        #cb = fig.colorbar(im,ax=axs[i],format=cbformat,shrink=.8)
        cb = fig.colorbar(im1,ax=[axs_split[i],axs_split[i+3]],format=cbformat,shrink=.8)
        cb.ax.yaxis.get_offset_text().set_horizontalalignment('left')
        cb.ax.yaxis.get_offset_text().set_position((-1,0))
        

    # set y labels. could do it above but more readable here maybe.
    for i in range(len(models)):
        #i,j = idxs_plot[idx]
        #model = idxs_model[idx]
        model = models[i]

        pars,par_names = lib.load_pars(model,0,method='de',return_names=True)

        axs_split[i].set_ylim(ylos[i][0],yhis[i][0])
        axs_split[i+3].set_ylim(ylos[i][1],yhis[i][1])
        axs_split[i].text(-.32,0.5,ylabels[i][0],va='center',rotation=90,
                    transform=axs_split[i].transAxes,size=fsizelabel)
        
        axs_split[i+3].text(-.32,0.5,ylabels[i][1],va='center',rotation=90,
                      transform=axs_split[i+3].transAxes,size=fsizelabel)

        if i+3 > 3:
            axs_split[i+3].set_xticks([0,0.5,1])
            
        else:
            axs_split[i+3].axes.get_xaxis().set_visible(False)

    axs_split[-1].set_xlabel(r'$\varepsilon^*$',size=fsizelabel)
    axs_split[-2].set_xlabel(r'$\varepsilon^*$',size=fsizelabel)

    return axs_split


def identifiability():
    """
    identifiable and non-id. models
    """

    import matplotlib
    from matplotlib.patches import Rectangle
    import matplotlib.pyplot as plt
    from matplotlib.gridspec import GridSpec
    import matplotlib.gridspec as gridspec
    import matplotlib.ticker as ticker

    from matplotlib import cm

    nrows=2*5;ncols=3
    fig = plt.figure(figsize=(10,10))
    gs = GridSpec(nrows=nrows,ncols=ncols)

    # manual loop over index... easier to do this for now.
    mechanism = ['t1','t2','jamming']
    scenario = ['a','b','c','d','e']
    models = [mechanism[j]+scenario[i] for i in range(int(nrows/2)) for j in range(ncols)]

    del models[-1];del models[-1]
    
    axs = [fig.add_subplot(gs[2*i:2*i+2,j]) for i in range(int(nrows/2)) for j in range(ncols)]
    
    # turn off corresponding axs
    axs[9].axis('off')
    axs[10].axis('off')
    axs[11].axis('off')

    gs2 = gridspec.GridSpecFromSubplotSpec(2,3,subplot_spec=gs[6:8,:],hspace=0)

    import inspect
    
    for i in range(len(models)):        
        model = models[i]

        # compile err for given model
        errs = np.array(get_errs(model))
        err_min = np.amin(errs);err_max=np.amax(errs)        
        errs_range = np.linspace(err_min,err_max,100)
        seeds_sorted = np.argsort(errs)[::-1]
                
        # plot all
        xlist = []
        ylist = []
                
        for seed in seeds_sorted:
            fname = lib.get_parameter_fname(model,seed,method='de')
            pars = np.loadtxt(fname)

            xlist.append(pars[0])
                        
            if (i >= 3 and i <= 4) or (i == 5) or (i == 8):
                #print(i,seed,pars)
                ylist.append(pars[2])
            else:
                ylist.append(pars[1])
            
        if i >= 9 and i <= 11:
            pass
        
        else:
            cbformat = matplotlib.ticker.ScalarFormatter()
            cbformat.set_powerlimits((1e-4,10))

            err_exp = np.round(np.exp(errs[seeds_sorted]),4)
            
            im = axs[i].scatter(xlist,ylist,c=err_exp,alpha=.8,s=5)
            cb = fig.colorbar(im,ax=axs[i],format=cbformat,shrink=.8)
            cb.ax.yaxis.get_offset_text().set_horizontalalignment('left')
            cb.ax.yaxis.get_offset_text().set_position((-1,0))

    for i in range(len(axs)):
        axs[i].set_xlim(-.1,1.1)
        axs[i].tick_params(axis='x',direction='in')
            
    axs[0].set_title('T1')
    axs[1].set_title('T2')
    axs[2].set_title('Jamming')

    # scenario label
    for i,s in enumerate(scenario):
        m = len(mechanism)

        if s == 'd':
            offset = -.49
        else:
            offset = -.6
        axs[i*m].text(offset,0.5,s,va='center',rotation=0,transform=axs[i*m].transAxes,
                      size=fsizelabel+2)
        axs[i*(m-1)].set_ylim(-.1,1.1)

    # x-axis label
    axs[-3].set_xlabel(r'$\varepsilon^*$',size=fsizelabel)
    axs[-4].set_xlabel(r'$\varepsilon^*$',size=fsizelabel)
    axs[-5].set_xlabel(r'$\varepsilon^*$',size=fsizelabel)

    # remove bottom left and bottom right plots
    axs[-1].axis('off')
    axs[-2].axis('off')

    # set y labels. could do it above but more readable here maybe.    
    for i in range(len(models)):
        #i,j = idxs_plot[idx]
        #model = idxs_model[idx]
        model = models[i]

        pars,par_names = lib.load_pars(model,0,method='de',return_names=True)

        if par_names[1] == 'df':
            par_name = r'$d_f^*$';ylo=-1;yhi=21
        elif par_names[1] == 'dp1':
            par_name = r'$d_{p_1}^*$';
            if model[-1] == 'a':
                ylo=-1;yhi=21
            else:
                ylo=-10;yhi=210
                
        elif par_names[1] == 'dp':
            par_name = r'$d_p^*$';ylo=-1;yhi=21
        elif par_names[1] == 'imax':
            par_name = r'$V_\text{max}^*$';ylo=-.1;yhi=1.1
        elif par_names[1] == 'us0':
            par_name = r'$\bar{u}^*$';ylo=-.4;yhi=4.4
        else:
            par_name = par_names[1]

        if i >= 3 and i <= 4:
            par_name = r'$\bar{u}^*$';ylo=-.4;yhi=4.4

        if i == 5 or i == 8:
            par_name = r'$u_\text{max}^*$'; ylo=-.4;yhi=4.4
            
        if i >= 9 and i <= 11:
            pass
        
        else:
            axs[i].set_ylim(ylo,yhi)
            axs[i].text(-.32,0.5,par_name,va='center',rotation=90,
                        transform=axs[i].transAxes,size=fsizelabel)

    fig.subplots_adjust(top=.87,right=.95,left=.15,bottom=0.05,hspace=.05,wspace=.5)

    
    # split axes scenario d
    axs_split = [fig.add_subplot(gs2[i,j]) for i in range(2) for j in range(3)]
    axs_split = plot_axs_split(fig,axs,axs_split)


    # highlight every other row
    js = [3,9]
    shift = .14
    for j in js:
        x0 = axs[j].axes.get_position().x0; y0 = axs[j].axes.get_position().y0
        x1 = axs[j+2].axes.get_position().x1; y1 = axs[j].axes.get_position().y1
        r1 = Rectangle((x0-shift,y0),(x1-x0+shift),(y1-y0),alpha=.05,color='gray',zorder=3)
        #print(x0,y0,x1,y1)
        fig.add_artist(r1)#Rectangle((x0,y0),(x1-x0),(y1-y0),zorder=10)


        
    return fig

def collect_identifiability_data(model):
    
    errs = np.array(get_errs(model))
    err_min = np.amin(errs);err_max=np.amax(errs)        
    errs_range = np.linspace(err_min,err_max,100)
    seeds_sorted = np.argsort(errs)[::-1]
    
    # plot all
    xlist = []
    ylist = []
    
    for seed in seeds_sorted:
        fname = lib.get_parameter_fname(model,seed,method='de')
        pars = np.loadtxt(fname)
        
        xlist.append(pars[0])
        
        if (model == 't1b') or (model == 'jammingb') or (model == 'jammingc'):
            #print(i,seed,pars)
            ylist.append(pars[2])
        else:
            ylist.append(pars[1])

    err_exp = np.round(np.exp(errs[seeds_sorted]),4)
    
    return err_exp,xlist,ylist


def plot_axs_split2(fig,axs_split,model):
    """
    axs: list of gridspec 
    """

    import matplotlib
    
    models = ['t1d','t2d','jammingd']
    ylabels = [(r'$d_p^*$',r'$\bar{u}^*$'),(r'$d_{p_2}^*/d_{p_1}^*$',r'$\bar{u}^*$'),
               (r'$d_p^*$',r'$V_\text{max}^*/u_\text{max}^*$')]
    ylos = [(-2,-.4),(-.1,-.4),(-2,-.1)]
    yhis = [(22,4.4),(1.1,4.4),(22,1.1)]
    
    axs_split[0].set_zorder(2)
    axs_split[0].set_xlim(-.1,1.1);axs_split[1].set_xlim(-.1,1.1)

    if model == 't1d':
        #axs_split[1].set_xticklabels([])
        #axs_split[1].tick_params(axis='x',direction='in')
        pass
        
    axs_split[0].set_xticklabels([])
    axs_split[0].tick_params(axis='x',direction='in')
        
    # plot all
    xlist = []
    ylist1 = []
    ylist2 = []
    
    errs = np.array(get_errs(model))
    err_min = np.amin(errs);err_max=np.amax(errs)        
    errs_range = np.linspace(err_min,err_max,100)
    seeds_sorted = np.argsort(errs)[::-1]

    for seed in seeds_sorted:
        fname = lib.get_parameter_fname(model,seed,method='de')
        pars = np.loadtxt(fname)
        color, s, zorder = get_scatter_params(errs,errs[seed])

        xlist.append(pars[0])
        
        if model == 't1d':
            ylist1.append(pars[1])
            ylist2.append(pars[2])

        if model == 'jammingd':
            ylist1.append(pars[3])
            ylist2.append(pars[1]/pars[2])
        
    cbformat = matplotlib.ticker.ScalarFormatter()
    cbformat.set_powerlimits((0,0))
    
    err_exp = np.round(np.exp(errs[seeds_sorted]),4)

    if model == 'jammingd':
        s = 1.05
    else:
        s = 1
    im1 = axs_split[0].scatter(xlist,ylist1,c=err_exp,s=s,zorder=zorder,alpha=.8,
                               vmin=np.amin(err_exp),vmax=np.amax(err_exp)*s)
    im2 = axs_split[1].scatter(xlist,ylist2,c=err_exp,s=s,zorder=zorder,alpha=.8)

    cb = fig.colorbar(im1,ax=[axs_split[0],axs_split[1]],format=cbformat,shrink=.8)
    cb.ax.yaxis.get_offset_text().set_horizontalalignment('left')
    cb.ax.yaxis.get_offset_text().set_position((-1,0))

    ylabelpad = -.45
    if model == 't1d':
        axs_split[0].set_ylim(-2,22)
        axs_split[1].set_ylim(-.4,4.4)
        
        axs_split[0].text(ylabelpad,0.5,r'$d_p^*$',va='center',rotation=90,
                          transform=axs_split[0].transAxes,size=fsizelabel)
        axs_split[1].text(ylabelpad,0.5,r'$\bar{u}$',va='center',rotation=90,
                          transform=axs_split[1].transAxes,size=fsizelabel)

        axs_split[0].text(0.1,0.8,'T1D',va='center',
                          transform=axs_split[0].transAxes,size=fsizelabel)

    if model == 'jammingd':
        axs_split[0].set_ylim(-2,22)
        axs_split[1].set_ylim(-.1,1.1)
        
        axs_split[0].text(ylabelpad,0.5,r'$d_p^*$',va='center',rotation=90,
                          transform=axs_split[0].transAxes,size=fsizelabel)
        axs_split[1].text(ylabelpad,0.5,r'\phantom{1}\\ $V_\text{max}^*/u_\text{max}^*$',va='center',rotation=90,
                          transform=axs_split[1].transAxes,size=fsizelabel)

        axs_split[0].text(0.1,0.8,'T2D',va='center',
                          transform=axs_split[0].transAxes,size=fsizelabel)
        
    return axs_split


def identifiability2():
    """
    identifiable and non-id. models
    """

    import matplotlib
    from matplotlib.patches import Rectangle
    import matplotlib.pyplot as plt
    from matplotlib.gridspec import GridSpec
    import matplotlib.gridspec as gridspec
    import matplotlib.ticker as ticker

    from matplotlib import cm

    nrows=2*3;ncols=4
    fig = plt.figure(figsize=(10,5.4))
    #gs = GridSpec(nrows=nrows,ncols=ncols)
    gs = GridSpec(3,2,height_ratios=[1,1,1.4])

    gs_t1 = gridspec.GridSpecFromSubplotSpec(2,2,subplot_spec=gs[:2,0],wspace=.53,hspace=.05)
    gs_t2 = gridspec.GridSpecFromSubplotSpec(2,2,subplot_spec=gs[:2,1],wspace=.53,hspace=.05)
    gs_t3 = gridspec.GridSpecFromSubplotSpec(1,2,subplot_spec=gs[2:,0],wspace=.53,hspace=.05)

    gs_t1d = gridspec.GridSpecFromSubplotSpec(2,1,subplot_spec=gs_t1[1,1],hspace=0)
    gs_t2d = gridspec.GridSpecFromSubplotSpec(2,1,subplot_spec=gs_t2[1,1],hspace=0)

    axs_t1 = [[fig.add_subplot(gs_t1[i,j]) for j in range(2)] for i in range(2) ]
    axs_t2 = [[fig.add_subplot(gs_t2[i,j]) for j in range(2)] for i in range(2) ]
    axs_t3 = [fig.add_subplot(gs_t3[0,j]) for j in range(2)]

    axs_t1d = [fig.add_subplot(gs_t1d[i,0]) for i in range(2)]
    axs_t2d = [fig.add_subplot(gs_t2d[i,0]) for i in range(2)]

    axs_t1[1][1].axis('off')
    axs_t2[1][1].axis('off')

    # manual loop over index... easier to do this for now.
    mechanism = ['t1','jamming']
    scenario = ['a','b','c','d','e']

    t1_models = [['t1a','t1b'],['t1c','t1d']]
    t2_models = [['jamminga','jammingb'],['jammingc','jammingd']]
    t3_models = ['t1e','t1f']
    
    cbformat = matplotlib.ticker.ScalarFormatter()

    ylabelpad = -.4

    # plot points for T1
    ylims = [[(-2,22),(-.4,4.4)],[(-.4,4.4),(0,0)],[(-2,22),(-.4,4.4)]]
    ylabels = [[r'$d_f^*$',r'$\bar{u}^*$'],[r'$\bar{u}^*$',r''],[r'$d_p^*$','$d_p^*$']]
    mlabels_t1 = [['T1A','T1B'],['T1C','T1D']]
    mlabels_t3 = [['T3E','T3F']]

    for i in range(2):
        for j in range(2):

            if i == 1 and j == 1:
                pass
            else:

                err_exp,xlist,ylist = collect_identifiability_data(t1_models[i][j])

                cbformat = matplotlib.ticker.ScalarFormatter()
                cbformat.set_powerlimits((1e-4,10))

                
                im = axs_t1[i][j].scatter(xlist,ylist,c=err_exp,alpha=.8,s=5,
                                          vmin=np.amin(err_exp),vmax=np.amax(err_exp)*1.01)
                cb = fig.colorbar(im,ax=axs_t1[i][j],format=cbformat,shrink=.8)
                cb.ax.yaxis.get_offset_text().set_horizontalalignment('left')
                cb.ax.yaxis.get_offset_text().set_position((-1,0))

                axs_t1[i][j].set_xlim(-.1,1.1)
                axs_t1[i][j].set_ylim(ylims[i][j][0],ylims[i][j][1])

                if i <= 0 and j <= 1:
                    axs_t1[i][j].tick_params(axis='x',direction='in')
                    axs_t1[i][j].set_xticklabels([])

                axs_t1[i][j].text(ylabelpad,0.5,ylabels[i][j],va='center',rotation=90,
                                  transform=axs_t1[i][j].transAxes,size=fsizelabel)
                
                axs_t1[i][j].text(0.1,0.9,mlabels_t1[i][j],va='center',
                                  transform=axs_t1[i][j].transAxes,size=fsizelabel)
    
    # plot points for J
    ylims = [[(-.1,1.1),(-.4,4.4)],[(-.4,4.4),(0,0)]]
    ylabels = [[r'$V_\text{max}^*$',r'$u_\text{max}^*$'],[r'$u_\text{max}^*$',r'']]
    mlabels = [['T2A','T2B'],['T2C','T2D']]

    for i in range(2):
        for j in range(2):

            if i == 1 and j == 1:
                pass
            else:

                err_exp,xlist,ylist = collect_identifiability_data(t2_models[i][j])

                cbformat = matplotlib.ticker.ScalarFormatter()
                cbformat.set_powerlimits((1e-4,10))

                if t2_models[i][j] == 'jammingc':
                    s = 1.01
                else:
                    s = 1

                im = axs_t2[i][j].scatter(xlist,ylist,c=err_exp,alpha=.8,s=5,
                                          vmin=np.amin(err_exp),vmax=np.amax(err_exp)*s)
                cb = fig.colorbar(im,ax=axs_t2[i][j],format=cbformat,shrink=.8)
                cb.ax.yaxis.get_offset_text().set_horizontalalignment('left')
                cb.ax.yaxis.get_offset_text().set_position((-1,0))

                axs_t2[i][j].set_xlim(-.1,1.1)
                axs_t2[i][j].set_ylim(ylims[i][j][0],ylims[i][j][1])

                if i <= 0 and j <= 1:
                    axs_t2[i][j].tick_params(axis='x',direction='in')
                    axs_t2[i][j].set_xticklabels([])

                axs_t2[i][j].text(ylabelpad,0.5,ylabels[i][j],va='center',rotation=90,
                                  transform=axs_t2[i][j].transAxes,size=fsizelabel)
                
                axs_t2[i][j].text(0.1,0.9,mlabels[i][j],va='center',
                                  transform=axs_t2[i][j].transAxes,size=fsizelabel)
                    
    # T3E, F    
    ylims = [(-2,22),(-.4,4.4)]
    ylabels = [r'$d_p^*$','$d_p^*$']
    mlabels = ['T3E','T3F']

    for i in range(2):

        err_exp,xlist,ylist = collect_identifiability_data(t3_models[i])

        cbformat = matplotlib.ticker.ScalarFormatter()
        cbformat.set_powerlimits((1e-4,10))

        if i == 1:
            s = 1.001
        else:
            s = 1
        
        im = axs_t3[i].scatter(xlist,ylist,c=err_exp,alpha=.8,s=5,
                               vmin=np.amin(err_exp),vmax=np.amax(err_exp)*s)
        cb = fig.colorbar(im,ax=axs_t3[i],format=cbformat,shrink=.8)
        cb.ax.yaxis.get_offset_text().set_horizontalalignment('left')
        cb.ax.yaxis.get_offset_text().set_position((-1,0))

        axs_t3[i].set_xlim(-.1,1.1)
        axs_t3[i].set_ylim(ylims[i][0],ylims[i][1])
        axs_t3[i].tick_params(axis='x',direction='in')

        axs_t3[i].text(ylabelpad,0.5,ylabels[i],va='center',rotation=90,
                          transform=axs_t3[i].transAxes,size=fsizelabel)

        axs_t3[i].text(0.1,0.9,mlabels[i],va='center',
                          transform=axs_t3[i].transAxes,size=fsizelabel)

        #if i < 1:
        #    axs_t3[i].set_xticklabels([])
                        
    # x-axis labels
    axs_t1[-1][0].set_xlabel(r'$\varepsilon^*$',size=fsizelabel);axs_t1[-1][0].xaxis.labelpad=0
    axs_t1d[1].set_xlabel(r'$\varepsilon^*$',size=fsizelabel);axs_t1d[1].xaxis.labelpad=0
    #axs_t1[-1][1].set_xlabel(r'$\varepsilon^*$',size=fsizelabel)

    axs_t2[-1][0].set_xlabel(r'$\varepsilon^*$',size=fsizelabel);axs_t2[-1][0].xaxis.labelpad=0
    axs_t2d[1].set_xlabel(r'$\varepsilon^*$',size=fsizelabel);axs_t2d[1].xaxis.labelpad=0
    

    axs_t3[0].set_xlabel(r'$\varepsilon^*$',size=fsizelabel);axs_t3[0].xaxis.labelpad=0
    axs_t3[1].set_xlabel(r'$\varepsilon^*$',size=fsizelabel);axs_t3[1].xaxis.labelpad=0

    # custom label heights
    
        
    # titles    
    fig.subplots_adjust(top=.95,right=.96,left=.06,bottom=0.06,hspace=.8,wspace=.3)
    
    x1t1 = axs_t1[0][0].axes.get_position().x0
    x2t1 = axs_t1[0][1].axes.get_position().x1

    x1t2 = axs_t2[0][0].axes.get_position().x0
    x2t2 = axs_t2[0][1].axes.get_position().x1

    x1t3 = axs_t3[0].axes.get_position().x0
    x2t3 = axs_t3[1].axes.get_position().x1    
    
    plt.text((x1t1+x2t1)/2,.965, '(a) T1',transform=fig.transFigure,size=fsizelabel+2,ha='center')
    plt.text((x1t2+x2t2)/2,.965, '(b) T2', transform=fig.transFigure,size=fsizelabel+2,ha='center')
    plt.text((x1t3+x2t3)/2,.31, '(c) T3', transform=fig.transFigure,size=fsizelabel+2,ha='center')
    
    # plot T1d, Jd
    axs_t1d = plot_axs_split2(fig,axs_t1d,'t1d')
    axs_t2d = plot_axs_split2(fig,axs_t2d,'jammingd')

    #axs_t2[0][1].yaxis.labelpad=-200

    return fig

def generate_figure(function, args, filenames, title="", title_pos=(0.5,0.95)):
    """
    code taken from Shaw et al 2012 code
    """

    fig = function(*args)
    fig.text(title_pos[0], title_pos[1], title, ha='center')
    
    if type(filenames) == list:
        for name in filenames:
            fig.savefig(name)
    else:
        fig.savefig(filenames)

def f_sol_names(model,method):
    """
    return list of fnames
    model: t1a -- jammingd
    """
    
    fname_pre = 'figs/f_sol_'+str(model)
    if method == '':
        pass
    else:
        fname_pre += '_method='+str(method)
    
    return [fname_pre+'.png',fname_pre+'.pdf']
        
def main():

    method = 'de'
    
    figures = [
        #(experiment_figure, [], ['figs/f_experiment.png','figs/f_experiment.pdf']),
        #(experiment_figure2, [], ['figs/f_experiment2.png','figs/f_experiment2.pdf']),
        #(data_figure, [], ['figs/f_data.png','figs/f_data.pdf']),
        (data_figure2, [10], ['figs/f_data10.png','figs/f_data10.pdf']),
        
        #(gaussian_fit, [], ['figs/f_gaussian_fit.png','figs/f_gaussian_fit.pdf']),
        #(gaussian_fit2, [], ['figs/f_gaussian_fit2.png','figs/f_gaussian_fit2.pdf']),
        
        #(velocity, [], ['figs/f_velocity.png','figs/f_velocity.pdf']),
        #(solution_schematic,[],['figs/f_solution_schematic.png','figs/f_solution_schematic.pdf']),
        #(identifiability,[],['figs/f_identifiability.png','figs/f_identifiability.pdf']),
        #(identifiability2,[],['figs/f_identifiability2.png','figs/f_identifiability2.pdf']),
        
        #(cost_function_t1e, [False], ['figs/f_cost_function_t1e.png','figs/f_cost_function_t1e.pdf']),
        #(cost_function_t1f, [False], ['figs/f_cost_function_t1f.png','figs/f_cost_function_t1f.pdf']),
        
        #(solution,['t1a',False,method],f_sol_names('t1a',method)),
        #(solution,['t1b',False,method],f_sol_names('t1b',method)),
        #(solution,['t1c',False,method],f_sol_names('t1c',method)),
        #(solution,['t1d',False,method],f_sol_names('t1d',method)),
       
        #(solution,['t1e',False,method],f_sol_names('t1e',method)),
        #(solution,['t1f',False,method],f_sol_names('t1f',method)),
       
        #(solution,['t2a',False,method],f_sol_names('t2a',method)),
        #(solution,['t2b',False,method],f_sol_names('t2b',method)),
        #(solution,['t2c',False,method],f_sol_names('t2c',method)),
        #(solution,['t2d',False,method],f_sol_names('t2d',method)),
        
        #(solution,['jamminga',False,method],f_sol_names('ja',method)),
        #(solution,['jammingb',False,method],f_sol_names('jb',method)),
        #(solution,['jammingc',False,method],f_sol_names('jc',method)),
        #(solution,['jammingd',False,method],f_sol_names('jd',method)),

        #(solution,['t1e',True,method],['figs/f_validation.png','figs/f_validation.pdf']),
        #(proof_c,[],['f_proof_c.png','f_proof_c.pdf'])
        ]

    # multiprocessing code from Kendrick Shaw
    # set up one process per figure
    processes = [multiprocessing.Process(target=generate_figure, args=args)
                 for args in figures]

    # start processes
    for p in processes:
        p.start()

    # wait
    for p in processes:
        p.join()
    
    for fig in figures:
        generate_figure(*fig)


if __name__ == "__main__":
    main()
