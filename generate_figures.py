import matplotlib as mpl
#from matplotlib import rc
#mpl.rc('text',usetex=True)
#mpl.rc('font',family='serif', serif=['Computer Modern Roman'])

#pgf_with_latex = {"pgf.preamble":"\n".join([r'\usepackage{siunitx}',
#                                            r'\userpackage{fontenc}'])}

#mpl.rcParams.update(pgf_with_latex)
mpl.rcParams['font.family'] = 'serif'
mpl.rcParams['text.usetex'] = True
mpl.rcParams['text.latex.preview'] = True
mpl.rcParams['pgf.texsystem'] = 'pdflatex'
mpl.rcParams['text.latex.preamble'] = r'\usepackage{amsmath} \usepackage{siunitx}'

fsizetick = 13
fsizelabel = 13
fsizetitle = 13
#tight_layout_pad =  0

import pde

import numpy as np

import os
import multiprocessing

def data_figure():
    """
    figure comparing representative data, average data
    and normalized versions of each
    """
    import matplotlib.pyplot as plt

    # load normed data
    d = pde.Data()

    # load unnormed data
    #data =  d._build_data_dict(L0=d.L0,L=d.L,normed=True)
    
    #list_hours = ['control','0.5h', '1h','2h','4h','8.5h','24h']
    list_hours = ['control','0.5h', '1h','2h','24h']
    #assert(len(list_hours) == len(d.data_avg.keys()))
    #assert(len(d.data_avg.keys()) == len(d.data_rep.keys()))
    
    #fig = plt.figure()
    fig,axs = plt.subplots(nrows=2,ncols=2,figsize=(8,5))
    
    for i,hour in enumerate(list_hours):

        color = str((1-i/len(list_hours))/1.2)

        if hour == 'control':
            label = '0h'
        else:
            label = hour

        if hour == '24h':
            color = 'tab:blue'
        else:
            color = color

        # raw, rep
        x = d.data_rep_raw[hour][:,0]
        y = d.data_rep_raw[hour][:,1]
        axs[0,0].plot(x,y,label=label,color=color)

        # raw, avg
        x = d.data_avg_raw[hour][:,0]
        y = d.data_avg_raw[hour][:,1]
        axs[0,1].plot(x,y,label=label,color=color)
        
        # normed, rep
        x = d.data_rep[hour][:,0]
        y = d.data_rep[hour][:,1]
        axs[1,0].plot(x,y,label=label,color=color)

        # normed, avg
        x = d.data_avg[hour][:,0]
        y = d.data_avg[hour][:,1]
        axs[1,1].plot(x,y,label=label,color=color)

    for i in range(2):
        for j in range(2):
            axs[i,j].legend()
            axs[i,j].set_xlabel(r'Radius ($\si{\um}$)',fontsize=fsizelabel)
            axs[i,j].tick_params(axis='both',labelsize=fsizetick)


    axs[0,0].set_title('A. Representative',loc='left',size=fsizetitle)
    axs[0,1].set_title('B. Average',loc='left',size=fsizetitle)
    axs[1,0].set_title('C. Representative (Normalized)',loc='left',size=fsizetitle)
    axs[1,1].set_title(r'\textbf{D. Average (Normalized)}',loc='left',size=fsizetitle)
    
    axs[0,0].set_ylabel('Intensity',fontsize=fsizelabel)
    axs[0,1].set_ylabel('Intensity',fontsize=fsizelabel)
    axs[1,0].set_ylabel('Norm. Intensity',fontsize=fsizelabel)
    axs[1,1].set_ylabel('Norm. Intensity',fontsize=fsizelabel)

    #plt.tight_layout(pad=tight_layout_pad)
    plt.tight_layout()

    return fig


def gaussian_fit():
    """
    figure for fitting Gaussian to data
    """

    import matplotlib.pyplot as plt
    
    fig = plt.figure(figsize=(4,3))
    ax = fig.add_subplot(111)


    d = pde.Data(recompute=False,normed=True)
    x_data = d.data_avg['control'][:,0]
    y_data = d.data_avg['control'][:,1]
    
    
    ax.plot(x_data,y_data,label='Data',lw=2)
    ax.plot(x_data,d.control_fn(x_data),label='Approx.',lw=2)
    #ax.set_title(t)

    ax.set_xlabel(r'$r$',fontsize=fsizelabel)
    ax.set_ylabel(r'Norm. Intensity $\tilde I_0(r)$',fontsize=fsizelabel)
    ax.tick_params(axis='both',labelsize=fsizetick)

    ax.set_xlim(x_data[0],x_data[-1])
    ax.legend()

    plt.tight_layout()

    return fig

def solution_schematic():

    import matplotlib.pyplot as plt

    fig = plt.figure(figsize=(8,5))
    
    d = pde.Data(recompute=False,normed=True)
    #p = PDEModel()
    L0=d.L0; L=d.L
    
    nrows=2;ncols=3
    
    fig,axs = plt.subplots(nrows=nrows,ncols=ncols,figsize=(8,5))
    

    r = np.linspace(L0,L)
    m = 1/(L0-L)

    # initial
    f1 = 1 + m*(r-L0)
    p1 = 1 + m*(r-L0)

    axs[0,0].plot(r,f1,color='gray',ls='--')
    axs[1,0].plot(r,p1,color='gray',ls='--')

    # intermediate
    f2 = f1 + f1**2/5
    p2 = 1 + m*(r-L0+8)
    print(len(p2),len(r))
    
    axs[0,1].plot(r,f2,color='tab:blue',lw=2,label='Current Solution')
    axs[1,1].plot(r[r<=L-8],p2[r<=L-8],color='tab:blue',lw=2)
    axs[1,1].plot([L-8,L],[0,0],color='tab:blue',lw=2)

    axs[0,1].plot(r,f1,color='gray',ls='--',label='Initial Solution')
    axs[1,1].plot(r,p1,color='gray',ls='--')
    
    # steady-state
    f3 = f2 + f1**2/2
    p3 = np.zeros(len(r))
    axs[0,2].plot(r,f3,color='tab:blue',lw=2)
    axs[1,2].plot(r,p3,color='tab:blue',lw=2)

    axs[0,2].plot(r,f1,color='gray',ls='--')
    axs[1,2].plot(r,p1,color='gray',ls='--')    

    axs[0,0].text((L+L0)/2,2.7,'Initial ($t=0$)',ha='center',size=15)
    axs[0,1].text((L+L0)/2,2.7,'$t>0$',ha='center',size=15)
    axs[0,2].text((L+L0)/2,2.7,r'Steady-State ($t \rightarrow \infty$)',ha='center',size=15)

    # arrow from P to F, vice-versa.
    #axs[2,1].set_clip_on(False)
    a1 = axs[0,1].arrow(19.75,-1.75,0,1,width=.1,ec='k',fc='k',
                        head_width=.8,head_length=.5,
                        **{'shape':'right'})
    a1.set_clip_on(False)
    
    a2 = axs[0,1].arrow(20.25,-.25,0,-1,width=.1,ec='k',fc='k',
                        head_width=.8,head_length=.5,
                        **{'shape':'right'})
    a2.set_clip_on(False)

    axs[0,1].text(18.5,-1.1,r'$d_p$',size=fsizelabel,ha='center')
    axs[0,1].text(21.5,-1.1,r'$d_f$',size=fsizelabel,ha='center')

    # advection label
    arrow_str = 'simple, head_width=.7, tail_width=.15, head_length=1'
    axs[1,1].annotate('',(15,.4),(21,.4),ha='left',
                      arrowprops=dict(arrowstyle=arrow_str,
                                      fc='k',
                                      ec='k'))

    axs[1,1].annotate('Advection',(15.3,.55),(15.3,.55),
                      bbox=dict(boxstyle='round',fc='w',alpha=.7))
    
    #axs[3,1].arrow(25,.5,-3,0,width=.05,ec='k',fc='k',
    #               head_width=.5,head_length=2)
    
    # labels
    axs[0,0].set_title('A',loc='left',size=fsizetitle)
    axs[0,1].set_title('B',loc='left',size=fsizetitle)
    axs[0,2].set_title('C',loc='left',size=fsizetitle)
    
    axs[1,0].set_title('D',loc='left',size=fsizetitle)
    axs[1,1].set_title('E',loc='left',size=fsizetitle)
    axs[1,2].set_title('F',loc='left',size=fsizetitle)

    
    axs[0,0].set_ylabel('Immobile ($F$)',fontsize=fsizelabel)
    axs[1,0].set_ylabel('Mobile ($P$)',fontsize=fsizelabel)
    
    
    
    for i in range(ncols):
        
        axs[0,i].set_ylim(0,2)
        axs[1,i].set_ylim(-.1,1.1)

    for i in range(nrows):
        for j in range(ncols):
            # remove y axis label
            axs[i,j].tick_params(axis='both',labelsize=fsizetick)
            axs[i,j].tick_params(axis='y',which='both',left=False,labelleft=False)

            # x axis label
            axs[i,j].set_xticks([L0,L])
            axs[i,j].set_xticklabels([r'$L_0$',r'$L$'])

            # set axis limit
            axs[i,j].set_xlim(L0,L)
            
    axs[0,1].legend()
            
    plt.tight_layout()
    fig.subplots_adjust(hspace=1,top=.85)
    return fig

def generate_figure(function, args, filenames, title="", title_pos=(0.5,0.95)):
    """
    code taken from Shaw et al 2012 code
    """
    
    #tempfile._name_sequence = None
    fig = function(*args)
    fig.text(title_pos[0], title_pos[1], title, ha='center')
    
    if type(filenames) == list:
        for name in filenames:
            fig.savefig(name)
    else:
        fig.savefig(filenames)

def main():

    figures = [
        (data_figure, [], ['f_data.png','f_data.pdf']),
        (gaussian_fit, [], ['f_gaussian_fit.png','f_gaussian_fit.pdf']),
        (solution_schematic, [], ['f_solution_schematic.png','f_solution_schematic.pdf'])
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
