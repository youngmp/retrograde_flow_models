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
mpl.rcParams['text.latex.preamble'] = r'\usepackage{amsmath} \usepackage{siunitx} \usepackage{enumitem} \usepackage{xcolor}'

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
    
    pars = {'eps':.3,'df':0,'dp':.01,'T':1500,'dt':.01,'N':100,'Nvel':1,'u_nonconstant':True}

    p = pde.PDEModel(**pars)
    p._run_euler('2')
        
    #p = PDEModel()
    L0=p.L0; L=p.L
    
    nrows=5;ncols=2
    
    fig,axs = plt.subplots(nrows=nrows,ncols=ncols,figsize=(8,8),sharey='col')

    r = np.linspace(L0,L)
    m = 1/(L0-L)

    # initial
    f1 = p.y[:p.N,0]
    p1 = p.y[p.N:,0]

    #axs[2,0].plot(p.r,f1,color='gray',ls='--')
    #axs[2,1].plot(p.r,p1,color='gray',ls='--')
    
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
    

    #axs[3,0].plot(p.r,f1,color='gray',ls='--',label='Initial Solution')
    #axs[3,1].plot(p.r,p1,color='gray',ls='--')
    
    # steady-state
    f3 = p.y[:p.N,-1]
    p3 = p.y[p.N:,-1]

    axs[4,0].plot(p.r,f3,color='tab:blue',lw=2)
    axs[4,1].plot(p.r,p3,color='tab:orange',lw=2)
    
    #axs[2,0].text((L+L0)/2,2.7,'Initial',ha='center',size=15)
    axs[3,0].text((L+L0)/2,2.7,'',ha='center',size=15)

    plt.text(.12,.94,"A.",transform=fig.transFigure,size=fsizetitle)
    plt.text(.12,.83,"B.",transform=fig.transFigure,size=fsizetitle)
    
    axs[2,0].set_title(r'C. Initial ($t=0$)',loc='left',size=fsizetitle)
    axs[3,0].set_title(r'D. $t>0$',loc='left',size=fsizetitle)
    axs[4,0].set_title(r'E. Steady-State ($t\rightarrow \infty$)',loc='left',size=fsizetitle)
    
    axs[2,1].set_title(r'F. Initial ($t=0$)',loc='left',size=fsizetitle)
    axs[3,1].set_title(r'G. $t>0$',loc='left',size=fsizetitle)
    axs[4,1].set_title(r'H. Steady-State  ($t\rightarrow \infty$)',loc='left',size=fsizetitle)
    
    axs[2,0].set_ylabel('$F$',fontsize=fsizelabel)
    axs[3,0].set_ylabel('$F$',fontsize=fsizelabel)
    axs[4,0].set_ylabel('$F$',fontsize=fsizelabel)
    
    axs[2,1].set_ylabel('$P$',fontsize=fsizelabel)
    axs[3,1].set_ylabel('$P$',fontsize=fsizelabel)
    axs[4,1].set_ylabel('$P$',fontsize=fsizelabel)
    

    for i in range(2):
        for j in range(2):
            #axs[j,i].set_xticks([])
            #axs[j,i].set_yticks([])
            axs[j,i].axis('off')
    
    for i in range(2,ncols):
        
        axs[i,0].set_ylim(0,2)
        axs[i,1].set_ylim(-.1,1.1)

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
            #ticklabel_format(axis='y',style='scientific',scilimits=(0,0))
            #axs[i,j].get_yaxis().get_offset_text().set_position((0,0))
    
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

    ax_i0.set_title(r'Initial Data $\tilde I_0(r) = F(r,0)+P(r,0)$',size=fsizetitle)
    ax_i0.plot(p.r,p.data_avg_fns['control'](p.r),lw=2,color='k')
    ax_i0.ticklabel_format(axis='y',style='scientific',scilimits=(0,0))

    #bbox=dict(boxstyle="round",fc='tab:blue',alpha=0.4)
    ax_i0.annotate(r'$F(r,0) = \varepsilon\tilde I_0(r)$',xy=(.1,-.6),xycoords='axes fraction',size=15,ha='right',bbox=dict(boxstyle="round",fc='tab:blue',alpha=0.4))
    ax_i0.annotate(r'',xy=(.15, -1), xycoords='axes fraction', xytext=(0.5,-.1), arrowprops=dict(arrowstyle="simple,head_width=1,head_length=1", color='tab:blue'))

    ax_i0.annotate(r'$P(r,0) = (1-\varepsilon)\tilde I_0(r)$',xy=(.9,-.6),xycoords='axes fraction',size=15,ha='left',bbox=dict(boxstyle="round",fc='tab:orange',alpha=0.4))
    ax_i0.annotate(r'',xy=(.85, -1), xycoords='axes fraction', xytext=(.5,-.1), arrowprops=dict(arrowstyle="simple,head_width=1,head_length=1", color='tab:orange'))


    ax_i0.set_xticks([L0,L])
    ax_i0.set_xticklabels([r'$L_0$ (Nuclear Envelope)',r'$L$ (Cell Membrane)'])    
    
    
    # F to P and vice-versa diagram
    c1 = (axs[3,0].get_position().x0+axs[3,0].get_position().x1)/2
    c2 = (axs[3,1].get_position().x0+axs[3,1].get_position().x1)/2
    plt.text(c1, .9, "$F$: Immobile Material", ha="center", va="center", size=15,
             transform=fig.transFigure,bbox=dict(boxstyle="round",fc='tab:blue',alpha=0.4))

    plt.text(c2, .9, "$P$: Material Subject to\n Retrograde Flow", ha="center", va="center", size=15,transform=fig.transFigure,bbox=dict(boxstyle="round",fc='tab:orange',alpha=0.4))

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
    
    plt.text(x,.87,r'$d_p$',size=fsizelabel,ha='center',transform=fig.transFigure)
    plt.text(x,.92,r'$d_f$',size=fsizelabel,ha='center',transform=fig.transFigure)

    
    # arrow from P to F in plots, vice-versa.
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

    plt.text(x-.03,y+.02,r'$d_f$',size=fsizelabel,ha='center',transform=fig.transFigure)
    plt.text(x-.03,y-.03,r'$d_p$',size=fsizelabel,ha='center',transform=fig.transFigure)

    plt.text(x-.03,y+.05,r'(Trapping)',size=fsizelabel,ha='center',transform=fig.transFigure)
    

    # advection label
    arrow_str = 'simple, head_width=.7, tail_width=.15, head_length=1'
    axs[3,1].annotate('',(.2,.4),(.4,.4),ha='left',
                      va='center',
                      xycoords='axes fraction',
                      arrowprops=dict(arrowstyle=arrow_str,
                                      fc='k',
                                      ec='k'))
    
    advection_label = (r'Advection:\begin{itemize}[noitemsep,topsep=0pt,labelsep=.1em,label={}]'
                       r'\setlength{\itemindent}{-1.5em}'
                       r'\item $u$ constant'
                       r'\item $u(r)$ space-dependent'
                       r'\item $u(I)$ conc.-dependent'
                       r'\end{itemize}')
    
    axs[3,1].annotate(advection_label,(.4,.5),(.4,.5),xycoords='axes fraction',va='center')

    curly_str = (r'$'
                 r'\begin{cases}'
                 r'\phantom{1}\\'
                 r'\phantom{1}\\'
                 r'\phantom{1}\\'
                 r'\end{cases}'
                 r'$')
    #curly_str = (r'\begin{equation}'
    #             r'test'
    #            r'\end{equation}')
    axs[3,1].annotate(curly_str,(.4,.4),(.4,.4),xycoords='axes fraction',size=10,va='center')
    

    
    
    # time arrow
    plt.annotate(r'',xy=(.05,.05), xycoords='figure fraction', xytext=(.05, .56), arrowprops=dict(arrowstyle="simple,head_width=1,head_length=1", color='k'))
    
    plt.text(.034,.25,'Time',ha='center',transform=fig.transFigure,size=20,rotation=90)


    # update scientific notation
    fig.canvas.draw()

    offset = ax_i0.get_yaxis().get_offset_text().get_text()
    #offset = ax_i0.yaxis.get_major_formatter().get_offset()
    
    
    print(offset)
    
    ax_i0.yaxis.offsetText.set_visible(False)
    ax_i0.yaxis.set_label_text(r"$\tilde I_0$" + " (" + offset+")",size=fsizelabel)
    ax_i0.tick_params(axis='both',labelsize=fsizetick)


    
    for i in range(2,nrows):
        for j in range(ncols):
            # remove y axis label

            offset = axs[i,j].get_yaxis().get_offset_text().get_text()

            axs[i,j].yaxis.offsetText.set_visible(False)
            ylabel = axs[i,j].yaxis.get_label().get_text()
            #print(ylabel.__dict__)
            axs[i,j].yaxis.set_label_text(ylabel + " (" + offset+")")

            axs[i,j].tick_params(axis='both',labelsize=fsizetick)
            #axs[i,j].tick_params(axis='y',which='both',left=False,labelleft=False)



    return fig

def velocity():

    import matplotlib.pyplot as plt
    from matplotlib.gridspec import GridSpec


    fig = plt.figure(figsize=(6,4))
    gs = GridSpec(2, 2,hspace=0.6,wspace=.8)
    #gs = GridSpec(2, 2)
    #gs.update()

    ax1 = fig.add_subplot(gs[:,0])
    ax2 = fig.add_subplot(gs[0,1])
    ax3 = fig.add_subplot(gs[1,1])

    axs = [ax1,ax2,ax3]

    nrows=2;ncols=2
    #fig,axs = plt.subplots(nrows=nrows,ncols=ncols,figsize=(6,4))

    pars = {'eps':1,'df':0,'dp':.01,'T':60,'dt':.01,'N':100,'Nvel':1,'u_nonconstant':False,
            'us0':0.2}

    p = pde.PDEModel(**pars)
    p._run_euler('2')
    I = p.y[:p.N,-1] + p.y[p.N:,-1]
    imax = np.amax(I)

    print(imax)

    axs[0].plot(p.r,p._s2_vel(),lw=2)
    axs[1].plot(p.r,I,lw=2)
    axs[2].plot(p.r,p.us0*(1-I/imax),lw=2)

    for i in range(3):
        axs[i].set_xticks([p.L0,p.L])
        axs[i].set_xticklabels([r'$L_0$',r'$L$'])
        axs[i].tick_params(axis='both',labelsize=fsizetick)

    axs[0].set_ylabel(r'$u(r)$',size=fsizelabel)
    axs[1].set_ylabel(r'$I(r,t)$',size=fsizelabel)
    axs[2].set_ylabel(r'$u(I(r,t))$',size=fsizelabel)
    
    axs[0].set_title(r'A. Spatially-Dependent',loc='left',size=fsizetitle)
    axs[1].set_title(r'B. Jamming Velocity',loc='left',size=fsizetitle)
    axs[2].set_title(r'',loc='left',size=fsizetitle)

    # center left plot
    left = axs[0].get_position().x0
    bottom = (axs[2].get_position().y1+axs[2].get_position().y0)/1.6
    width = (axs[0].get_position().x1 - axs[0].get_position().x0)
    height = ((axs[1].get_position().y1+axs[1].get_position().y0)/2 - bottom)*.9

    print([left,bottom,width,height])
    axs[0].set_position([left,bottom,width,height])


    # arrow betwen right plots

    x_start = (axs[1].get_position().x0+axs[1].get_position().x1)/2
    y_start = (axs[1].get_position().y0)-.05
    
    plt.annotate(r'',xytext=(x_start, y_start), xy=(x_start,y_start-.1),
                 ha='center', xycoords='figure fraction',arrowprops=dict(arrowstyle="simple,tail_width=1.4,head_width=2.3,head_length=1", color='tab:blue'))
    

    #print(y2,y1)
    
    #axs[1].set_title(r'D. ?',loc='left')

    #plt.tight_layout()

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
        #(data_figure, [], ['f_data.png','f_data.pdf']),
        #(gaussian_fit, [], ['f_gaussian_fit.png','f_gaussian_fit.pdf']),
        #(solution_schematic, [], ['f_solution_schematic.png','f_solution_schematic.pdf']),
        (velocity, [], ['f_velocity.png','f_velocity.pdf'])
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
