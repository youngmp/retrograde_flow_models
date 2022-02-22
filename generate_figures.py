import matplotlib as mpl
#from matplotlib import rc
#mpl.rc('text',usetex=True)
#mpl.rc('font',family='serif', serif=['Computer Modern Roman'])

#pgf_with_latex = {"pgf.preamble":"\n".join([r'\usepackage{siunitx}',
#                                            r'\userpackage{fontenc}'])}

#mpl.rcParams.update(pgf_with_latex)
mpl.rcParams['font.family'] = 'serif'
mpl.rcParams['text.usetex'] = True
mpl.rcParams['pgf.texsystem'] = 'pdflatex'
mpl.rcParams['text.latex.preamble'] = r'\usepackage{amsmath} \usepackage{siunitx}'

fsizetick = 13
fsizelabel = 13
fsizetitle = 13


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
    data_avg_raw, data_rep_raw = d._build_data_dict(L0=d.L0,L=d.L,normed=False)

    
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
        x = data_rep_raw[hour][:,0]
        y = data_rep_raw[hour][:,1]
        axs[0,0].plot(x,y,label=label,color=color)

        # raw, avg
        x = data_avg_raw[hour][:,0]
        y = data_avg_raw[hour][:,1]
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

    plt.tight_layout()

    return fig


def gaussian_fit():
    """
    figure for fitting Gaussian to data
    """

    import matplotlib.pyplot as plt
    
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
        (data_figure, [], ['f_data.png']),
        (data_figure, [], ['f_data_test_plot.png'])
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
