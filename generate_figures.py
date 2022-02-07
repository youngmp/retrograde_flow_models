from matplotlib import rc
rc('text',usetex=True)
rc('font',family='serif', serif=['Computer Modern Roman'])

import numpy as np
import matplotlib.pyplot as plt

import multiprocessing


def data_figure():
    """
    figure comparing representative data, average data
    and normalized versions of each
    """

    fig = plt.figure()

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
        (data_figure, [], ['test.png'])
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
