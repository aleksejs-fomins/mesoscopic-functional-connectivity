import numpy as np


##################################
# All routines in this file assume shape [nTrial, nTime, nChannel]
##################################

def mean_variance_plots(ax, data):
    nChannel = data.shape[2]
    mu = np.mean(data, axis=(0, 1))
    std = np.std(data, axis=(0, 1))
    ax.errorbar(np.arange(nChannel), mu, yerr=std, linestyle='.')


