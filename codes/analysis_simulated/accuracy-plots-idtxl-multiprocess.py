import os, sys
import h5py
import numpy as np
import copy

from codes.lib.data_io.qt_wrapper import gui_fnames
from codes.lib.plots.accuracy import fc_accuracy_plots_fromfile


#############################
# Files
#############################
dataFileNames = gui_fnames("Get simulated data files", "./", "hdf5 (*.h5)")

#############################
# Params
#############################
pTHR = 0.01     # P-value for IDTxl thresholding
fExt = '.svg'   # Request vector graphics. Can also use .png
methods = ['BivariateMI', 'MultivariateMI', 'BivariateTE', 'MultivariateTE']


for fname in dataFileNames:
    fig_fname = fname[:-3] + fExt



    #############################
    # Width / Depth plots
    #############################
    if ("width" in fname) or ("depth" in fname):
        fc_accuracy_plots_fromfile(dataFileNames, methods, pTHR, logx=True, percenty=True, fig_fname=fig_fname)


    #############################
    # SNR plots
    #############################
    if "snr" in fname:
        pass

    #############################
    # Window plots
    #############################
    if "window" in fname:
        pass

    #############################
    # Lag plots
    #############################
    if "lag" in fname:
        pass

    #############################
    # Downsample plots
    #############################
    if "downsample" in fname:
        pass