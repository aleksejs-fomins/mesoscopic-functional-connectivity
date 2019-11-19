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

dataParamsDict = {
    "analysis"  : ["width", "depth", "snr", "window", "lag", "downsample"],
    "logx"      : [True, True, True, False, False, False],

}
for fname in dataFileNames:
    fig_fname = fname[:-3] + fExt

    fc_accuracy_plots_fromfile(fig_fname, methods, pTHR, logx=True, percenty=True, fig_fname=fig_fname)
