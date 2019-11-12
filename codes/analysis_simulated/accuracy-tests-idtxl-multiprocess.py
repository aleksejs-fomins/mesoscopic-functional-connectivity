import os, sys
import h5py
import numpy as np
import copy

from codes.lib.data_io.qt_wrapper import gui_fnames, gui_fname
from codes.lib.analysis import simulated

'''
  Width/Depth Test Code:
    Load all width files
    Compute TE
    Make Plots, Store TE

  SNR Code:
    Load Typical File
    Loop over added noise / freq
    Compute TE
    Make Plots, Store TE

  Win/Lag/DS Code:
    Load Typical File
    Loop over windows, lag, ds
    Compute TE
    Make Plots, Store TE
'''

nNode = 12
dataFileNames = gui_fnames("Get simulated data files", "./", "hdf5 (*.h5)")
typicalFileName = gui_fname("Get typical data file", os.path.dirname(dataFileNames), "hdf5 (*.h5)")

#############################
# IDTxl parameters
#############################
idtxlSettings = {
    'dim_order'       : 'rsp',
    'methods'         : ['BivariateMI', 'MultivariateMI', 'BivariateTE', 'MultivariateTE'],
    'cmi_estimator'   : 'JidtGaussianCMI',
    'max_lag_sources' : 1,
    'min_lag_sources' : 1}

#############################
# Width / Depth Tests
#############################
simulated.analysis_width_depth(dataFileNames, idtxlSettings)

#############################
# SNR Tests
#############################
nStep = 40  # Number of different data sizes to pick
simulated.analysis_snr(typicalFileName, "dynsys", idtxlSettings, nStep)

################
# Window / Lag / Downsample
################
windowRange = np.arange(2, 11)
simulated.analysis_window(typicalFileName, idtxlSettings, windowRange)

simulated.anaylsis_lag(typicalFileName, idtxlSettings, window=6)

downsampleFactors = [1,2,4,6,8,10,12]
simulated.analysis_downsample(typicalFileName, idtxlSettings, downsampleFactors)