import os, sys

# Export library path
rootname = "mesoscopic-functional-connectivity"
thispath   = os.path.dirname(os.path.abspath(__file__))
rootpath = os.path.join(thispath[:thispath.index(rootname)], rootname)
print("Appending project path", rootpath)
sys.path.append(rootpath)

from codes.lib.data_io.qt_wrapper import gui_fnames, gui_fname
from codes.lib.analysis import simulated


dataFileNames = gui_fnames("Get simulated data files", "./", "hdf5 (*.h5)")
#typicalFileName = gui_fname("Get typical data file", os.path.dirname(dataFileNames[0]), "hdf5 (*.h5)")

#############################
# IDTxl parameters
#############################
methods = ['BivariateMI', 'MultivariateMI', 'BivariateTE', 'MultivariateTE']
idtxlSettings = {
    'dim_order'       : 'rsp',
    'cmi_estimator'   : 'JidtGaussianCMI',
    'max_lag_sources' : 1,
    'min_lag_sources' : 1}

#############################
# Width / Depth Tests
#############################
# simulated.analysis_width_depth(dataFileNames, idtxlSettings, methods)

#############################
# SNR Tests
#############################
nStep = 40  # Number of different data sizes to pick
simulated.analysis_snr(dataFileNames, idtxlSettings, methods, "dynsys", nStep, NCore=4)

#
# ################
# # Window / Lag / Downsample
# ################
# windowRange = np.arange(2, 11)
# simulated.analysis_window(typicalFileName, idtxlSettings, windowRange)
#
# simulated.anaylsis_lag(typicalFileName, idtxlSettings, window=6)
#
# downsampleFactors = [1,2,4,6,8,10,12]
# simulated.analysis_downsample(typicalFileName, idtxlSettings, downsampleFactors)