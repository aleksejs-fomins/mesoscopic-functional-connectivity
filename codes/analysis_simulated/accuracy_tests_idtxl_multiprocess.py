import os, sys

# Export library path
rootname = "mesoscopic-functional-connectivity"
thispath   = os.path.dirname(os.path.abspath(__file__))
rootpath = os.path.join(thispath[:thispath.index(rootname)], rootname)
print("Appending project path", rootpath)
sys.path.append(rootpath)

from codes.lib.data_io.qt_wrapper import gui_fnames, gui_fname
from codes.lib.analysis import fc_accuracy_analysis


dataFileNames = gui_fnames("Get simulated data files", "./", "hdf5 (*.h5)")
#typicalFileName = gui_fname("Get typical data file", os.path.dirname(dataFileNames[0]), "hdf5 (*.h5)")

#############################
# Simulation parameters
#############################
param = {
    'library' :  "idtxl",
    'methods' :  ['BivariateMI', 'MultivariateMI', 'BivariateTE', 'MultivariateTE'],
    'pTHR'    :  0.01,
    'figExt'  :  '.svg',
    'parTrg'  :  True,
    'nCore'   :  4,
    'serial'  :  False,
    'paramLib' : {  # IDTxl parameters
        'dim_order'       : 'rsp',
        'cmi_estimator'   : 'JidtGaussianCMI',
        'max_lag_sources' : 1,
        'min_lag_sources' : 1
    }
}

#############################
# Width / Depth Tests
#############################
# simulated.analysis_width_depth(dataFileNames, param)

#############################
# SNR Tests
#############################
nStep = 40  # Number of different data sizes to pick
fc_accuracy_analysis.analysis_snr(dataFileNames, "dynsys", nStep, param)

# ################
# # Window / Lag / Downsample
# ################
# wMin = 2
# wMax = 2
# fc_accuracy_analysis.analysis_window(dataFileNames, wMin, wMax, param)
#
# ################
# # Lag
# ################
# lMin = 1
# lMax = 5
# fc_accuracy_analysis.analysis_lag(dataFileNames, lMin, lMax, param)
#
# ################
# # Downsample
# ################
# downsampleFactors = [1,2,4,6,8,10,12]
# fc_accuracy_analysis.analysis_downsample(dataFileNames, downsampleFactors, param)