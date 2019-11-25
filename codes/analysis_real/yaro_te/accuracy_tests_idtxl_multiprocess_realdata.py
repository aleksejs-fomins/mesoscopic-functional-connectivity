import os, sys
import numpy as np

# Export library path
rootname = "mesoscopic-functional-connectivity"
thispath   = os.path.dirname(os.path.abspath(__file__))
rootpath = os.path.join(thispath[:thispath.index(rootname)], rootname)
print("Appending project path", rootpath)
sys.path.append(rootpath)

from codes.lib.data_io.qt_wrapper import gui_fpath
from codes.lib.analysis import fc_accuracy_analysis
from codes.lib.data_io.yaro.yaro_data_read import read_neuro_perf

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
# Load and read data file
#############################

fpath_realdata = gui_fpath("Get real data filepath", "./")
data, behaviour, performance = read_neuro_perf(fpath_realdata)
dataGo = data[np.array(behaviour["iGO"], dtype=int) - 1]
dataBaseName = "realdata"

#############################
# SNR Tests
#############################
nStep = 40  # Number of different data sizes to pick
fc_accuracy_analysis.analysis_snr(dataGo, None, nStep, dataBaseName + '.h5', param)

################
# Window
################
wMin = 2
wMax = 10
fc_accuracy_analysis.analysis_window(dataGo, None, wMin, wMax, dataBaseName + '.h5', param)


################
# Lag
################
lMin = 1
lMax = 5
fc_accuracy_analysis.analysis_lag(dataGo, None, lMin, lMax, dataBaseName + '.h5', param)

################
# Downsample
################
downsampleFactors = [1,2,4,6,8,10,12]
fc_accuracy_analysis.analysis_downsample(dataGo, None, downsampleFactors, dataBaseName + '.h5', param)