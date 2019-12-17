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
    'nCore'   :  None,
    'serial'  :  False,
    'paramLib' : {  # IDTxl parameters
        'dim_order'       : 'rsp',
        'cmi_estimator'   : 'JidtGaussianCMI',
        'max_lag_sources' : 1,
        'min_lag_sources' : 1
    }
}

startTime = 3.0  # Seconds, time at which to start analyzing data
timestep = 0.05  # Seconds, neuronal recording timestep
startStep = int(startTime / timestep)  # Timestep at which to start analysis

#############################
# Load and read data file
#############################

fpath_realdata = gui_fpath("Get real data filepath", "./")
data, behaviour, performance = read_neuro_perf(fpath_realdata)

for trialType in ["iNOGO"]:
    dataTrial = data[np.array(behaviour[trialType], dtype=int) - 1]
    print("For", trialType, "have shape", dataTrial.shape)
    dataTrial = dataTrial[:, startStep:, :]
    print("After shifting the data by", startStep, "timesteps, the new shape is", dataTrial.shape)
    outnameH5 = trialType + "_realdata.h5"

    #############################
    # SNR Tests
    #############################
    nStep = 40  # Number of different data sizes to pick
    fc_accuracy_analysis.analysis_snr(dataTrial, None, nStep, outnameH5, param)

    ################
    # Window
    ################
    wMin = 2
    wMax = 10
    fc_accuracy_analysis.analysis_window(dataTrial, None, wMin, wMax, outnameH5, param)


    ################
    # Lag
    ################
    lMin = 1
    lMax = 5
    fc_accuracy_analysis.analysis_lag(dataTrial, None, lMin, lMax, outnameH5, param)

    ################
    # Downsample
    ################
    downsampleFactors = [1,2,4,6,8,10,12]
    fc_accuracy_analysis.analysis_downsample(dataTrial, None, downsampleFactors, outnameH5, param)
