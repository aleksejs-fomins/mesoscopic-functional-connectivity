import os, sys
import numpy as np
import matplotlib.pyplot as plt

# Export library path
rootname = "mesoscopic-functional-connectivity"
thispath = os.path.dirname(os.path.abspath(__file__))
rootpath = os.path.join(thispath[:thispath.index(rootname)], rootname)
print("Appending project path", rootpath)
sys.path.append(rootpath)

from codes.lib.info_metrics.info_metrics_generic import parallel_metric_1d
from codes.lib.models.test_lib import dynsys
from codes.lib.sweep_lib import DataSweep


############################
# Parameters
############################

# DynSys parameters
dynsysParam = {
    'nNode'   : 4,    # Number of variables
    'nData'   : 4000,  # Number of timesteps
    'nTrial'  : 20,    # Number of trials
    'dt'      : 50,    # ms, timestep
    'tau'     : 500,   # ms, timescale of each mesoscopic area
    'inpT'    : 100,   # Period of input oscillation
    'inpMag'  : 0.0,   # Magnitude of the periodic input
    'std'     : 0.2,   # STD of neuron noise
}

# IDTxl parameters
npeetParam = {
    'dim_order'   : 'rps',
    'max_lag'     : 1,
    'window': 50
}

############################
# Data
############################

nSweep = 10
data = dynsys(dynsysParam)   #[trial x channel x time]
print("Generated data of shape", data.shape)

methods = ['Entropy', 'PI']
dataSweep1 = DataSweep(data, npeetParam, nSweepMax=nSweep)
timeIdxs = dataSweep1.get_target_time_idxs()

results = parallel_metric_1d(dataSweep1.iterator(), "npeet", methods, npeetParam, nCh=dynsysParam['nNode'], parCh=True, nCore=None)

fig, ax = plt.subplots(nrows=nSweep, ncols=2)
fig.suptitle("TE computation for several windows of the data")
for iMethod, method in enumerate(methods):
    ax[0][iMethod].set_title(method)

    print(results[method].shape)

    for iSweep in range(nSweep):
        ax[iSweep][0].set_ylabel("time="+str(timeIdxs[iSweep]))
        ax[iSweep][iMethod].plot(results[method][iSweep][0])
plt.show()