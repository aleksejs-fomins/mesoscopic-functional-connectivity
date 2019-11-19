import os, sys
import numpy as np
import matplotlib.pyplot as plt

# Export library path
rootname = "mesoscopic-functional-connectivity"
thispath = os.path.dirname(os.path.abspath(__file__))
rootpath = os.path.join(thispath[:thispath.index(rootname)], rootname)
print("Appending project path", rootpath)
sys.path.append(rootpath)

from codes.lib.fc.corr_lib import corr3D
from codes.lib.fc.te_idtxl_wrapper import idtxlParallelCPUMulti
from codes.lib.models.test_lib import dynsys, dynsys_gettrueconn


############################
# Generate Data
############################

# DynSys parameters
dynsysParam = {
    'nNode'   : 12,    # Number of variables
    'nData'   : 4000,  # Number of timesteps
    'nTrial'  : 20,    # Number of trials
    'dt'      : 50,    # ms, timestep
    'tau'     : 500,   # ms, timescale of each mesoscopic area
    'inpT'    : 100,   # Period of input oscillation
    'inpMag'  : 0.0,   # Magnitude of the periodic input
    'std'     : 0.2,   # STD of neuron noise
}

data = dynsys(dynsysParam)   #[trial x channel x time]
print("Generated data of shape", data.shape)

############################
# Compute Functional Conn
############################

# IDTxl parameters
idtxlParam = {
    'dim_order'       : 'rsp',
    'method'          : 'BivariateTE',
    'cmi_estimator'   : 'JidtGaussianCMI',
    'max_lag_sources' : 5,
    'min_lag_sources' : 1
}


nSweep = 10

methods = ['BivariateTE', 'MultivariateTE']
dataSweep = [data.transpose((0, 2, 1))[:, i:i+50, :] for i in range(nSweep)]

results = idtxlParallelCPUMulti(dataSweep, idtxlParam, methods, NCore=4)

fig, ax = plt.subplots(nrows=nSweep, ncols=2)
for iMethod, method in enumerate(methods):
    ax[0][iMethod].set_title(method)
    for iSweep in range(nSweep):
        ax[iSweep][iMethod].imshow(results[method][iSweep][0])
plt.show()