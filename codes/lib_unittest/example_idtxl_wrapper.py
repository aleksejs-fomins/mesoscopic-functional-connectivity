import os, sys
import numpy as np
import matplotlib.pyplot as plt

# Export library path
rootname = "mesoscopic-functional-connectivity"
thispath = os.path.dirname(os.path.abspath(__file__))
rootpath = os.path.join(thispath[:thispath.index(rootname)], rootname)
print("Appending project path", rootpath)
sys.path.append(rootpath)

from codes.lib.fc.fc_generic import fc_parallel_target
from codes.lib.models.test_lib import dynsys


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
    'cmi_estimator'   : 'JidtGaussianCMI',
    'max_lag_sources' : 5,
    'min_lag_sources' : 1
}

te, lag, p = fc_parallel_target(data.transpose((0, 2, 1))[:, -200:, :], "idtxl", 'BivariateTE', idtxlParam, serial=False, nCore=None)

fig, ax = plt.subplots(ncols=3)
ax[0].imshow(te)
ax[1].imshow(lag)
ax[2].imshow(p, vmin=0, vmax=1)
plt.show()