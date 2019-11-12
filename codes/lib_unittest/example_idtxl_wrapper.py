import numpy as np
import matplotlib.pyplot as plt

# Export library path
import sys
from os.path import dirname, abspath, join
thispath   = dirname(abspath(__file__))
parentpath = dirname(thispath)
rootpath    = join(parentpath, 'lib')
sys.path.append(rootpath)

from codes.lib.fc.corr_lib import corr3D
from codes.lib.fc.te_idtxl_wrapper import idtxlParallelCPU
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


te, lag, p = idtxlParallelCPU(data.transpose((0, 2, 1))[:, -200:, :], idtxlParam, NCore=4)

fig, ax = plt.subplots(ncols=3)
ax[0].imshow(te)
ax[1].imshow(lag)
ax[2].imshow(p, vmin=0, vmax=1)
plt.show()