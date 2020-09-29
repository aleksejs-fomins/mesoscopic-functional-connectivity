# import standard libraries
import os, sys
import matplotlib.pyplot as plt
import numpy as np

# # Export library path
# rootname = "mesoscopic-functional-connectivity"
# thispath = os.path.dirname(os.path.abspath(__file__))
# rootpath = os.path.join(thispath[:thispath.index(rootname)], rootname)
# print("Appending project path", rootpath)
# sys.path.append(rootpath)

# import special libraries
from lib.models.test_lib import dynsys, dynsys_gettrueconn
from lib.info_metrics.corr_lib import corr_3D

# Set parameters
param = {
    'nNode'   : 12,    # Number of variables
    'nData'   : 4000,  # Number of timesteps
    'nTrial'  : 5,     # Number of trials
    'dt'      : 50,    # ms, timestep
    'tau'     : 500,   # ms, timescale of each mesoscopic area
    'inpT'    : 100,   # Period of input oscillation
    'inpMag'  : 0.0,   # Magnitude of the periodic input
    'std'     : 0.2,   # STD of neuron noise
}

data = dynsys(param)

iTrial = 0

plt.figure()
for iNode in range(param['nTrial']):
    plt.plot(data[iTrial, iNode])
plt.show(block=False)

#[channel x time x trial]
fc, pval = corr_3D(data.transpose((1, 2, 0)), {"dim_order" : "psr"})

plt.figure()
plt.imshow(np.abs(fc), vmin=0, vmax=1)
plt.show()