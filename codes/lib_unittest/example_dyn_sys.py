# import standard libraries
import sys
from os.path import dirname, abspath, join
import matplotlib.pyplot as plt
import numpy as np

# Export library path
thispath   = dirname(abspath(__file__))
parentpath = dirname(thispath)
rootpath    = join(parentpath, 'lib')
sys.path.append(rootpath)

# import special libraries
from codes.lib.models.test_lib import dynsys, dynsys_gettrueconn
from codes.lib.fc.corr_lib import corr3D

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
plt.show()

#[channel x time x trial]
fc, pval = corr3D(data.transpose((1, 2, 0)))

plt.figure()
plt.imshow(np.abs(fc), vmin=0, vmax=1)
plt.show()