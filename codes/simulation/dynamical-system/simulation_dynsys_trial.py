# Export library path
import os, sys
thispath = os.path.dirname(os.path.abspath(__file__))
p1path = os.path.abspath(os.path.join(thispath, os.pardir))
p2path = os.path.abspath(os.path.join(p1path, os.pardir))
sys.path.append(os.path.join(p1path, 'lib/'))

# Locate results path
rezPath = os.path.join(os.path.join(p2path, 'data/'), 'sim_ds_h5')

# Standard libraries
import numpy as np
import matplotlib.pyplot as plt
import h5py

# Local libraries
from models.dyn_sys import DynSys
from models.test_lib import sampleTrials


# Set parameters
param = {
    'ALPHA'   : 0.1,  # 1-connectivity strength
    'N_NODE'  : 12,   # Number of variables
    'N_DATA'  : 4000, # Number of timesteps
    'MAG'     : 0,    # Magnitude of input
    'T'       : 20,   # Period of input oscillation
    'STD'     : 0.2   # STD of neuron noise
}

# Create dynamical system
DS1 = DynSys(param)

# Sample trials from data
N_TRIAL = 1000                       # Number of trials
N_DATA_TRIAL = 7                    # Number of timesteps per trial (to test max_lag=6)
data3D = sampleTrials(DS1.data, N_TRIAL, N_DATA_TRIAL)

########################
## Save result as HDF5
########################
src_path_h5 = os.path.join(rezPath, "sim_dynsys_trial_2.h5")
print("Writing source data to", src_path_h5)
src_file_h5 = h5py.File(src_path_h5, "w")
src_file_h5['data'] = data3D
src_file_h5.close()

########################
## Plot result for checking
########################
plt.figure()
for trial in range(0, 5):
    plt.plot(data3D[trial, :, 0])
plt.show()
