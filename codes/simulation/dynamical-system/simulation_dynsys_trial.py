# Load standard libraries
import os, sys
import numpy as np
import matplotlib.pyplot as plt
import h5py

# Find relative local paths to stuff
path1p = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
path2p = os.path.dirname(path1p)
path3p = os.path.dirname(path2p)
pwd_lib = os.path.join(path2p, "lib/")
pwd_rez = os.path.join(path3p, "data/")

# Set paths
print(pwd_lib)
sys.path.append(pwd_lib)

# Local libraries
from models.dyn_sys import DynSys
from models.test_lib import sampleTrials


# Set parameters
param = {
    'ALPHA'   : 0.1,  # 1-connectivity strength
    'N_NODE'  : 12,   # Number of variables
    'N_DATA'  : 50,   # Number of timesteps
    'MAG'     : 0,    # Magnitude of input
    'T'       : 20,   # Period of input oscillation
    'STD'     : 0.2   # STD of neuron noise
}

# Create dynamical system
DS1 = DynSys(param)

# Sample trials from data
N_TRIAL = 400                       # Number of trials
rez_data = np.array([DS1.compute().T for i in range(N_TRIAL)])
print("Final shape", rez_data.shape)    

########################
## Save result as HDF5
########################
res_path_h5 = os.path.join(pwd_rez, "sim_dynsys_trial_2.h5")
print("Writing source data to", res_path_h5)
rez_file_h5 = h5py.File(res_path_h5, "w")
rez_file_h5['data'] = rez_data
rez_file_h5.close()

########################
## Plot result for checking
########################
plt.figure()
for trial in range(0, 5):
    plt.plot(rez_data[trial, :, 0])
plt.show()
