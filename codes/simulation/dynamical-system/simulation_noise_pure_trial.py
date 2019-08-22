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

# Load user libraries
from signal_lib import approxDelayConv
from models.test_lib import noisePure


########################
## Generate input data
########################

param = {
    'N_NODE'      : 12,             # Number of channels 
    'T_TOT'       : 10,             # seconds, Total simulation time
    'DT'          : 0.2,            # seconds, Neuronal spike timing resolution
    'STD'         : 1               # Standard deviation of random data
}

N_TRIAL = 400                       # Number of trials
rez_data = np.array([noisePure(param).T for i in range(N_TRIAL)])

print("Resulting shape", rez_data.shape)

########################
## Save result as HDF5
########################
res_path_h5 = os.path.join(pwd_rez, "sim_noise_pure_trial_1.h5")
print("Writing source data to", res_path_h5)
rez_file_h5 = h5py.File(res_path_h5, "w")
rez_file_h5['data'] = rez_data
rez_file_h5.close()

# ########################
# ## Plot Data
# ########################

# plt.figure()
# for i in range(param['N_NODE']):
#     plt.plot(src_data3D[0, :, i])
    
# plt.show()
