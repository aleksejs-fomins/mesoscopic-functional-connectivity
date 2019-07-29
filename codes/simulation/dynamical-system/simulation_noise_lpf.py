# Load standard libraries
import os, sys
import numpy as np
import matplotlib.pyplot as plt
import h5py

# Find relative local paths to stuff
path1p = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
path2p = os.path.dirname(path1p)
pwd_lib = os.path.join(path1p, "lib/")
pwd_rez = os.path.join(os.path.join(path2p, "data/"), "sim_ds_h5")

# Set paths
sys.path.append(pwd_lib)

# Load user libraries
from signal_lib import approxDelayConv
from models.test_lib import noiseLPF


########################
## Generate input data
########################

param = {
    'N_NODE'      : 12,             # Number of channels 
    'T_TOT'       : 10,             # seconds, Total simulation time
    'TAU_CONV'    : 0.5,            # seconds, Ca indicator decay constant
    'DT'          : 0.001,          # seconds, Neuronal spike timing resolution
    'DT_MACRO'    : 0.2,            # seconds, Binned optical recording resolution
    'STD'         : 1               # Standard deviation of random data
}

src_data = noiseLPF(param)
print("Resulting shape:", src_data.shape)

########################
## Save result as HDF5
########################
src_path_h5 = os.path.join(pwd_rez, "sim_noise_lpf_1.h5")
print("Writing source data to", src_path_h5)
src_file_h5 = h5py.File(src_path_h5, "w")
src_file_h5['data'] = src_data
src_file_h5.close()

########################
## Plot Data
########################

plt.figure()
for i in range(len(src_data)):
    plt.plot(src_data[i])
    
plt.show()
