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
sys.path.append(pwd_lib)
print(pwd_lib)

# Load user libraries
from signal_lib import approxDelayConv
from models.test_lib import noiseLPF, sampleTrials


########################
## Generate input data
########################

param = {
    'N_NODE'      : 12,             # Number of channels 
    'T_TOT'       : 10,             # seconds, Total simulation time
    'TAU_CONV'    : 0.5,            # seconds, Ca indicator decay constant
    'DT_MICRO'    : 0.001,          # seconds, Neuronal spike timing resolution
    'DT'          : 0.2,            # seconds, Binned optical recording resolution
    'STD'         : 1               # Standard deviation of random data
}

N_TRIAL = 400                       # Number of trials
rez_data = np.array([noiseLPF(param).T for i in range(N_TRIAL)])

print("Resulting shape", rez_data.shape)

########################
## Save result as HDF5
########################
rez_path_h5 = os.path.join(pwd_rez, "sim_noise_lpf_trial_1.h5")
print("Writing source data to", rez_path_h5)
rez_file_h5 = h5py.File(rez_path_h5, "w")
rez_file_h5['data'] = rez_data
rez_file_h5.close()
