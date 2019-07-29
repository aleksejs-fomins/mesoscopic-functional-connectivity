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
from models.test_lib import noiseLPF, sampleTrials


########################
## Generate input data
########################

param = {
    'N_NODE'      : 12,             # Number of channels 
    'T_TOT'       : 50,             # seconds, Total simulation time
    'TAU_CONV'    : 0.5,            # seconds, Ca indicator decay constant
    'DT'          : 0.001,          # seconds, Neuronal spike timing resolution
    'DT_MACRO'    : 0.2,            # seconds, Binned optical recording resolution
    'STD'         : 1               # Standard deviation of random data
}

src_data2D = noiseLPF(param)
print("Shape before subsampling:", src_data2D.shape)

N_TRIAL = 200                       # Number of trials
N_DATA_TRIAL = 7                    # Number of timesteps per trial (to test max_lag=6)

src_data3D = sampleTrials(src_data2D, N_TRIAL, N_DATA_TRIAL)
print("Shape after subsampling:", src_data3D.shape)

########################
## Save result as HDF5
########################
src_path_h5 = os.path.join(pwd_rez, "sim_noise_lpf_trial_1.h5")
print("Writing source data to", src_path_h5)
src_file_h5 = h5py.File(src_path_h5, "w")
src_file_h5['data'] = src_data3D
src_file_h5.close()
