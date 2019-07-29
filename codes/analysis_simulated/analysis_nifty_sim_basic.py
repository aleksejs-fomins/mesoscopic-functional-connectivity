# Load standard libraries
import os, sys, subprocess
import numpy as np
import matplotlib.pyplot as plt
import scipy.io
import h5py

# Find relative local paths to stuff
path1p = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
path2p = os.path.dirname(path1p)
pwd_lib = os.path.join(path1p, "lib/")
pwd_mat = os.path.join(os.path.join(path2p, "data/"), "sim_ds_mat")
pwd_h5 = os.path.join(os.path.join(path2p, "data/"), "sim_ds_h5")

# Set paths
sys.path.append(pwd_lib)

# Load user libraries
from signal_lib import approxDelayConv
from nifty_wrapper.nifty_wrapper import nifty_wrapper
from plots.plt_imshow_mat import plotImshowMat


fname_lst = [
    #'sim_noise_pure_trial_1.h5',
    #'sim_noise_lpf_trial_1.h5',
    #'sim_cycle_trial_1.h5',
    'sim_dynsys_trial_2.h5'
]

for filename in fname_lst:
    
    #######################
    # Run NIFTY wrapper
    #######################
    rez_path_h5 = nifty_wrapper(filename, pwd_h5, pwd_mat)
    #rez_path_h5 = os.path.join(pwd_h5, 'result_' + filename.split('.')[0] + '.h5')

    #######################
    # Load NIFTY result from HDF5
    #######################
    DELAY_MIN = 1
    DELAY_MAX = 6
    
    rez_file_h5 = h5py.File(rez_path_h5, "r")
    rez_data = rez_file_h5['results']
    
    TE_mat_3D  = np.copy(rez_data['TE_table'])
    Lag_mat_2D = np.copy(rez_data['delay_table']).astype(float)
    P_mat_3D   = np.copy(rez_data['p_table'])
    
    #######################
    # Extract significant connections
    #######################
    
    N_NODE, N_NODE_Y, N_T = P_mat_3D.shape
    
    # 1) Set all diagonal entries to NAN
    P_mat_3D[P_mat_3D == 0] = 100
    
    # 2) Set all entries with p > 1% to NAN    
    idxNanConn = np.array([[[np.isnan(P_mat_3D[i, j, k]) or P_mat_3D[i,j,k] > 0.01 for k in range(N_T)] for j in range(N_NODE)] for i in range(N_NODE)])
    
    TE_mat_3D[idxNanConn]  = np.nan
    P_mat_3D[idxNanConn]   = np.nan
    
    # 3) For each pair, find how many times there is a non-NAN connection
    numNotNAN = N_T - np.nansum(idxNanConn, axis = 2)
    
    # 4) For each pair, select connection with strongest TE, out of those that are not NAN
    TE_mat_2D = np.zeros((N_NODE, N_NODE)) + np.nan
    P_mat_2D = np.zeros((N_NODE, N_NODE)) + np.nan
    
    for i in range(N_NODE):
        for j in range(N_NODE):
            maxThis = np.nanmax(TE_mat_3D[i, j])
            if not np.isnan(maxThis):
                TE_mat_2D[i, j] = maxThis
                P_mat_2D[i, j]   = P_mat_3D[i,j,np.nanargmax(TE_mat_3D[i,j])]
            else:
                Lag_mat_2D[i, j] = np.nan

    #######################
    # Plot Results
    #######################

    plotImshowMat(
        [TE_mat_2D, Lag_mat_2D, P_mat_2D, numNotNAN],
        ['NiftyTE', "Delays", "p-value", "count"],
        'NIfTy analysis of ' + filename,
        (1, 4),
        lims = [[0, 1], [0, DELAY_MAX], [0, 1], [0, N_NODE]],
        draw = True,
        savename = filename.split('.')[0] + '.png'
    )


plt.show()
