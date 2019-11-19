##############################
#  Includes
##############################

'''


1!!!!!!!!!!!!!!!!!!!!!!!!!!!!1

DANGER: DOES NOT YET COMPARE nTrial < 50

!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

'''


# Standard libraries
import os,sys
import h5py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Append base directory
rootname = "mesoscopic-functional-connectivity"
thispath = os.path.dirname(os.path.abspath(__file__))
rootpath = os.path.join(thispath[:thispath.index(rootname)], rootname)
sys.path.append(rootpath)
print("Appended root directory", rootpath)

# User libraries
from codes.lib.signal_lib import resample
from codes.lib.data_io.os_lib import getfiles_walk
from codes.lib.data_io.qt_wrapper import gui_fpath
from codes.lib.data_io.yaro.yaro_data_read import read_neuro_perf
from codes.lib.fc.te_idtxl_wrapper import idtxlParallelCPUMulti, idtxlResultsParse



##############################
#  Constants
##############################

params = {
    "exp_timestep" : 0.05, # 50ms, the standard measurement interval
    "bin_timestep" : 0.2,  # 200ms, the binned interval

    # The standard timesteps for different scenarios
    "range_CUE" : (1.0, 1.5),    # 1-1.5 seconds trial time
    "range_TEX" : (2.0, 3.5),    # 2-3.5 seconds trial time
    "range_LIK" : (3.5, 6.0),    # 3.5-6 seconds trial time
    "range_ALL" : (0.0, 10.0),   # 0-10  seconds

#     "samples_window" : "ALL",
    "trial_types" : ["iGO", "iNOGO"],
#     "resample"    : {'method' : 'averaging', 'kind' : 'kernel'}  # None if raw data is prefered
    "resample" : None,
}

idtxl_settings = {
    'dim_order'       : 'rsp',
#    'methods'         : ["BivariateMI", "MultivariateMI"],
    'methods'          : ["BivariateTE", "MultivariateTE"],
    'cmi_estimator'   : 'JidtGaussianCMI',
    'min_lag_sources' : 1,
    'max_lag_sources' : 3
}


##############################
#  Paths
##############################
root_path = gui_fpath("Path to root folder containing neuronal data", "./")
pathswalk = getfiles_walk(root_path, ['behaviorvar.mat'])
datapaths = {os.path.basename(path) : path for path, file in pathswalk}
pd.DataFrame(datapaths, index=['dirname']).T

out_path  = gui_fpath("Path where to save results", "./")


##############################
#  Processing
##############################

for iFile, (folderName, folderPathName) in enumerate(datapaths.items()):
    
    if iFile > 0:

        #############################
        # Reading and downsampling
        #############################

        # Read LVM file from command line
        data, behaviour, performance = read_neuro_perf(folderPathName)

        # Get parameters
        nTrials, nTimes, nChannels = data.shape
        print("Loaded neuronal data with (nTrials, nTimes, nChannels)=", data.shape)

        #assert nTimes == 201, "The number of timesteps must be 201 for all data, got "+str(nTimes)
        

        # Timeline (x-axis)
        tlst = params["exp_timestep"] * np.linspace(0, nTimes, nTimes)

        # Downsample data
        if params["resample"] is not None:
            print("Downsampling from", params["exp_timestep"], "ms to", params["bin_timestep"], "ms")
            params["timestep"] = params["bin_timestep"]
            nTimesDownsampled = int(nTimes * params["exp_timestep"] / params["timestep"])
            tlst_down = params["timestep"] * np.linspace(0, nTimesDownsampled, nTimesDownsampled)
            data_down = np.array([[resample(tlst, data[i, :, j], tlst_down, params["resample"])
                                for i in range(nTrials)]
                                for j in range(nChannels)])
            
            # Replace old data with subsampled one
            tlst, data = tlst_down, data_down.transpose(1, 2, 0)
            nTrials, nTimes, nChannels = data.shape
            print("After downsampling data shape is (nTrials, nTimes, nChannels)=", data.shape)
            
        else:
            print("Skip resampling")
            params["timestep"] = params["exp_timestep"]

            
        for trialType in params['trial_types']:
            
            if trialType is None:
                dataEff = data
                fileNameSuffix = ""
            else:
                dataEff = data[behaviour[trialType] - 1]
                fileNameSuffix = "_" + trialType
                print("For trialType =", trialType, "the shape is (nTrials, nTimes, nChannels)=", dataEff.shape)

            #############################
            # Analysis
            #############################

            teWindow = idtxl_settings["max_lag_sources"] + 1
            #teWindow = 2

            data_range = list(range(nTimes - teWindow + 1))
            data_lst = [dataEff[:, i:i + teWindow, :] for i in data_range]
            rez = idtxlParallelCPUMulti(data_lst, idtxl_settings, folderName)

            print(data_range)

            for iMethod, method in enumerate(idtxl_settings['methods']):
                te_data = np.full((3, nChannels, nChannels, nTimes), np.nan)
                
                for iRange in data_range:
                    te_data[..., iRange + idtxl_settings["max_lag_sources"]] = np.array(
                        idtxlResultsParse(rez[iMethod][iRange], nChannels, method=method, storage='matrix')
                    )

                #######################
                # Save results to file
                #######################
                savename = os.path.join(out_path, folderName + fileNameSuffix + '_' + method + '_swipe' + '.h5')
                print(savename)

                h5f = h5py.File(savename, "w")

                grp_rez = h5f.create_group("results")
                grp_rez['TE_table']    = te_data[0]
                grp_rez['delay_table'] = te_data[1]
                grp_rez['p_table']     = te_data[2]

                h5f.close()
