##############################
#  Includes
##############################

# Standard libraries
import os,sys
import json
import h5py
import copy
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Append base directory
rootname = "mesoscopic-functional-connectivity"
thispath = os.path.dirname(os.path.abspath(__file__))
rootpath = os.path.join(thispath[:thispath.index(rootname)], rootname)
codepath = os.path.join(rootpath, "code_python")
sys.path.append(codepath)
print("Appended root directory", codepath)

# User libraries
from lib.data_io.os_lib import getfiles_walk
from lib.data_io.data_read import read_neuro_perf
from lib.data_io.qt_wrapper import gui_fpath, gui_fname


##############################
# Step 1: Get all data files
##############################
thisFolder = './'
allFolders = []
while thisFolder != '':
    thisFolder = gui_fpath("Root folder for data files", thisFolder)
    if thisFolder != '':
        folders = getfiles_walk(thisFolder, ['behaviorvar'])[:, 0]
        allFolders += list(folders)

print("Total Folders Found", len(allFolders))

params = {
    "trial_types" : ["iGO", "iNOGO"]
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
#  Processing
##############################

for iFile, folderPathName in enumerate(allFolders):
    folderName = os.path.basename(folderPathName)
    #############################
    # Reading and downsampling
    #############################

    # Read LVM file from command line
    data, behaviour, performance = read_neuro_perf(folderPathName)

    # Get parameters
    nTrials, nTimes, nChannels = data.shape
    #print("Loaded neuronal data with (nTrials, nTimes, nChannels)=", data.shape)

    if nTimes != 201:
        print("--Warning, number of timesteps: ", nTimes)
        if nTimes > 201:
            nTimes = 201
            data = data[:, :nTimes, :]
            print("---Cropped down to", nTimes)
        

    for trialType in params['trial_types']:

        if trialType is None:
            dataEff = data
            fileNameSuffix = ""
        else:
            dataEff = data[np.array(behaviour[trialType], dtype=int) - 1]
            fileNameSuffix = "_" + trialType
            #print("For trialType =", trialType, "the shape is (nTrials, nTimes, nChannels)=", dataEff.shape)

        #############################
        # Analysis
        #############################

        teWindow = idtxl_settings["max_lag_sources"] + 1

        data_range = list(range(nTimes - teWindow + 1))
        data_lst = np.array([dataEff[:, i:i + teWindow, :] for i in data_range])
        
        if dataEff.shape[0] < 50:
            print("--Warning: Shape of sweep data:", data_lst.shape)
            
        if np.sum(np.isnan(dataEff)) > 0:
            print("--Warning: Data has", np.sum(np.isnan(dataEff)), "NANs")

        # for iMethod, method in enumerate(idtxl_settings['methods']):
        #     te_data = np.full((3, nChannels, nChannels, nTimes), np.nan)

#             #######################
#             # Save results to file
#             #######################
#             savename = os.path.join(out_path, folderName + fileNameSuffix + '_' + method + '_swipe' + '.h5')
#             print(savename)

#             h5f = h5py.File(savename, "w")

#             grp_rez = h5f.create_group("results")
#             grp_rez['TE_table']    = te_data[0]
#             grp_rez['delay_table'] = te_data[1]
#             grp_rez['p_table']     = te_data[2]

#             h5f.close()
