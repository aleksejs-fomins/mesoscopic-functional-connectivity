##############################
#  Includes
##############################

# Standard libraries
import os,sys
import json
import h5py
import copy
import pathos
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from time import gmtime, strftime
from pkg_resources import resource_filename

# Append base directory
rootname = "mesoscopic-functional-connectivity"
thispath = os.path.dirname(os.path.abspath(__file__))
rootpath = os.path.join(thispath[:thispath.index(rootname)], rootname)
sys.path.append(rootpath)
print("Appended root directory", rootpath)

# IDTxl libraries
from idtxl.data import Data
import jpype as jp

# User libraries
from codes.lib.data_io.os_lib import getfiles_walk
from codes.lib.data_io.qt_wrapper import gui_fpath, gui_fname
from codes.lib.data_io.yaro.yaro_data_read import read_neuro_perf
from codes.lib.signal_lib import resample



def idtxlParallelCPUMulti(dataLst, settings, taskName, NCore=None):
    '''
    Performs parameter sweep over methods, data sets and channels, distributing work equally among available processes
    * Number of processes (aka channels) must be equal for all datasets
    '''

    ##########################################
    # Determine parameters for the parameter sweep
    ##########################################
    idxProcesses = settings['dim_order'].index("p")  # Index of Processes dimension in data

    nMethods = len(settings['methods'])
    nDataSets = len(dataLst)
    nProcesses = dataLst[0].shape[idxProcesses]

    mIdxs = np.arange(nMethods)  # Indices of all methods
    dIdxs = np.arange(nDataSets)  # Indices of all data sets
    pIdxs = np.arange(nProcesses)  # Indices of all processes (aka data channels)

    sweepLst = [(m, d, p) for m in mIdxs for d in dIdxs for p in pIdxs]
    sweepIdxs = np.arange(len(sweepLst))

    ###############################
    # Convert data to ITDxl format
    ###############################
    dataIDTxl_lst = [Data(d, dim_order=settings['dim_order']) for d in dataLst]

    ###############################
    # Initialize multiprocessing pool
    ###############################
    if NCore is None:
        NCore = pathos.multiprocessing.cpu_count() - 1
    pool = pathos.multiprocessing.ProcessingPool(NCore)
    # pool = multiprocessing.Pool(NCore)

    ###############################
    # Compute estimators in parallel
    ###############################
    def multiParallelTask(sw):
        #print(sw)

        """Start JAVA virtual machine if it is not running."""
        jar_location = resource_filename(__name__, 'infodynamics.jar')
        with open("yololog.txt", "a+") as f:
            if not jp.isJVMStarted():
                jp.startJVM(jp.getDefaultJVMPath(), '-ea', ('-Djava.class.path=' + jar_location))
                f.write(strftime("[%Y.%m.%d %H:%M:%S]", gmtime()) + "Started new JVM\n")
            else:
                f.write(strftime("[%Y.%m.%d %H:%M:%S]", gmtime()) + "JVM already exists\n")

        return [1]

    return pool.map(multiParallelTask, sweepLst)



##############################
#  Constants
##############################
#NCore = pathos.multiprocessing.cpu_count()
NCore = int(sys.argv[1])

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
#    'cmi_estimator'   : 'JidtGaussianCMI',
    'cmi_estimator'   : 'JidtKraskovCMI',
    'min_lag_sources' : 1,
    'max_lag_sources' : 3
}


##############################
#  Paths
##############################
#in_path = "/home/cluster/alfomi/work/mesoscopic-functional-connectivity/codes/analysis_real/yaro_te/"
#out_path = "/scratch/alfomi/idtxl_results_kraskov/"
in_path = "./"
out_path = "../../tmp"
json_fname = in_path + "foldersMachine" + str(sys.argv[2]) + ".json"

with open(json_fname, 'r') as f:
    datapaths = json.load(f)['dataFolders']

##############################
#  Processing
##############################

for iFile, folderPathName in enumerate(datapaths):
    folderName = os.path.basename(folderPathName)
    #############################
    # Reading and downsampling
    #############################

    # Read LVM file from command line
    data, behaviour, performance = read_neuro_perf(folderPathName)

    # Get parameters
    nTrials, nTimes, nChannels = data.shape
    print("Loaded neuronal data with (nTrials, nTimes, nChannels)=", data.shape)

    if nTimes != 201:
        print("--Warning, number of timesteps: ", nTimes)
        if nTimes > 201:
            nTimes = 201
            data = data[:, :nTimes, :]
            print("---Cropped down to", nTimes)


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
            dataEff = data[np.array(behaviour[trialType], dtype=int) - 1]
            fileNameSuffix = "_" + trialType
            print("For trialType =", trialType, "the shape is (nTrials, nTimes, nChannels)=", dataEff.shape)

        #############################
        # Analysis
        #############################
        
        if dataEff.shape[0] < 50:
            print("Number of trials", dataEff.shape[0], "below threshold, skipping analysis")
        else:
            teWindow = idtxl_settings["max_lag_sources"] + 1

            data_range = list(range(nTimes - teWindow + 1))
            data_lst = [dataEff[:, i:i + teWindow, :] for i in data_range]
            rez = idtxlParallelCPUMulti(data_lst, idtxl_settings, folderName, NCore=NCore)

            # for iMethod, method in enumerate(idtxl_settings['methods']):
            #     te_data = np.full((3, nChannels, nChannels, nTimes), np.nan)
            #
            #     # for iRange in data_range:
            #     #     te_data[..., iRange + idtxl_settings["max_lag_sources"]] = np.array(
            #     #         idtxlResultsParse(rez[iMethod][iRange], nChannels, method=method, storage='matrix')
            #     #     )
            #
            #     #######################
            #     # Save results to file
            #     #######################
            #     savename = os.path.join(out_path, folderName + fileNameSuffix + '_' + method + '_swipe' + '.h5')
            #     print(savename)
            #
            #     h5f = h5py.File(savename, "w")
            #
            #     grp_rez = h5f.create_group("results")
            #     grp_rez['TE_table']    = te_data[0]
            #     grp_rez['delay_table'] = te_data[1]
            #     grp_rez['p_table']     = te_data[2]
            #
            #     h5f.close()
