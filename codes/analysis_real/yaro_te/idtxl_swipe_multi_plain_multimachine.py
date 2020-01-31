##############################
#  Includes
##############################

# Standard libraries
import os, sys
import json
import h5py
import numpy as np

# Append base directory
rootname = "mesoscopic-functional-connectivity"
thispath = os.path.dirname(os.path.abspath(__file__))
rootpath = os.path.join(thispath[:thispath.index(rootname)], rootname)
sys.path.append(rootpath)
print("Appended root directory", rootpath)

# User libraries
from codes.lib.data_io.yaro.yaro_data_read import read_neuro_perf
from codes.lib.signal_lib import downsample_int
from codes.lib.info_metrics.info_metrics_generic import parallel_metric_2d
from codes.lib.sweep_lib import DataSweep


##############################
#  Constants
##############################
#NCore = pathos.multiprocessing.cpu_count()
NCore = int(sys.argv[1])

params = {
    "raw_timestep" : 0.05, # 50ms, the standard measurement interval
    "bin_factor" :   4,    # 200ms, downsample factor

    # The standard timesteps for different scenarios
    "range_CUE" : (1.0, 1.5),    # 1-1.5 seconds trial time
    "range_TEX" : (2.0, 3.5),    # 2-3.5 seconds trial time
    "range_LIK" : (3.5, 6.0),    # 3.5-6 seconds trial time
    "range_ALL" : (0.0, 10.0),   # 0-10  seconds

#     "samples_window" : "ALL",
    "trial_types" : ["iGO", "iNOGO"],
#     "resample"    : {'method' : 'averaging', 'kind' : 'kernel'}  # None if raw data is prefered
    "resample" : None
}

methods = ["BivariateMI", "MultivariateMI"]
#methods =  ["BivariateTE", "MultivariateTE"]

idtxlSettings = {
    'dim_order'       : 'rsp',
    'cmi_estimator'   : 'JidtGaussianCMI',
#    'cmi_estimator'   : 'JidtKraskovCMI',
    'min_lag_sources' : 0,
    'max_lag_sources' : 0
}


##############################
#  Paths
##############################
inPath = "/home/cluster/alfomi/work/mesoscopic-functional-connectivity/codes/analysis_real/yaro_te/"
outPath = "/scratch/alfomi/idtxl_results_kraskov/"
jsonFileName = inPath + "foldersMachine" + str(sys.argv[2]) + ".json"

with open(jsonFileName, 'r') as f:
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
    times = params["raw_timestep"] * np.linspace(0, nTimes, nTimes)

    # Downsample data
    if params["resample"] is not None:
        params["timestep"] = params["raw_timestep"] * params["bin_factor"]
        print("Downsampling from", params["raw_timestep"], "ms to", params["timestep"], "ms")

        # Resample data
        timesDown, dataDown = downsample_int(times, data.transpose((1,0,2)), params["bin_factor"])

        # Replace old data with subsampled one
        times, data = timesDown, dataDown.transpose((1, 0, 2))
        nTrials, nTimes, nChannels = data.shape
        print("After downsampling data shape is (nTrials, nTimes, nChannels)=", data.shape)

    else:
        print("Skip resampling")
        params["timestep"] = params["raw_timestep"]


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
            sweepSettings = {
                "window"    : idtxlSettings["max_lag_sources"] + 1, # 2
                "dim_order" : idtxlSettings["dim_order"]
            }
            dataSweep = DataSweep(data, sweepSettings)

            rez = parallel_metric_2d(dataSweep.iterator(), 'idtxl', methods, idtxlSettings, nCore=NCore)
            timesSweep = times[dataSweep.get_target_time_idxs()]
            nTimesEff = len(timesSweep)

            for methodName, methodRez in rez.items():
                te, lag, p = methodRez.transpose((1, 2, 3, 0))

                #######################
                # Save results to file
                #######################
                savename = os.path.join(outPath, folderName + fileNameSuffix + '_' + methodName + '_swipe' + '.h5')
                print(savename)

                h5f = h5py.File(savename, "w")

                grp_rez = h5f.create_group("results")
                grp_rez['timestart']   = timesSweep[0]
                grp_rez['timestep']    = params["timestep"]
                grp_rez['TE_table']    = te
                grp_rez['delay_table'] = lag
                grp_rez['p_table']     = p

                h5f.close()
