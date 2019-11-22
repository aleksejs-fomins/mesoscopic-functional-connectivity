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
from codes.lib.signal_lib import resample
from codes.lib.fc.idtxl_wrapper import idtxlParallelCPUMulti


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

#methods = ["BivariateMI", "MultivariateMI"]
methods =  ["BivariateTE", "MultivariateTE"]

idtxl_settings = {
    'dim_order'       : 'rsp',
#    'cmi_estimator'   : 'JidtGaussianCMI',
    'cmi_estimator'   : 'JidtKraskovCMI',
    'min_lag_sources' : 1,
    'max_lag_sources' : 3
}


##############################
#  Paths
##############################
in_path = "/home/cluster/alfomi/work/mesoscopic-functional-connectivity/codes/analysis_real/yaro_te/"
out_path = "/scratch/alfomi/idtxl_results_kraskov/"
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
            rez = idtxlParallelCPUMulti(data_lst, idtxl_settings, methods, NCore=NCore)  # {method : [nRange, 3, nChannel, nChannel] }

            for methodName, methodRez in rez.items():
                te_data = np.full((3, nChannels, nChannels, nTimes), np.nan)
                te_data[..., idtxl_settings["max_lag_sources"]:] = methodRez.transpose((1,2,3,0))

                #######################
                # Save results to file
                #######################
                savename = os.path.join(out_path, folderName + fileNameSuffix + '_' + methodName + '_swipe' + '.h5')
                print(savename)

                h5f = h5py.File(savename, "w")

                grp_rez = h5f.create_group("results")
                grp_rez['TE_table']    = te_data[0]
                grp_rez['delay_table'] = te_data[1]
                grp_rez['p_table']     = te_data[2]

                h5f.close()
