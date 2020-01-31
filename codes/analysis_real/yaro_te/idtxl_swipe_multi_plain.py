##############################
#  Includes
##############################


# Standard libraries
import os,sys
import h5py
import numpy as np

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
from codes.lib.info_metrics.info_metrics_generic import parallel_metric_2d



##############################
#  Constants
##############################
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
    "resample" : None
}

methods = ["BivariateMI", "MultivariateMI"]
#methods = ["BivariateTE", "MultivariateTE"]

idtxlSettings = {
    'dim_order'       : 'rsp',
    'cmi_estimator'   : 'JidtGaussianCMI',
    'min_lag_sources' : 0,
    'max_lag_sources' : 0
}


##############################
#  Paths
##############################
rootPath = gui_fpath("Path to root folder containing neuronal data", "./")
pathswalk = getfiles_walk(rootPath, ['behaviorvar.mat'])
datapaths = {os.path.basename(path) : path for path, file in pathswalk}
outPath = gui_fpath("Path where to save results", "./")


##############################
#  Processing
##############################

for iFile, (folderName, folderPathName) in enumerate(datapaths.items()):
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
        tlst, data = tlst_down, data_down.transpose((1, 2, 0))
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

        if dataEff.shape[0] < 50:
            print("Number of trials", dataEff.shape[0], "below threshold, skipping analysis")
        else:
            teWindow = idtxlSettings["max_lag_sources"] + 1
            #teWindow = 2

            data_range = list(range(nTimes - teWindow + 1))
            data_lst = [dataEff[:, i:i + teWindow, :] for i in data_range]
            rez = parallel_metric_2d(data_lst, 'idtxl', methods, idtxlSettings, nCore=NCore)


            for methodName, methodRez in rez.items():
                te_data = np.full((3, nChannels, nChannels, nTimes), np.nan)
                te_data[..., idtxlSettings["max_lag_sources"]:] = methodRez.transpose((1,2,3,0))

                #######################
                # Save results to file
                #######################
                savename = os.path.join(outPath, folderName + fileNameSuffix + '_' + methodName + '_swipe' + '.h5')
                print(savename)

                h5f = h5py.File(savename, "w")

                grp_rez = h5f.create_group("results")
                grp_rez['TE_table']    = te_data[0]
                grp_rez['delay_table'] = te_data[1]
                grp_rez['p_table']     = te_data[2]

                h5f.close()
