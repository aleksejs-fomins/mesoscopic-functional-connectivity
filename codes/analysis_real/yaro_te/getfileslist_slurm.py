'''
  Plan:
  1. Get all files:
  2. For each file, on 1 core do 1 target, all sweeps
'''

###############################
#  Libraries
###############################

# Standard libraries
import os,sys
import json
import numpy as np

# Append base directory
rootname = "mesoscopic-functional-connectivity"
thispath = os.path.dirname(os.path.abspath(__file__))
rootpath = os.path.join(thispath[:thispath.index(rootname)], rootname)
sys.path.append(rootpath)
print("Appended root directory", rootpath)

# User libraries
from codes.lib.data_io.os_lib import getfiles_walk
from codes.lib.data_io.qt_wrapper import gui_fpath, gui_fname
from codes.lib.data_io.yaro.yaro_data_read import read_neuro_perf


###############################
#  Select all files to process
###############################

# Step 1: Get all data files
thisFolder = './'
allFolders = []
while thisFolder != '':
    thisFolder = gui_fpath("Root folder for data files", thisFolder)
    if thisFolder != '':
        folders = getfiles_walk(thisFolder, ['behaviorvar'])[:, 0]
        allFolders += list(folders)

print("Total Folders Found", len(allFolders))

###############################
#  Potentially exclude some files
###############################

excl_json = gui_fname("JSON file with folders to exclude", "./", "JSON (*.json)")
if excl_json != '':
    # Exclude some data that is already done or bad
    with open('excl_folders.json', 'r') as f:
       excl_folders = json.load(f)['excl_folders']

    allFoldersFiltered = []
    for folder in allFolders:
      if not np.any([excl in folder for excl in excl_folders]):
         allFoldersFiltered += [folder]
      else:
         print("Excluded folder", folder)
    print("Total Folders after exclusion", len(allFoldersFiltered))
    allFolders = allFoldersFiltered


###############################
#  Construct all tasks
###############################

params = {
    #"methods"    : ["BivariateMI", "MultivariateMI"],
    "methods"     :  ["BivariateTE", "MultivariateTE"],
    "trial_types" : ["iGO", "iNOGO"],
    "window"      : 4,
    'min_lag_sources': 1,
    'max_lag_sources': 3
}

taskIdx2task = []
for fpath in allFolders:
    data, behaviour, performance = read_neuro_perf(fpath)

    # Get parameters
    nTrials, nTimes, nChannels = data.shape
    print("Processing file (nTrials, nTimes, nChannels)=", data.shape)

    # Crop nTimes
    nTimes = np.min([201, nTimes])
    sweepRange = np.arange(nTimes - params["window"] + 1)

    for trialType in params['trial_types']:
        nTrialsEff = len(behaviour[trialType])

        if nTrialsEff < 50:
            print("-- Skipping", trialType, "because nTrials=", nTrialsEff)
        else:
            for sweep in sweepRange:
                for method in params["methods"]:
                    for iTrg in range(nChannels):
                        taskIdx2task += [(
                            fpath,
                            params["window"],
                            params["min_lag_sources"],
                            params["max_lag_sources"],
                            trialType,
                            int(sweep),
                            method,
                            iTrg)]


###############################
#  Construct all tasks
###############################

outfname = 'slurmtasks.txt'
print("Writing", len(taskIdx2task), "tasks to", outfname)
with open(outfname, 'w') as f:
    for task in taskIdx2task:
        f.write(",".join(str(el) for el in task))
