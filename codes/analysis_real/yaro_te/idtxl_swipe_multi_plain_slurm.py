##############################
#  Includes
##############################

# Standard libraries
import os,sys
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
from codes.lib.aux_functions import mem_now_as_str
from codes.lib.data_io.yaro.yaro_data_read import read_neuro_perf
from codes.lib.fc.idtxl_wrapper import analyse_single_target

##############################
#  Paths
##############################
in_path = "/home/cluster/alfomi/work/mesoscopic-functional-connectivity/codes/analysis_real/yaro_te/"
out_path = "/scratch/alfomi/idtxl_results_kraskov/"
json_fname = os.path.join(in_path, "slurmtasks.json")

##############################
#  Tasks
##############################
thisTaskIdx = int(sys.argv[1])

with open(json_fname, 'r') as f:
    tasks = json.load(f)

folderPathName, window, minlag, maxlag, trialType, sweep, method, iTrg = tasks[thisTaskIdx]

#############################
# Reading data and selecting
#############################

# Read LVM file from command line
data, behaviour, performance = read_neuro_perf(folderPathName)
dataEff = data[np.array(behaviour[trialType], dtype=int) - 1]
dataSweep = dataEff[:, sweep:sweep + window, :]

#############################
# Analysis
#############################

idtxlSettings = {
    'dim_order'       : 'rsp',
#    'cmi_estimator'   : 'JidtGaussianCMI',
    'cmi_estimator'   : 'JidtKraskovCMI',
    'min_lag_sources' : minlag,
    'max_lag_sources' : maxlag
}

# Returns [3 x nSource]
rez = analyse_single_target(iTrg, dataSweep, method, idtxlSettings)

#######################
# Save results to file
#######################
srcNameBare = os.path.splitext(os.path.basename(folderPathName))[0]
outNameBare = "_".join([srcNameBare, "swipe", trialType, str(sweep), method, str(iTrg)])

outPathName = os.path.join(out_path, outNameBare + '.h5')

print("writing file to", outPathName)

with h5py.File(outPathName, "w") as h5f:
    h5f["data"] = rez
