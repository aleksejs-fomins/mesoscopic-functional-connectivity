# Standard libraries
import os,sys,inspect
import h5py
import copy
import numpy as np
import pathos

# Export library path
rootname = "mesoscopic-functional-connectivity"
thispath   = os.path.dirname(os.path.abspath(__file__))
rootpath = os.path.join(thispath[:thispath.index(rootname)], rootname)
print("Appending project path", rootpath)
sys.path.append(rootpath)

# User libraries
from codes.lib.models.test_lib import noisePure, noiseLPF, dynsys, dynsys_gettrueconn
from codes.lib.data_io.qt_wrapper import gui_fpath

#############################
# Model parameters
#############################

# Introduce extra zealous factor, to return dynamics only after system has stabilized
def dynsys_zealous(param):
    zealous_factor = 10  #seconds
    param_tmp = copy.deepcopy(param)
    param_tmp['nData'] += zealous_factor
    return dynsys(param_tmp)[:, zealous_factor:]


def get_param_pure_noise(nTrial, nNode, nData, dt):
    return  {
        'nTrial'      : nTrial,         # Number of trials
        'nNode'       : nNode,          # Number of channels
        'tTot'        : nData*dt,       # seconds, Total simulation time
        'dt'          : dt,             # seconds, Binned optical recording resolution
        'std'         : 1               # Standard deviation of random data
    }

def get_param_lpf_sub(nTrial, nNode, nData, dt):
    return {
        'nTrial'      : nTrial,  # Number of trials
        'nNode'       : nNode,         # Number of channels
        'tTot'        : nData*dt,      # seconds, Total simulation time
        'tauConv'     : 0.5,           # seconds, Ca indicator decay constant
        'dtMicro'     : 0.001,         # seconds, Neuronal spike timing resolution
        'dt'          : dt,            # seconds, Binned optical recording resolution
        'std'         : 1              # Standard deviation of random data
    }

def get_param_dyn_sys(nTrial, nNode, nData, dt):
    return {
        'nTrial'  : nTrial,  # Number of trials
        'dt'      : dt,      # seconds, Binned optical recording resolution
        'tau'     : 0.2,     # seconds, Neuronal population timescale
        'nNode'   : nNode,   # Number of variables
        'nData'   : nData,   # Number of timesteps
        'inpMag'  : 0.0,     # Magnitude of input to first node
        'inpT'    : 20,      # Period of input oscillation
        'std'     : 0.2      # STD of neuron noise
    }

# Find model function by name
modelFuncDict = {
    'purenoise'    : noisePure,
    'lpfsubnoise'  : noiseLPF,
    'dynsys'       : dynsys_zealous,
}

# Group methods to extract standard parameters
modelParamFuncDict = {
    "purenoise"   : get_param_pure_noise,
    "lpfsubnoise" : get_param_lpf_sub,
    "dynsys"      : get_param_dyn_sys
}

# True connectivity matrices
modelTrueConnDict = {
    "purenoise"   : lambda p : np.full((p['nNode'], p['nNode']), np.nan),
    "lpfsubnoise" : lambda p : np.full((p['nNode'], p['nNode']), np.nan),
    "dynsys"      : dynsys_gettrueconn
}

#############################
# Generate Data, save to h5
#############################

# A function that generates data and saves it to h5
def processTask(task):
    # Extract parameters
    print("--Doing task", task)

    testType, outpath, nNode, nData, nTrial, dt, modelName = task
    paramThis = modelParamFuncDict[modelName](nTrial, nNode, nData, dt)
    trueConn = modelTrueConnDict[modelName](paramThis)
    modelFunc = modelFuncDict[modelName]

    # Compute results
    dataThis = modelFunc(paramThis).transpose((0, 2, 1))

    # Save to h5 file
    outname = os.path.join(outpath, testType + "_" + modelName + "_" + str(nTrial) + "_" + str(nNode) + "_" + str(nData) + ".h5" )

    with h5py.File(outname, "w") as h5f:
        grp_rez = h5f.create_group("results")
        grp_rez['modelName']   = modelName
        grp_rez['connTrue']    = trueConn
        grp_rez['data']        = dataThis

    print("--Finished task", task)

    return True



outpath = gui_fpath("Select output path", "./")
nNode = 12
dt = 0.05   # 50ms, Yaro non-downsampled temporal resolution
nStep = 40  # Number of different data sizes to pick
tStep = 2   # Minimal number of time steps
nDataLst = (2 * 10 ** (np.linspace(1.6, 2.9, nStep))).astype(int)

taskList = []

# Generate task list
for nDataRow in nDataLst:
    for modelName in modelFuncDict.keys(): 
        # Width analysis
        nData = nDataRow * tStep
        nTrial = 1
        taskList += [("width", outpath, nNode, nData, nTrial, dt, modelName)]

        # Depth analysis
        nData = tStep
        nTrial = nDataRow
        taskList += [("depth", outpath, nNode, nData, nTrial, dt, modelName)]

# Compute all tasks in parallel
nCore = pathos.multiprocessing.cpu_count() - 1
print("Using nCores", nCore)
pool = pathos.multiprocessing.ProcessingPool(nCore)
rez_multilst = pool.map(processTask, taskList)