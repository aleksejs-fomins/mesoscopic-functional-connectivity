import os, sys
from datetime import datetime
import numpy as np
import scipy.io

from codes.lib.data_io.os_lib import get_subfolders
from codes.lib.data_io.matlab_lib import loadmat, matstruct2dict
from codes.lib.aux_functions import merge_dicts

# Read data and behaviour matlab files given containing folder
def read_neuro_perf(folderpath, verbose=True):
    # Read MAT file from command line
    if verbose:
        print("Reading Yaro data from", folderpath)
    fname_data        = os.path.join(folderpath, "data.mat")
    fname_behaviour   = os.path.join(folderpath, "behaviorvar.mat")
    fpath_performance = os.path.join(folderpath, "Transfer Entropy")
    
    waitRetry = 3  # Seconds wait before trying to reload the file if it is not accessible
    data        = loadmat(fname_data, waitRetry=waitRetry)['data']
    behavior    = loadmat(fname_behaviour, waitRetry=waitRetry)
    
    if not os.path.exists(fpath_performance):
        print("--Warning: No performance metrics found for", os.path.dirname(folderpath), "returning, None")
        performance = None
    else:
        fname_performance = os.path.join(fpath_performance, "performance.mat")
        performance = loadmat(fname_performance, waitRetry=waitRetry)['performance']
    
    # Get rid of useless fields in behaviour
    behavior = {k : v for k, v in behavior.items() if k[0] != '_'}
    
    # Convert trials structure to a dictionary
    behavior['trials'] = merge_dicts([matstruct2dict(obj) for obj in behavior['trials']])
    
    # d_trials = matstruct2dict(behavior['trials'][0])
    # for i in range(1, len(behavior['trials'])):
    #     d_trials = merge_dict(d_trials, matstruct2dict(behavior['trials'][i]))
    # behavior['trials'] = d_trials
    
    # CONSISTENCY TEST 1 - If behavioural trials are more than neuronal, crop:
    dataNTrials = data.shape[0]
    behavToArray = lambda b: np.array([b], dtype=int) if type(b)==int else b   # If array has 1 index it appears as a number :(
    behKeys = ['iGO', 'iNOGO', 'iFA', 'iMISS']
    for behKey in behKeys:
        behArray = behavToArray(behavior[behKey])
        
        behNTrialsThis = len(behArray)
        if behNTrialsThis > 0:
            behMaxIdxThis  = np.max(behArray) - 1  # Note Matlab indices start from 1            
            if behMaxIdxThis >= dataNTrials:
                print("--Warning: For", behKey, "behaviour max index", behMaxIdxThis, "exceeds nTrials", dataNTrials)
                behavior[behKey] = behavior[behKey][behavior[behKey] < dataNTrials]
                print("---Cropped excessive behaviour trials from", behNTrialsThis, "to", len(behavior[behKey]))
            
    # CONSISTENCY TEST 2 - If neuronal trials are more than behavioural
    # Note that there are other keys except the above four, so data trials may be more than those for the four keys
    behIndices = [idx for key in behKeys for idx in behavToArray(behavior[key])]
    assert len(behIndices) <= dataNTrials, "After cropping behavioural trials may not exceed data trials"
    
    # CONSISTENCY TEST 3 - Test behavioural indices for duplicates
    for idx in behIndices:
        thisCount = behIndices.count(idx)
        if thisCount != 1:
            keysRep = {key : list(behavToArray(behavior[key])).count(idx) for key in behKeys if idx in behavToArray(behavior[key])}
            print("--WARNING: index", idx, "appears multiple times:", keysRep)
            
        #assert behIndices.count(idx) == 1, "Found duplicates in behaviour indices"
    
    return data, behavior, performance


# Read multiple neuro and performance files from a root folder
def read_neuro_perf_multi(rootpath):
    
    # Get all subfolders, mark them as mice
    mice = get_subfolders(rootpath)
    micedict = {}
    
    # For each mouse, get all subfolders, mark them as days
    for mouse in mice:
        mousepath = os.path.join(rootpath, mouse)
        days = get_subfolders(mousepath)
        micedict[mouse] = {day : {} for day in days}

        # For each day, read mat files
        for day in days:
            daypath = os.path.join(mousepath, day)
            data, behaviour = read_neuro_perf(daypath)
            micedict[mouse][day] = {
                'data' : data,
                'behaviour' : behaviour
            }
            
    return micedict


def read_lick(folderpath, verbose=True):
    if verbose:
        print("Processing lick folder", folderpath)
    
    rez = {}
    
    ################################
    # Process Reaction times file
    ################################
    rt_file = os.path.join(folderpath, "RT_264.mat")
    rt = loadmat(rt_file)
    
    rez['reaction_time'] = 3.0 + rt['reaction_time']
    
    ################################
    # Process lick_traces file
    ################################
    def lick_filter(data, bot_th, top_th):
        data[np.isnan(data)] = 0
        return np.logical_or(data <= bot_th, data >= top_th).astype(int)
    
    lick_traces_file = os.path.join(folderpath, "lick_traces.mat")
    lick_traces = loadmat(lick_traces_file)
    
    nTimesLick = len(lick_traces['licks_go'])
    freqLick = 100 # Hz
    rez['tLicks'] = np.arange(0, nTimesLick) / freqLick
    
    # Top threshold is wrong sometimes. Yaro said to use exact one
    thBot, thTop = lick_traces['bot_thresh'], 2.64
    
    for k in ['licks_go', 'licks_nogo', 'licks_miss', 'licks_FA', 'licks_early']:
        rez[k] = lick_filter(lick_traces[k], thBot, thTop)
        
    ################################
    # Process trials file
    ################################
    TIMESCALE_TRACES = 0.001 # ms
    trials_file = os.path.join(folderpath, os.path.basename(folderpath)+".mat")
    #print(trials_file)
    
    lick_trials = loadmat(trials_file)

    # NOTE: lick_trials['licks']['lick_vector'] is just a repeat from above lick_traces file
#     lick_trials['licks'] = merge_dicts([matstruct2dict(obj) for obj in lick_trials['licks']])
    lick_trials['trials'] = merge_dicts([matstruct2dict(obj) for obj in lick_trials['trials']])
    fixearly = lambda trial : np.nan if trial=='Early' else trial
    lick_trials['trials']['reward_time'] = [fixearly(trial) for trial in lick_trials['trials']['reward_time']]
    rez['reward_time'] = np.array(lick_trials['trials']['reward_time'], dtype=float) * TIMESCALE_TRACES
    rez['puff'] = [np.array(puff, dtype=float)*TIMESCALE_TRACES for puff in lick_trials['trials']['puff']]
        
    return rez


def read_paw(folderpath, verbose=True):
    if verbose:
        print("Processing paw folder", folderpath)
    
    filepath = os.path.join(folderpath, 'trials.mat')
    rezdict = {'trialsPaw' : loadmat(filepath)['trials']}
    
    nTrialsPaw, nTimesPaw = rezdict['trialsPaw'].shape
    if nTimesPaw == 64:
        freqPaw = 7
    elif nTimesPaw > 250:
        freqPaw = 30
    else:
        raise ValueError("Unexpected number of paw timesteps", nTimePaw)

    rezdict['tPaw'] = np.arange(0, nTimesPaw) / freqPaw
    rezdict['freqPaw'] = freqPaw
    return rezdict


def read_whisk(folderpath, verbose=True):
    if verbose:
        print("Processing whisk folder", folderpath)
    
    #############################
    # Read whisking angle
    #############################
    rezdict = {'whiskAngle' : loadmat(os.path.join(folderpath, 'whiskAngle.mat'))['whiskAngle']}
    nTimesWhisk, nTrialsWhisk = rezdict['whiskAngle'].shape
    if nTimesWhisk <= 900:
        freqWhisk = 40
    elif nTimesWhisk >= 1600:
        freqWhisk = 200
    else:
        freqWhisk = 40
        # raise ValueError("Unexpected number of whisk timesteps", nTimesWhisk)
        print("Unexpected number of whisk timesteps", nTimesWhisk)
    
    rezdict['tWhisk']           = np.arange(0, nTimesWhisk) / freqWhisk
    rezdict['whiskAbsVelocity'] = np.vstack((np.abs(rezdict['whiskAngle'][1:] - rezdict['whiskAngle'][:-1])*freqWhisk, np.zeros(nTrialsWhisk)))
        
    #############################
    # Read first touch
    #############################
    firstTouchFilePath = os.path.join(folderpath, os.path.basename(folderpath)+'.txt')
    if not os.path.isfile(firstTouchFilePath):
        print("Warning: first touch file does not exist", firstTouchFilePath)
        rezdict['firstTouch'] = None
    else:    
        with open(firstTouchFilePath) as fLog:
            rezdict['firstTouch'] = np.array([line.split('\t')[1] for line in fLog.readlines()[1:]], dtype=float)

    return rezdict

    
def read_lvm(filename, verbose=True):
    if verbose:
        print("Reading LVM file", filename, "... ")
    
    # Read file
    f = open(filename, 'r')
    lines = f.readlines()
    f.close()

    # Figure out where the headers are
    header_endings = [i for i in range(len(lines)) if "End_of_Header" in lines[i]]
        
    # Read data after the last header ends
    idx_data_start = header_endings[-1] + 2
    data = np.array([line.strip().split('\t') for line in lines[idx_data_start:]]).astype(float)
    channel_idxs = data[:,0].astype(int)
    times = data[:,1]

    # Figure out how many channels there are
    min_channel = np.min(channel_idxs)
    max_channel = np.max(channel_idxs)
    nChannel = max_channel - min_channel + 1
    nData = len(channel_idxs)//nChannel

    # Partition data into 2D array indexed by channel
    data2D = np.zeros((nChannel, nData))
    for i in range(nChannel):
        data2D[i] = times[channel_idxs == i]
        
    print("... done! Data shape read ", data2D.shape)
    return data2D