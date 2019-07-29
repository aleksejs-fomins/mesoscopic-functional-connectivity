import numpy as np
import os, sys
import h5py

# Export library path
thispath = os.path.dirname(os.path.abspath(__file__))
libpath = os.path.dirname(thispath)
sys.path.append(libpath)

from qt_wrapper import gui_fnames

# Extract TE from H5 file
def readTE_H5(fname):
    print("Reading file", fname)
    #filename = os.path.join(pwd_h5, os.path.join("real_data", fname))
    #h5f = h5py.File(filename, "r")
    h5f = h5py.File(fname, "r")
    TE = np.copy(h5f['results']['TE_table'])
    lag = np.copy(h5f['results']['delay_table'])
    p = np.copy(h5f['results']['p_table'])
    h5f.close()
    return (TE, lag, p)

# Parse metadata from TE filenames
def getStatistics(basenames):
    stat = {}
    
    # By Analysis type
    stat['isAnalysis'] = {}
    stat['isAnalysis']['swipe'] = np.array(["swipe" in name for name in basenames], dtype=int)
    stat['isAnalysis']['range'] = np.array(["range" in name for name in basenames], dtype=int)
    stat['isAnalysis']['all'] = stat['isAnalysis']['swipe'] + stat['isAnalysis']['range'] == 0

    # Determine if file uses GO, NOGO, or all
    stat['isTrial'] = {}
    stat['isTrial']['GO'] = np.array(["iGO" in name for name in basenames], dtype=int)
    stat['isTrial']['NOGO'] = np.array(["iNOGO" in name for name in basenames], dtype=int)
    stat['isTrial']['ALL']  = stat['isTrial']['GO'] + stat['isTrial']['NOGO'] == 0

    # Determine range types
    stat['isRange'] = {}
    stat['isRange']['CUE'] = np.array(["CUE" in name for name in basenames], dtype=int)
    stat['isRange']['TEX'] = np.array(["TEX" in name for name in basenames], dtype=int)
    stat['isRange']['LIK'] = np.array(["LIK" in name for name in basenames], dtype=int)
    stat['isRange']['none'] = stat['isRange']['CUE'] + stat['isRange']['TEX'] + stat['isRange']['LIK'] == 0

    # Determine which method was used
    stat['isMethod'] = {}
    stat['isMethod']['BTE'] = np.array(["BivariateTE" in name for name in basenames], dtype=int)
    stat['isMethod']['MTE'] = np.array(["MultivariateTE" in name for name in basenames], dtype=int)

    # Determine mouse which was used
    stat['mouse_names'] = ["_".join(name.split('_')[:2]) for name in basenames]
    
    summary = {
        "mousename" : {k: stat['mouse_names'].count(k) for k in set(stat['mouse_names'])},
        "analysis"  : {k: np.sum(v) for k,v in stat['isAnalysis'].items()},
        "trial"     : {k: np.sum(v) for k,v in stat['isTrial'].items()},
        "range"     : {k: np.sum(v) for k,v in stat['isRange'].items()},
        "method"    : {k: np.sum(v) for k,v in stat['isMethod'].items()}
    }
    
    return stat, summary

# User selects multiple sets of H5 files, corresponding to different datasets
# Parse filenames and get statistics of files in each dataset
def parseTEfolders(pwd_tmp = "./"):
    datafilesets = []
    basenamesets = []
    statistics = []

    # GUI: Select videos for training
    datafilenames = None
    while datafilenames != ['']:
        datafilenames = gui_fnames("IDTXL swipe result files...", directory=pwd_tmp, filter="HDF5 Files (*.h5)")
        if datafilenames != ['']:
            print("Total user files in dataset", len(datafilesets), "is", len(datafilenames))   

            pwd_tmp = os.path.dirname(datafilenames[0])  # Next time choose from parent folder
            datafilesets += [np.array(datafilenames)]
            basenamesets += [np.array([os.path.basename(name) for name in datafilenames])]
            stat, summary = getStatistics(basenamesets[-1])
            statistics   += [stat]
            print(summary)
            
    return datafilesets, basenamesets, statistics

# Extract indices of TE files based on constraints
def getTitlesAndIndices(stat, mouse, trials, methods, analysis_type, ranges=[None]):
    rez = []
    isCorrectMouse = np.array([mname == mouse for mname in stat['mouse_names']], dtype=int)
    for trial in trials:
        for method in methods:
            for rng in ranges:
                title = '_'.join([mouse, analysis_type, trial, method])
                select = np.copy(isCorrectMouse)
                select += stat['isAnalysis'][analysis_type]
                select += stat['isTrial'][trial]
                select += stat['isMethod'][method]
                test = 4
                if rng is not None:
                    title+='_'+rng
                    test+=1
                    select += stat['isRange'][rng]
                rez.append((title, select == test))
    return rez