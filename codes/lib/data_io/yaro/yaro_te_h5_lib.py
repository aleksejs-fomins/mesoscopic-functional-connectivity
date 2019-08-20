import numpy as np
import os, sys
import h5py

# Export library path
thispath = os.path.dirname(os.path.abspath(__file__))
libpath = os.path.dirname(thispath)
sys.path.append(libpath)

from qt_wrapper import gui_fnames


# Extract TE from H5 file
def readTE_H5(fname, summary):
    print("Reading file", fname)
    #filename = os.path.join(pwd_h5, os.path.join("real_data", fname))
    #h5f = h5py.File(filename, "r")
    h5f = h5py.File(fname, "r")
    data = np.array([
        np.copy(h5f['results']['TE_table']),
        np.copy(h5f['results']['delay_table']),
        np.copy(h5f['results']['p_table'])
    ])
    h5f.close()
    
    # Crop data based on delay and window
    
    # Based on max lag and averaging time window
    #  some initial and final time steps of the data
    #  are essentialy not computed. First verify that
    #  they are indeed not computed, then crop them
    N_TIMES = data.shape[-1]
    GAP_L = summary["max_lag"]
    GAP_R = summary["window"] - summary["max_lag"] - 1
    if (GAP_L <= 0) or (GAP_R < 0):
        raise ValueError("Incompatible window and maxlag values ", summary["window"], summary["max_lag"])
    
    conn_L = np.sum(1 - np.isnan(data[..., :GAP_L]))
    if conn_L > 0:
        raise ValueError("While maxlag is", GAP_L, "found", conn_L, "non-nan connections in the first", GAP_L, "timesteps")
        
    if GAP_R > 0:
        conn_R = np.sum(1 - np.isnan(data[..., N_TIMES-GAP_R:]))
        if conn_R > 0:
            raise ValueError("While win-lag-1 is", GAP_R, "found", conn_R, "non-nan connections in the last", GAP_R, "timesteps")
            
    # Compute effective sampling times
    times = summary["timestep"] * (np.arange(GAP_L, N_TIMES-GAP_R) + GAP_R / 2)

    return times, data[..., GAP_L:N_TIMES-GAP_R]


# Find indices of partial occurences of keys in list
# Skip key if it has no occurences
# Add extra key 'other' for all elements that match no key
def idxs_by_keys(lst, keys):
    vals = [np.array([k in el for el in lst], dtype=int) for k in keys]
    d    = {k : v for k,v in zip(keys, vals) if np.sum(v) > 0}

    val_other = np.sum(vals, axis=0) == 0
    if np.sum(val_other) > 0:
        d['other'] = val_other

    return d


# Parse metadata from TE filenames
def getStatistics(dataname, basenames):        
    stat = {}
    METHOD_KEYS = ['BivariateMI', 'MultivatiateMI', 'BivariateTE', 'MultivariateTE']
    mouse_names = list(set(["_".join(name.split('_')[:2]) for name in basenames]))
    
    stat['isMouse']     = idxs_by_keys(basenames, mouse_names)           # Determine mouse which was used
    stat['isAnalysis']  = idxs_by_keys(basenames, ['swipe', 'range'])    # By Analysis type
    stat['isTrial']     = idxs_by_keys(basenames, ['iGO', 'iNOGO'])      # Determine if file uses GO, NOGO, or all
    stat['isRange']     = idxs_by_keys(basenames, ['CUE', 'TEX', 'LIK']) # Determine range types
    stat['isMethod']    = idxs_by_keys(basenames, METHOD_KEYS)           # Determine which method was used
    
    parse_dataname = dataname.split('_')

    assert parse_dataname[0] in ["raw", "subsample"], "Can't infer downsampling from " + dataname
    assert parse_dataname[1] == "delay", "Unexpected data folder name " + dataname
    assert parse_dataname[3] == "window", "Unexpected data folder name " + dataname
    
    if "raw" in dataname:
        timestep = 0.05 # seconds
    else:
        timestep = 10 / 49 # seconds  (I resampled to 50, not to 51 points [whoops], so DT is not 0.2 but a bit more
    
    summary = {
        "downsampling"  : parse_dataname[0],
        "timestep"      : timestep,
        "max_lag"       : int(parse_dataname[2]),
        "window"        : int(parse_dataname[4]),
        "mousename"     : {k: np.sum(v) for k,v in stat['isMouse'].items()},
        "analysis"      : {k: np.sum(v) for k,v in stat['isAnalysis'].items()},
        "trial"         : {k: np.sum(v) for k,v in stat['isTrial'].items()},
        "range"         : {k: np.sum(v) for k,v in stat['isRange'].items()},
        "method"        : {k: np.sum(v) for k,v in stat['isMethod'].items()}
    }
    
    return stat, summary


# User selects multiple sets of H5 files, corresponding to different datasets
# Parse filenames and get statistics of files in each dataset
def parseTEfolders(pwd_tmp = "./"):
    result = {}

    # GUI: Select videos for training
    datafilenames = None
    while datafilenames != ['']:
        datafilenames = gui_fnames("IDTXL swipe result files...", directory=pwd_tmp, filter="HDF5 Files (*.h5)")
        if datafilenames != ['']:
            pwd_tmp = os.path.dirname(datafilenames[0])  # Next time choose from parent folder
            
            name          = os.path.basename(pwd_tmp)
            filepathnames = np.array(datafilenames)
            filebasenames = np.array([os.path.basename(name) for name in datafilenames])
            stat, summary = getStatistics(name, filebasenames)
            
            print("Total user files in dataset", name, "is", len(datafilenames))
            print(summary)
            
            result[name] = {
                "filepaths"  : filepathnames,
                "filenames"  : filebasenames,
                "statistics" : stat,
                "summary"    : summary}

    return result


def flatten_param_sweep(keys, data_dict):
    getfirstkey = lambda d : next(iter(d.keys()))
    getfirstval = lambda d : next(iter(d.values()))
    
    # Reconstruct title
    stat = data_dict["statistics"]
    summary = data_dict["summary"]
    title = summary["downsampling"] + '_' + str(summary["max_lag"]) + '_' + str(summary["window"])
    
    # Initialize indices with ones
    N_ELEMS = len(getfirstval(stat[keys[0]]))
    thisTitles  = [title]
    thisIndices = [np.ones(N_ELEMS, dtype=int)]
    
    for key in keys:
        stat_dict_this = stat[key]
        # If this parameter does not exhibit interesting behaviour, omit it from the loop
        if len(stat_dict_this) == 1:
            key = getfirstkey(stat_dict_this)
            nextTitles  = thisTitles if key == 'other' else [title + "_" + key for title in thisTitles]
            nextIndices = thisIndices
        else:
            nextTitles  = []
            nextIndices = []
            for key, val in stat_dict_this.items():
                for title, idxs in zip(thisTitles, thisIndices):
                    nextTitles  += [title + "_" + key]
                    nextIndices += [idxs & val]
                
        thisTitles = nextTitles
        thisIndices = nextIndices
        
    return thisTitles, thisIndices
    

# # Extract indices of TE files based on constraints
# def getTitlesAndIndices_automatic(stat, dataname):    
    
#     def augment_title_idxs(title, idxs, d, type_k, type_v):
#         if (len(d) == 1) and (type_k == 'other'):
#             return title, idxs
#         else:
#             return title + "_" + type_k, idxs & type_v#.astype(int)
        
#     rez = {}
#     for mousename_k, mousename_v in stat['isMouse'].items():
#         for analysis_k, analysis_v in stat['isAnalysis'].items():
#             for trial_k, trial_v in stat['isTrial'].items():
#                 for rng_k, rng_v in stat['isRange'].items():
#                     for method_k, method_v in stat['isMethod'].items():
                        
#                         title = dataname
#                         idxs  = np.ones(len(mousename_v), dtype=int)
#                         title, idxs = augment_title_idxs(title, idxs, stat['isMouse'], mousename_k, mousename_v)
#                         title, idxs = augment_title_idxs(title, idxs, stat['isAnalysis'], analysis_k, analysis_v)
#                         title, idxs = augment_title_idxs(title, idxs, stat['isTrial'], trial_k, trial_v)
#                         title, idxs = augment_title_idxs(title, idxs, stat['isRange'], rng_k, rng_v)
#                         title, idxs = augment_title_idxs(title, idxs, stat['isMethod'], method_k, method_v)
                        
#                         rez[title] = idxs
#     return rez


# # Extract indices of TE files based on constraints
# def getTitlesAndIndices(stat, mouse, trials, methods, analysis_type, ranges=[None]):
#     rez = {}
#     isCorrectMouse = np.array([mname == mouse for mname in stat['mouse_names']], dtype=int)
#     for trial in trials:
#         for method in methods:
#             for rng in ranges:
#                 title = '_'.join([mouse, analysis_type, trial, method])
#                 select = np.copy(isCorrectMouse)
#                 select += stat['isAnalysis'][analysis_type]
#                 select += stat['isTrial'][trial]
#                 select += stat['isMethod'][method]
#                 test = 4
#                 if rng is not None:
#                     title+='_'+rng
#                     test+=1
#                     select += stat['isRange'][rng]
                    
#                 rez[title] = select == test
#     return rez