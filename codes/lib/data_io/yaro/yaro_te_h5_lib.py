import numpy as np
import os, sys
import h5py
import pandas as pd
import datetime


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