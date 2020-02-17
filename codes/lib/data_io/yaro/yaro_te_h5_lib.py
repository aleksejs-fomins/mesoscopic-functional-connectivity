import numpy as np


# def flatten_param_sweep(keys, data_dict):
#     getfirstkey = lambda d : next(iter(d.keys()))
#     getfirstval = lambda d : next(iter(d.values()))
#
#     # Reconstruct title
#     stat = data_dict["statistics"]
#     summary = data_dict["summary"]
#     title = summary["downsampling"] + '_' + str(summary["max_lag"]) + '_' + str(summary["window"])
#
#     # Initialize indices with ones
#     N_ELEMS = len(getfirstval(stat[keys[0]]))
#     thisTitles  = [title]
#     thisIndices = [np.ones(N_ELEMS, dtype=int)]
#
#     for key in keys:
#         stat_dict_this = stat[key]
#         # If this parameter does not exhibit interesting behaviour, omit it from the loop
#         if len(stat_dict_this) == 1:
#             key = getfirstkey(stat_dict_this)
#             nextTitles  = thisTitles if key == 'other' else [title + "_" + key for title in thisTitles]
#             nextIndices = thisIndices
#         else:
#             nextTitles  = []
#             nextIndices = []
#             for key, val in stat_dict_this.items():
#                 for title, idxs in zip(thisTitles, thisIndices):
#                     nextTitles  += [title + "_" + key]
#                     nextIndices += [idxs & val]
#
#         thisTitles = nextTitles
#         thisIndices = nextIndices
#
#     return thisTitles, thisIndices
