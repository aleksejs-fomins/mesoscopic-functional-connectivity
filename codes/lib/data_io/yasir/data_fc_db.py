import numpy as np
import pandas as pd
import itertools

from os.path import basename, dirname, join, isfile

# from codes.lib.aux_functions import bin_data_by_keys, strlst2date, slice_sorted
# from codes.lib.data_io.yaro.mouse_performance import mouse_performance_allsessions
from codes.lib.pandas_lib import filter_rows_colval, filter_rows_colvals
from codes.lib.data_io.matlab_lib import loadmat
from codes.lib.data_io.os_lib import getfiles_walk

from IPython.display import display
from ipywidgets import IntProgress

class DataFCDatabase :
    def __init__(self, param):

        # Adapt paths
        param["root_path_data"] = dirname(param["experiment_path"])

        # Find and parse Data filenames
        self.mice = set()
        self.metaDataFrames = {}


        ##################################
        # Define resampling frequency
        ##################################
        # self.targetRange = [0, 8]  # Seconds goal
        self.targetFreq = 20  # Hz
        # self.targetNTimes = int((self.targetRange[1] - self.targetRange[0]) * self.targetFreq) + 1
        # self.targetTimes = np.linspace(self.targetRange[0], self.targetRange[1], self.targetNTimes)
        # print("Target trial within", self.targetRange, "sec. Total target timesteps", self.targetNTimes)
    #
    #
    #     ##################################
    #     # Find and parse TE dataset
    #     ##################################
    #     if "root_path_te" in param.keys():
    #         print("Searching for TE files")
    #         self._find_parse_te_files(param["root_path_te"])
    #
        ##################################
        # Find and parse data files
        ##################################
        if "root_path_data" in param.keys():
            print("Reading channel label file")
            self._find_read_channel_labels(param["root_path_data"])
            print("Searching for data files")
            self._find_parse_neuro_files(param["experiment_path"])
        else:
            print("No data path provided, skipping")
    #
    #     ##################################
    #     # Compute summary
    #     ##################################
    #     sumByMouse = lambda dataset: [dataset[dataset['mousename'] == mousename].shape[0] for mousename in self.mice]
    #
    #     self.summary = pd.DataFrame({
    #         key : sumByMouse(dataFrame) for key, dataFrame in self.metaDataFrames.items()
    #     }, index=self.mice)
    #
    #
    # # User selects multiple sets of H5 files, corresponding to different datasets
    # # Parse filenames and get statistics of files in each dataset
    # def _find_parse_te_files(self, datapath):
    #     self.summaryTE = parse_TE_folder(datapath)
    #
    #     # Get basenames and paths
    #     fileswalk = getfiles_walk(datapath, ".h5")
    #     fbasenames = fileswalk[:, 1]
    #     print("Total user files in dataset", self.summaryTE["dataname"], "is", len(fbasenames))
    #
    #     # Extract other metric from basenames
    #     methodKeys = ["BivariateMI", "MultivariateMI", "BivariateTE", "MultivariateTE"]
    #     metaDict = {
    #         "mousename" : ["_".join(name.split('_')[:2]) for name in fbasenames],
    #         "mousekey"  : ["_".join(name.split('_')[:6]) for name in fbasenames],
    #         # "date"      : [strlst2date(name.split('_')[2:5]) for name in fbasenames],
    #         "analysis"  : bin_data_by_keys(fbasenames, ['swipe', 'range']),
    #         "trial"     : bin_data_by_keys(fbasenames, ['iGO', 'iNOGO']),
    #         "range"     : bin_data_by_keys(fbasenames, ['CUE', 'TEX', 'LIK']),
    #         "method"    : bin_data_by_keys(fbasenames, methodKeys),
    #         "path"      : np.array([join(path, fname) for path, fname in fileswalk])
    #     }
    #
    #     self.metaDataFrames["TE"] = pd.DataFrame.from_dict(metaDict)
    #
    #     summaryTEExtra = {
    #         "mousename": dict(zip(*np.unique(metaDict["mousename"], return_counts=True))),
    #         "analysis": dict(zip(*np.unique(metaDict["analysis"], return_counts=True))),
    #         "trial": dict(zip(*np.unique(metaDict["trial"], return_counts=True))),
    #         "range": dict(zip(*np.unique(metaDict["range"], return_counts=True))),
    #         "method": dict(zip(*np.unique(metaDict["method"], return_counts=True)))
    #     }
    #     self.summaryTE.update(summaryTEExtra)
    #
    #
    # Channel labels are brain regions associated to each channel index
    # The channel labels need not be consistent across mice, or even within one mouse
    def _find_read_channel_labels(self, path):
        labelFileName = join(path, "ROIs_names.mat")

        if not isfile(labelFileName):
            raise ValueError("Can't find file", labelFileName)

        self.channelLabels = loadmat(labelFileName)['ROIs_names']

    def _find_parse_neuro_files(self, path):
        dataPaths = [p[0] for p in getfiles_walk(path, ["data.mat"])]
        dataPathsRel = np.array([p[len(path) + 1:].split('/') for p in dataPaths])

        if dataPathsRel.shape[1] == 4:
            columns = ['mousename', 'activity', 'task_type', 'lolo']
        else:
            columns = ['mousename', 'lolo']

        self.metaDataFrames['neuro'] = pd.DataFrame(dataPathsRel, columns=columns)

        self.metaDataFrames['neuro'].insert(1, "path", dataPaths)
        self.mice.update(set(self.metaDataFrames['neuro']['mousename']))
    #
    #
    #
    #
    # def read_te_files(self):
    #     if "TE" in self.metaDataFrames.keys():
    #         self.dataTEtimes = []  # Timesteps for neuronal data
    #         self.dataTEFC = []     # (te, lag, p) of FC estimate
    #
    #         progBar = IntProgress(min=0, max=len(self.metaDataFrames["TE"]["path"]), description='Reading TE files')
    #         display(progBar)  # display the bar
    #         for fpath in self.metaDataFrames["TE"]["path"]:
    #             times, data = readTE_H5(fpath, self.summaryTE)
    #             self.dataTEtimes += [times]
    #             self.dataTEFC += [data]
    #             progBar.value += 1
    #     else:
    #         print("No TE files loaded, skipping reading part")
    #

    def read_neuro_files(self):
        if 'neuro' in self.metaDataFrames.keys():
            nNeuroFiles = self.metaDataFrames['neuro'].shape[0]

            self.dataNeuronal = []
            progBar = IntProgress(min=0, max=nNeuroFiles, description='Read Neuro Data:')
            display(progBar)  # display the bar
            for idx, datapath in enumerate(self.metaDataFrames['neuro']['path']):
                filepath = join(datapath, 'data.mat')
                self.dataNeuronal += [loadmat(filepath, waitRetry=3)['data']]
                progBar.value += 1

        else:
            print("No Neuro files loaded, skipping reading part")

    #
    #
    # # Mark days as naive or expert based on performance threshold
    # def mark_days_expert_naive(self, pTHR):
    #     nNeuroFiles = self.metaDataFrames['neuro'].shape[0]
    #     isExpert = np.zeros(nNeuroFiles, dtype=bool)
    #     deltaDays = np.zeros(nNeuroFiles)
    #     deltaDaysCentered = np.zeros(nNeuroFiles)
    #
    #     # For each mouse, determine which sessions are naive and which expert
    #     # Also determine number of days passed since start and since expert
    #     for mousename in self.mice:
    #         thisMouseMetadata = filter_rows_colval(self.metaDataFrames['neuro'], 'mousename', mousename)
    #         thisMouseDataIdxs = np.array(thisMouseMetadata["date"].index)
    #         perf = self.dataPerformance[thisMouseDataIdxs]
    #         skillRez = mouse_performance_allsessions(list(thisMouseMetadata["date"]), perf, pTHR)
    #         isExpert[thisMouseDataIdxs], deltaDays[thisMouseDataIdxs], deltaDaysCentered[thisMouseDataIdxs] = skillRez
    #
    #     # Add these values to metadata
    #     self.metaDataFrames['neuro']['isExpert'] = isExpert
    #     self.metaDataFrames['neuro']['deltaDays'] = deltaDays
    #     self.metaDataFrames['neuro']['deltaDaysCentered'] = deltaDaysCentered
    #
    #
    # def get_channel_labels(self, mousename):
    #     return self.channelLabelsDict[mousename]
    #
    #
    # def get_nchannels(self, mousename):
    #     return len(self.channelLabelsDict[mousename])
    #
    #
    def get_rows(self, frameName, coldict):
        return filter_rows_colvals(self.metaDataFrames[frameName], coldict)
    #
    #
    # # Find FC data for specified rows, then crop to selected time range
    # def get_fc_data(self, idx, rangeSec=None):
    #     timesThis = self.dataTEtimes[idx]
    #     fcThis = self.dataTEFC[idx]
    #     if rangeSec is None:
    #         return timesThis, fcThis
    #     else:
    #         rng = slice_sorted(timesThis, rangeSec)
    #         return timesThis[rng[0]:rng[1]], fcThis[..., rng[0]:rng[1]]
    #
    #
    # # Provide rows for all sessions of the same mouse, iterating over combinations of other anaylsis parameters
    # def mouse_iterator(self):
    #     sweepCols = ["mousename",  "analysis", "trial", "range", "method"]
    #     sweepValues = [self.summaryTE[colname].keys() for colname in sweepCols]
    #     sweepProduct = list(itertools.product(*sweepValues))
    #
    #     for sweepComb in sweepProduct:
    #         sweepCombDict = dict(zip(sweepCols, sweepComb))
    #         rows = self.get_rows('TE', sweepCombDict)
    #         if rows.shape[0] > 0:
    #             yield sweepCombDict, rows
