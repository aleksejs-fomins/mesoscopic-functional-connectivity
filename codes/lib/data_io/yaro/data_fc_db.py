import numpy as np
import pandas as pd
import itertools

from os.path import basename, dirname, join, abspath

from codes.lib.aux_functions import bin_data_by_keys, strlst2date, slice_sorted
from codes.lib.metrics.mouse_performance import mouse_performance_allsessions
from codes.lib.pandas_lib import filter_rows_colval, filter_rows_colvals
from codes.lib.data_io.os_lib import getfiles_walk
from codes.lib.data_io.yaro.yaro_data_read import read_neuro_perf, read_paw, read_lick, read_whisk, readTE_H5, parse_TE_folder
from codes.lib.data_io.yaro.yaro_behaviour_preprocess import resample_lick, resample_paw, resample_whisk

from IPython.display import display
from ipywidgets import IntProgress

class DataFCDatabase :
    def __init__(self, param):

        # Find and parse Data filenames
        self.mice = set()
        self.metaDataFrames = {}

        ##################################
        # Define resampling frequency
        ##################################
        self.targetRange = [0, 8]  # Seconds goal
        self.targetFreq = 20  # Hz
        self.targetNTimes = int((self.targetRange[1] - self.targetRange[0]) * self.targetFreq) + 1
        self.targetTimes = np.linspace(self.targetRange[0], self.targetRange[1], self.targetNTimes)
        print("Target trial within", self.targetRange, "sec. Total target timesteps", self.targetNTimes)


        ##################################
        # Find and parse TE dataset
        ##################################
        if "root_path_te" in param.keys():
            print("Searching for TE files")
            self._find_parse_te_files(param["root_path_te"])

        ##################################
        # Find and parse data files
        ##################################
        if "root_path_data" in param.keys():
            print("Searching for channel labels")
            self._find_parse_channel_labels(param["root_path_data"])
            print("Searching for data files")
            self._find_parse_neuro_files(param["root_path_data"])
        else:
            print("No data path provided, skipping")

        ##################################
        # Find and parse paw files
        ##################################
        if "root_path_paw" in param.keys():
            print("Searching for paw files")
            self._find_parse_paw_files(param["root_path_paw"])
        else:
            print("No paw path provided, skipping")

        ##################################
        # Find and parse lick files
        ##################################
        if "root_path_lick" in param.keys():
            print("Searching for lick files")
            self._find_parse_lick_files(param["root_path_lick"])
        else:
            print("No lick path provided, skipping")

        ##################################
        # Find and parse whisk files
        ##################################
        if "root_path_whisk" in param.keys():
            print("Searching for whisk files")
            self._find_parse_whisk_files(param["root_path_whisk"])
        else:
            print("No whisk path provided, skipping")

        ##################################
        # Compute summary
        ##################################
        sumByMouse = lambda dataset: [dataset[dataset['mousename'] == mousename].shape[0] for mousename in self.mice]

        self.summary = pd.DataFrame({
            key : sumByMouse(dataFrame) for key, dataFrame in self.metaDataFrames.items()
        }, index=self.mice)


    # User selects multiple sets of H5 files, corresponding to different datasets
    # Parse filenames and get statistics of files in each dataset
    def _find_parse_te_files(self, datapath):
        self.summaryTE = parse_TE_folder(datapath)

        # Get basenames and paths
        fileswalk = getfiles_walk(datapath, ".h5")
        fbasenames = fileswalk[:, 1]
        print("Total user files in dataset", self.summaryTE["dataname"], "is", len(fbasenames))

        # Extract other info from basenames
        methodKeys = ["BivariateMI", "MultivariateMI", "BivariateTE", "MultivariateTE"]
        metaDict = {
            "mousename" : ["_".join(name.split('_')[:2]) for name in fbasenames],
            "mousekey"  : ["_".join(name.split('_')[:6]) for name in fbasenames],
            # "date"      : [strlst2date(name.split('_')[2:5]) for name in fbasenames],
            "analysis"  : bin_data_by_keys(fbasenames, ['swipe', 'range']),
            "trial"     : bin_data_by_keys(fbasenames, ['iGO', 'iNOGO']),
            "range"     : bin_data_by_keys(fbasenames, ['CUE', 'TEX', 'LIK']),
            "method"    : bin_data_by_keys(fbasenames, methodKeys),
            "path"      : np.array([join(path, fname) for path, fname in fileswalk])
        }

        self.metaDataFrames["TE"] = pd.DataFrame.from_dict(metaDict)

        summaryTEExtra = {
            "mousename": dict(zip(*np.unique(metaDict["mousename"], return_counts=True))),
            "analysis": dict(zip(*np.unique(metaDict["analysis"], return_counts=True))),
            "trial": dict(zip(*np.unique(metaDict["trial"], return_counts=True))),
            "range": dict(zip(*np.unique(metaDict["range"], return_counts=True))),
            "method": dict(zip(*np.unique(metaDict["method"], return_counts=True)))
        }
        self.summaryTE.update(summaryTEExtra)


    # Channel labels are brain regions associated to each channel index
    # The channel labels need not be consistent across mice, or even within one mouse
    def _find_parse_channel_labels(self, path):
        labelPaths = getfiles_walk(path, ['channel_labels.mat'])
        channelData = [[basename(path), join(path, name)] for path, name in labelPaths]
        channelDict = {k: v for k, v in zip(['mousename', 'path'], np.array(channelData).T)}
        self.metaDataFrames['channel_labels'] = pd.DataFrame(channelDict)
        self.mice.update(set(self.metaDataFrames['channel_labels']['mousename']))


    def _find_parse_neuro_files(self, path):
        dataPaths = getfiles_walk(path, ["data.mat"])
        neuroData = [[
            basename(path),
            path,
            basename(dirname(path)),
            strlst2date(basename(path).split("_")[2:5])
          ] for path, name in dataPaths
        ]
        neuroDict = {k: v for k, v in zip(['mousekey', 'path', 'mousename', 'date'], np.array(neuroData).T)}
        self.metaDataFrames['neuro'] = pd.DataFrame(neuroDict)
        self.mice.update(set(self.metaDataFrames['neuro']['mousename']))


    def _find_parse_paw_files(self, path):
        paw_paths = getfiles_walk(path, ["deltaI_paw.mat"])
        paw_data = [[
            basename(path),
            path,
            basename(dirname(path)),
            strlst2date(basename(path).split("_")[2:5])
          ] for path, name in paw_paths
        ]
        paw_dict = {k: v for k, v in zip(['mousekey', 'path', 'mousename', 'date'], np.array(paw_data).T)}
        self.metaDataFrames['paw'] = pd.DataFrame(paw_dict)
        self.mice.update(set(self.metaDataFrames['paw']['mousename']))


    def _find_parse_lick_files(self, path):
        lick_paths = getfiles_walk(path, ["lick_traces.mat"])
        lick_data = [[
            basename(path),
            path,
            basename(dirname(path)),
            strlst2date(basename(path).split("_")[2:5])
          ] for path, name in lick_paths
        ]
        lick_dict = {k: v for k, v in zip(['mousekey', 'path', 'mousename', 'date'], np.array(lick_data).T)}
        self.metaDataFrames['lick'] = pd.DataFrame(lick_dict)
        self.mice.update(set(self.metaDataFrames['lick']['mousename']))


    def _find_parse_whisk_files(self, path):
        whisk_paths = getfiles_walk(path, ["whiskAngle.mat"])
        whisk_data = [[
            basename(path),
            path,
            basename(dirname(path)),
            strlst2date(basename(path).split("_")[2:5])
          ] for path, name in whisk_paths
        ]
        whisk_dict = {k: v for k, v in zip(['mousekey', 'path', 'mousename', 'date'], np.array(whisk_data).T)}
        self.metaDataFrames['whisk'] = pd.DataFrame(whisk_dict)
        self.mice.update(set(self.metaDataFrames['whisk']['mousename']))


    def read_te_files(self):
        if "TE" in self.metaDataFrames.keys():
            self.dataTEtimes = []  # Timesteps for neuronal data
            self.dataTEFC = []     # (te, lag, p) of FC estimate

            progBar = IntProgress(min=0, max=len(self.metaDataFrames["TE"]["path"]), description='Reading TE files')
            display(progBar)  # display the bar
            for fpath in self.metaDataFrames["TE"]["path"]:
                times, data = readTE_H5(fpath, self.summaryTE)
                self.dataTEtimes += [times]
                self.dataTEFC += [data]
                progBar.value += 1
        else:
            print("No TE files loaded, skipping reading part")


    def read_neuro_files(self):
        if 'neuro' in self.metaDataFrames.keys():
            nNeuroFiles = self.metaDataFrames['neuro'].shape[0]

            self.dataNeuronal = []
            self.dataTrials = []
            self.dataPerformance = []
            badPerfIdxs = []

            progBar = IntProgress(min=0, max=nNeuroFiles, description='Read Neuro Data:')
            display(progBar)  # display the bar
            for idx, datapath in enumerate(self.metaDataFrames['neuro']['path']):
                data, behaviour, performance = read_neuro_perf(datapath, verbose=False)
                if (performance >= 0) and (performance <= 1):
                    self.dataNeuronal += [data]
                    self.dataTrials += [behaviour]
                    self.dataPerformance += [performance]
                else:
                    badPerfIdxs += [idx]
                progBar.value += 1
            self.dataPerformance = np.array(self.dataPerformance)

            # Drop all mice for which performance exceeds 1
            nBadPerfIdxs = len(badPerfIdxs)
            if nBadPerfIdxs > 0:
                print("Bad performance in", nBadPerfIdxs, "sessions, fixing")
                nRowsBefore = self.metaDataFrames['neuro'].shape[0]
                self.metaDataFrames['neuro'] = self.metaDataFrames['neuro'].drop(badPerfIdxs).reset_index(drop=True)
                nRowsAfter = self.metaDataFrames['neuro'].shape[0]
                if nRowsBefore - nRowsAfter != nBadPerfIdxs:
                    raise ValueError("Bad stuff", nRowsBefore, nRowsAfter, nBadPerfIdxs)

            # Fix mousekeys to remove trailing underscore
            for idx, row in self.metaDataFrames['neuro'].iterrows():
                if row['mousekey'][-1] == '_':
                    self.metaDataFrames['neuro'].at[idx, 'mousekey'] = row['mousekey'][:-1]
        else:
            print("No Neuro files loaded, skipping reading part")


    def read_resample_paw_files(self):
        if 'paw' in self.metaDataFrames.keys():
            nPawFiles = self.metaDataFrames['paw'].shape[0]
            dataPawResampled = []
            progBar = IntProgress(min=0, max=nPawFiles, description='Read paw data:')
            display(progBar) # display the bar
            for pawpath in self.metaDataFrames['paw']['path']:
                dataPaw = read_paw(pawpath, verbose=False)
                dataPawResampled += [resample_paw(dataPaw, self.targetTimes, self.targetFreq)]
                progBar.value += 1
        else:
            print("No paw files loaded, skipping reading part")


    def read_resample_lick_files(self):
        if 'lick' in self.metaDataFrames.keys():
            nLickFiles = self.metaDataFrames['lick'].shape[0]
            self.dataLickResampled = []
            progBar = IntProgress(min=0, max=nLickFiles, description='Read lick data:')
            display(progBar) # display the bar

            for index, row in self.metaDataFrames['lick'].iterrows():
                # Find behaviour associated with this lick
                dataIdxs = filter_rows_colval(self.metaDataFrames['lick'], 'mousekey', row['mousekey']).index
                if dataIdxs.shape[0] == 0:
                    self.dataLickResampled += [None]
                else:
                    dataIdx = dataIdxs[0]
                    neuro = self.dataNeuronal[dataIdx]
                    behaviour = self.dataTrials[dataIdx]
                    dataLick = read_lick(row['path'], verbose=False)
                    self.dataLickResampled += [resample_lick(dataLick, neuro, behaviour, self.targetTimes, self.targetFreq)]
                progBar.value += 1
        else:
            print("No lick files loaded, skipping reading part")


    def read_resample_whisk_files(self):
        if 'whisk' in self.metaDataFrames.keys():
            nWhiskFiles = self.metaDataFrames['whisk'].shape[0]
            progBar = IntProgress(min=0, max=nWhiskFiles, description='Read whisk data:')
            display(progBar) # display the bar
            self.dataWhiskResampled = []
            for whiskpath in self.metaDataFrames['whisk']['path']:
                dataWhisk = read_whisk(whiskpath, verbose=False)
                self.dataWhiskResampled += [resample_whisk(dataWhisk, self.targetTimes)]
                progBar.value += 1
        else:
            print("No whisk files loaded, skipping reading part")


    # Mark days as naive or expert based on performance threshold
    def mark_days_expert_naive(self, pTHR):
        nNeuroFiles = self.metaDataFrames['neuro'].shape[0]
        isExpert = np.zeros(nNeuroFiles, dtype=bool)
        deltaDays = np.zeros(nNeuroFiles)
        deltaDaysCentered = np.zeros(nNeuroFiles)

        # For each mouse, determine which sessions are naive and which expert
        # Also determine number of days passed since start and since expert
        for mousename in self.mice:
            thisMouseMetadata = filter_rows_colval(self.metaDataFrames['neuro'], 'mousename', mousename)
            thisMouseDataIdxs = np.array(thisMouseMetadata["date"].index)
            perf = self.dataPerformance[thisMouseDataIdxs]
            skillRez = mouse_performance_allsessions(list(thisMouseMetadata["date"]), perf, pTHR)
            isExpert[thisMouseDataIdxs], deltaDays[thisMouseDataIdxs], deltaDaysCentered[thisMouseDataIdxs] = skillRez

        # Add these values to metadata
        self.metaDataFrames['neuro']['isExpert'] = isExpert
        self.metaDataFrames['neuro']['deltaDays'] = deltaDays
        self.metaDataFrames['neuro']['deltaDaysCentered'] = deltaDaysCentered


    def get_rows(self, frameName, coldict):
        return filter_rows_colvals(self.metaDataFrames[frameName], coldict)


    # Find FC data for specified rows, then crop to selected time range
    def get_fc_data(self, rows, rangeSec=None):
        timesFC = []
        dataFC = []

        for idx in rows.index:
            if rangeSec is None:
                timesFC += [self.dataTEtimes[idx]]
                dataFC += [self.dataTEFC[idx]]
            else:
                rng = slice_sorted(timesFC, rangeSec)
                timesFC += [self.dataTEtimes[idx][rng[0]:rng[1]]]
                dataFC += [self.dataTEFC[idx][..., rng[0]:rng[1]]]

        return timesFC, dataFC


    # Provide rows for all sessions of the same mouse, iterating over combinations of other anaylsis parameters
    def mouse_iterator(self):
        sweepCols = ["mousename",  "analysis", "trial", "range", "method"]
        sweepValues = [self.summaryTE[colname].keys() for colname in sweepCols]
        sweepProduct = list(itertools.product(*sweepValues))

        for sweepComb in sweepProduct:
            sweepCombDict = dict(zip(sweepCols, sweepComb))
            rows = self.get_rows('TE', sweepCombDict)
            if rows.shape[0] > 0:
                yield sweepCombDict, rows