import numpy as np
import pandas as pd
import itertools

from os.path import basename, dirname, join, isfile, splitext

from codes.lib.sys_lib import strlst2date
# from codes.lib.data_io.yaro.mouse_performance import mouse_performance_allsessions
# from codes.lib.pandas_lib import filter_rows_colval, filter_rows_colvals
from codes.lib.data_io.matlab_lib import loadmat
from codes.lib.data_io.os_lib import getfiles_walk

from IPython.display import display
from ipywidgets import IntProgress

class BehaviouralNeuronalDatabase :
    def __init__(self, param):

        # Find and parse Data filenames
        self.mice = set()
        self.metaDataFrames = {}

        ##################################
        # Find and parse data files
        ##################################
        self._find_parse_data_files(param["root_path_data"])


    def _find_parse_data_files(self, rootPath):
        dataWalk = getfiles_walk(rootPath, [".mat", "AcceptedCells"])
        behavWalk = getfiles_walk(rootPath, [".mat", "behavior"])

        drop_non_digit = lambda s: ''.join([i for i in s if i.isdigit()])
        
        def digits2date(s):
            sd = drop_non_digit(s)
            return strlst2date([sd[:4], sd[4:6], sd[6:]])


        dataSplit = [splitext(name)[0].split('_') for path, name in dataWalk]
        behavSplit = [splitext(name)[0].split('_') for path, name in behavWalk]

        dataDict = {
            "mousename" : [sp[0].lower() for sp in dataSplit],
            "date"      : [digits2date(drop_non_digit(sp[-1])) for sp in dataSplit],
            "mousekey"  : ['_'.join([sp[0].lower(), drop_non_digit(sp[-1])]) for sp in dataSplit],
            "path"      : [join(path, name) for path, name in dataWalk]
        }

        behavDict = {
            "mousename" : [sp[0].lower() for sp in behavSplit],
            "date"      : [digits2date(drop_non_digit(sp[-1])) for sp in behavSplit],
            "mousekey"  : ['_'.join([sp[0].lower(), drop_non_digit(sp[-1])]) for sp in behavSplit],
            "path"      : [join(path, name) for path, name in behavWalk]
        }

        self.metaDataFrames['neuro'] = pd.DataFrame(dataDict)
        self.metaDataFrames['behavior'] = pd.DataFrame(behavDict)

        self.mice = set(dataDict['mousename'])
        self.mice.update(behavDict['mousename'])


    def read_neuro_files(self):
        if 'neuro' in self.metaDataFrames.keys():
            nNeuroFiles = self.metaDataFrames['neuro'].shape[0]

            self.dataNeuronal = []
            progBar = IntProgress(min=0, max=nNeuroFiles, description='Read Neuro Data:')
            display(progBar)  # display the bar
            for idx, filepath in enumerate(self.metaDataFrames['neuro']['path']):
                self.dataNeuronal += [list(loadmat(filepath, waitRetry=3).values())[0].shape]
                progBar.value += 1

        else:
            print("No Neuro files loaded, skipping reading part")


    def read_behavior_files(self):
        if 'behavior' in self.metaDataFrames.keys():
            nBehaviorFiles = self.metaDataFrames['behavior'].shape[0]

            self.dataNeuronal = []
            progBar = IntProgress(min=0, max=nBehaviorFiles, description='Read Neuro Data:')
            display(progBar)  # display the bar
            for idx, filepath in enumerate(self.metaDataFrames['behavior']['path']):
                loadmat(filepath, waitRetry=3)
                # self.dataNeuronal += [list(.values())[0].shape]
                progBar.value += 1

        else:
            print("No Neuro files loaded, skipping reading part")
