import numpy as np
from copy import copy

'''
The following functions provide iterators over slices of a dataset

Modifiers:
    * 1D             - (True) sweep over each channel                           (False) Multidimensional over channels
    * window_pooled  - (True) Time-points within a window are pooled together   (False) Multidimensional over window points
    * time_pooled    - (True) All windows from sweep are pooled together        (False) Sweep over windows
'''

# Add extra dimensions of size 1 to array at given locations
def numpy_reshape_extradim(x, reducedAxis):
    newShape = list(x.shape)
    for axis in reducedAxis:
        newShape.insert(axis, 1)
    return x.reshape(tuple(newShape))


class DataSweep:
    def __init__(self, data, settings, nSweepMax=None):
        self.data = data
        self.window = settings["window"]
        self.axisSamples = settings["dim_order"].index("s")  # Index of Samples dimension in data
        self.nSamples = data.shape[self.axisSamples]
        nSweepPossible = self.nSamples - self.window + 1
        self.nSweep = np.min([nSweepPossible, nSweepMax]) if nSweepMax is not None else nSweepPossible

    def iterator(self):
        if self.window is None:
            yield self.data
        else:
            for iSweep in range(self.nSweep):
                yield np.split(self.data, [iSweep, iSweep + self.window], axis=self.axisSamples)[1]

    def get_target_time_idxs(self):
        if self.window is None:
            raise ValueError("Not applicable to non-sweep")
        else:
            return self.window - 1 + np.arange(self.nSweep).astype(int)


class Sweep1D:
    def __init__(self, dataIter, methods, dimOrder, parCh, nCh):
        self.methods = methods    # Methods to sweep over
        self.dimOrder = dimOrder  # Dimension order of the data
        self.dataIter = dataIter  # Iterator to generate data
        self.nCh = nCh            # Number of channels
        self.parCh = parCh        # Whether to parallelize over channels

    def iterator(self):
        for data in self.dataIter:
            axisProcesses = self.dimOrder.index("p")  # Index of Processes dimension in data
            nProcesses = data.shape[axisProcesses]

            for method in self.methods:
                if self.parCh:
                    for iSrc in range(nProcesses):
                        x = np.take(data, iSrc, axis=axisProcesses)  # Select this channel
                        yield method, numpy_reshape_extradim(x, [axisProcesses])  # Add dummy channel dimension back
                else:
                    yield method, data

    def unpack(self, rezLst):
        nMethods = len(self.methods)
        rezArr = np.array(rezLst)

        if not self.parCh:   # rezArr.shape = [nData * nMethod, nRez, nCh]
            nComp, nRez, nCh = rezArr.shape

            nData = nComp // nMethods

            rezArrSplit = rezArr.reshape((nData, nMethods, nRez, nCh))

            # Convert data [nMethod, nData, nRez, nCh] -> {method : [nData, nRez, nCh]} - no transposes needed
            return {method : rezArrSplit[:, iMethod] for iMethod, method in enumerate(self.methods)}
        else:  # rezArr.shape = [nData * nMethod * nCh , nRez]
            nComp, nRez = rezArr.shape

            nData = nComp // (nMethods * self.nCh)

            rezArrSplit = rezArr.reshape((nData, nMethods, self.nCh, nRez))

            # Convert data [nMethod, nData, nRez, nCh] -> {method : [nData, nRez, nCh]}
            return {method : rezArrSplit[:, iMethod].transpose((0, 2, 1)) for iMethod, method in enumerate(self.methods)}


# Sweep over parameters of a 2D information metric.
#    2D means all combinations of self-coupling, resulting in matrix of shape [nChannel x nChannel]
class Sweep2D:
    def __init__(self, dataIter, methods, dimOrder, parTarget):
        self.methods = methods
        self.dimOrder = dimOrder
        self.dataIter = dataIter
        self.parTarget = parTarget

    def iterator(self):
        for data in self.dataIter:
            for method in self.methods:
                if self.parTarget:
                    axisProcesses = self.dimOrder.index("p")      # Index of Processes dimension in data
                    nProcesses = data.shape[axisProcesses]

                    for iTrg in range(nProcesses):
                        yield method, data, iTrg
                else:
                    yield method, data

    def unpack(self, rezLst):
        nMethods = len(self.methods)
        rezArr = np.array(rezLst)

        if not self.parTarget:   # [nData * nMethod, nRez, nSrc, nTrg]
            nComp, nRez, nSrc, nTrg = rezArr.shape
            nData = nComp // nMethods
            if nComp % nMethods != 0:
                raise ValueError("Unexpected resulting dimension", nComp, "for nMethods =", nMethods)

            rezArrSplit = rezArr.reshape((nData, nMethods, nRez, nSrc, nTrg))

            # Convert data [nData, nMethod, nRez, nSrc, nTrg] -> {method : [nData, nRez, nSrc, nTrg]} - no transposes needed
            return {method : rezArrSplit[:, iMethod] for iMethod, method in enumerate(self.methods)}
        else:  # [nData * nMethod * nTrg, nRez, nSrc]
            nComp, nRez, nSrc = rezArr.shape
            nTrg = nSrc                          # Currently both are assumed to be equal. We are using different names to make it easier to read which one is a source and which is a target
            nData = nComp // (nMethods * nTrg)
            if nComp % (nMethods * nTrg) != 0:
                raise ValueError("Unexpected resulting dimension", nComp, "for", nMethods, nTrg)

            rezArrSplit = rezArr.reshape((nData, nMethods, nTrg, nRez, nSrc))

            # Convert data [nData, nMethod, nTrg, nRez, nSrc] -> {method : [nData, nRez, nSrc, nTrg]}
            return {method : rezArrSplit[:, iMethod].transpose((0, 2, 3, 1)) for iMethod, method in enumerate(self.methods)}


