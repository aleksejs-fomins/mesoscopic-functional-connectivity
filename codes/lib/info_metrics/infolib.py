import numpy as np

from codes.lib.signal_lib import zscore
from codes.lib.array_lib import perm_map_str, unique_subtract
from codes.lib.info_metrics.corr_lib import corr_3D, cross_corr_3D, autocorr_3D, autocorr_d1_3D, avg_corr_3D
from codes.lib.info_metrics.npeet_wrapper import entropy, total_correlation, predictive_info
from codes.lib.info_metrics.mar_wrapper import ar1_coeff, ar1_testerr, ar_testerr


'''
TODO:
    * Move to new repository, check how to install/reinstall it
    * Initialize sweep once, then apply to all metrics
    * Store settings for each computed metric. If new settings entered, recompute
    * Have some adequate solution for extra axis for methods that return non-scalars. Especially dynamics like PI, TE
    * Copy over connectomics metrics
    * Window-based iterator
    * H5 I/O For data results
    * Optimization
        * intermediate storage
        * Parallelization
'''

class InfoLib:
    def __init__(self, data, srcDimOrder, trgDimOrder, zscoreDim=None):
        self.data = np.copy(data)

        # extract params
        self.srcDimOrder = srcDimOrder
        self.trgDimOrder = trgDimOrder

        # zscore whole data array if requested
        if zscoreDim is not None:
            axisZScore = tuple([i for i, e in enumerate(srcDimOrder) if e in zscoreDim])
            self.data = zscore(data, axisZScore)
        else:
            self.data = data

        # Determine axis that are going to disappear
        self.axis = tuple([i for i, e in enumerate(srcDimOrder) if e not in trgDimOrder]) if trgDimOrder != "" else None

        # Initialize metric library
        self.metricDict = {
            "mean"         : lambda data, axis, settings: np.nanmean(data, axis=axis),
            "std"          : lambda data, axis, settings: np.nanstd(data, axis=axis),
            "autocorr"     : lambda data, axis, settings: self._sweep(data, autocorr_3D, axis, settings),
            "corr"         : lambda data, axis, settings: self._sweep(data, corr_3D, axis, settings),
            "crosscorr"    : lambda data, axis, settings: self._sweep(data, cross_corr_3D, axis, settings),
            "autocorr_d1"  : lambda data, axis, settings: self._sweep(data, autocorr_d1_3D, axis, settings),
            "ar1_coeff"    : lambda data, axis, settings: self._sweep(data, ar1_coeff, axis, settings),
            "ar1_testerr"  : lambda data, axis, settings: self._sweep(data, ar1_testerr, axis, settings),
            "ar_testerr"   : lambda data, axis, settings: self._sweep(data, ar_testerr, axis, settings),
            "avgcorr"      : lambda data, axis, settings: self._sweep(data, avg_corr_3D, axis, settings),
            "entropy"      : lambda data, axis, settings: self._sweep(data, entropy, axis, settings),
            "TC"           : lambda data, axis, settings: self._sweep(data, total_correlation, axis, settings),
            "PI"           : lambda data, axis, settings: self._sweep(data, predictive_info, axis, settings)
        }

        # 2nd order metrics for converting non-scalar metrics into scalars
        self.integralDict = {}

        # Store results of each metric, in case they need to be reused
        self.resultDict = {}


    def wrapper_TC(self):


    def _sweep(self, data, metricFunc, axis, settings):
        '''
            Plan:
            1. Axis parameter denotes axis that are going to disappear. Calc dual axis that are going to stay
            2. Iterate over dual axes, project array, calc function
            3. Combine results into array, return
        '''

        # Axis to iterate over
        dualAxis = unique_subtract(tuple(range(3)), axis) if axis is not None else ()

        # Dim order after specifying the iterated axes
        settingsThis = settings.copy()
        settingsThis["dim_order"] = "".join(e for i, e in enumerate(settings["dim_order"]) if i not in dualAxis)

        # Transpose data such that dual axis is in front
        # Iterate over dual axis, assemble stuff in a list
        # Return array of that stuff
        if len(dualAxis) == 0:
            return metricFunc(data, settingsThis)
        elif len(dualAxis) == 1:
            transAxis = dualAxis + unique_subtract(tuple(range(3)), dualAxis)
            dataTrans = data.transpose(transAxis)
            rezLst = [metricFunc(dataTrans[i], settingsThis) for i in range(dataTrans.shape[0])]
            return np.array(rezLst)
        elif len(dualAxis) == 2:
            transAxis = dualAxis + unique_subtract(tuple(range(3)), dualAxis)
            dataTrans = data.transpose(transAxis)

            rezLst = []
            for i in range(dataTrans.shape[0]):
                rezLst += [[]]
                for j in range(dataTrans.shape[1]):
                    rezLst[-1] += [metricFunc(dataTrans[i][j], settingsThis)]
            return np.array(rezLst)
        else:
            raise ValueError("Weird axis", axis, dualAxis)


    def metric3D(self, metricName, metricSettings=None):
        # construct settings
        settings = {"dim_order" : self.srcDimOrder}
        if metricSettings is not None:
            settings.update(metricSettings)

        if metricName in self.resultDict:
            return self.resultDict[metricName]

        # Calculate metric
        rez = self.metricDict[metricName](self.data, self.axis, settings)

        # Determine final transpose that will be performed after application
        if len(self.trgDimOrder) < 2:
            self.resultDict[metricName] = rez
        else:
            postDimOrder = "".join([e for e in self.srcDimOrder if e in self.trgDimOrder])

            # The result may be non-scalar, we need to only transpose the loop dimensions, not the result dimensions
            # Thus, add fake dimensions for result dimensions
            lenRez = len(rez) if hasattr(rez, "__len__") else 0
            fakeDim = "".join([str(i) for i in range(lenRez)])

            self.resultDict[metricName] = rez.transpose(perm_map_str(postDimOrder + fakeDim, self.trgDimOrder + fakeDim))

        return self.resultDict[metricName]


    # Take existing metric value and convert it into scalar using an integral metric
    def pipe(self, metricName, integralName):
        resultKey = (metricName, integralName)
        if resultKey in self.resultDict.keys():
            return self.resultDict[resultKey]
        if metricName not in self.resultDict.keys():
            raise ValueError("Must first compute metric", metricName)

        self.resultDict[resultKey] = self.integralDict[integralName](self.resultDict[metricName])
        return self.resultDict[resultKey]
