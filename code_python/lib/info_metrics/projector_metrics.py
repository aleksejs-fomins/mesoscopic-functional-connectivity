import numpy as np

from lib.signal_lib import zscore
from lib.array_lib import perm_map_str, unique_subtract
from lib.info_metrics.corr_lib import corr_3D, cross_corr_3D, autocorr_3D, autocorr_d1_3D, avg_corr_3D
from lib.info_metrics.npeet_wrapper import entropy, total_correlation, predictive_info
from lib.info_metrics.mar_wrapper import ar1_coeff, ar1_testerr, ar_testerr


'''
    Temporal
        Static          history irrelevant, times stay the same
        Dynamic         history used, times truncated by window
    Spatial Input
        Univariate      1 channel at a time
        Multivariate    all channels at a time
    Spatial Output
        Const           1 number regardless of input
        Linear          1D wrt channels
        Quadratic       2D wrt channels
        
    TODO:
        * Window-based iterator
        * H5 I/O For data results
        * Optimization
            * Convert to class with intermediate storage
            * Parallelization
        * Convert to standalone library
        
'''

def applier(data, metricFunc, axis, settings):
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


def metric3D(data, metricName, metricParam):
    metricDict = {
        "mean"         : lambda data, axis, settings: np.nanmean(data, axis=axis),
        "std"          : lambda data, axis, settings: np.nanstd(data, axis=axis),
        "autocorr"     : lambda data, axis, settings: applier(data, autocorr_3D, axis, settings),
        "corr"         : lambda data, axis, settings: applier(data, corr_3D, axis, settings),
        "crosscorr"    : lambda data, axis, settings: applier(data, cross_corr_3D, axis, settings),
        "autocorr_d1"  : lambda data, axis, settings: applier(data, autocorr_d1_3D, axis, settings),
        "ar1_coeff"    : lambda data, axis, settings: applier(data, ar1_coeff, axis, settings),
        "ar1_testerr"  : lambda data, axis, settings: applier(data, ar1_testerr, axis, settings),
        "ar_testerr"   : lambda data, axis, settings: applier(data, ar_testerr, axis, settings),
        "avgcorr"      : lambda data, axis, settings: applier(data, avg_corr_3D, axis, settings),
        "entropy"      : lambda data, axis, settings: applier(data, entropy, axis, settings),
        "TC"           : lambda data, axis, settings: applier(data, total_correlation, axis, settings),
        "PI"           : lambda data, axis, settings: applier(data, predictive_info, axis, settings)
    }

    # extract params
    srcDimOrder = metricParam['src_dim_order']
    trgDimOrder = metricParam['trg_dim_order']

    # construct settings
    settings = {"dim_order" : metricParam['src_dim_order']}
    if 'settings' in metricParam.keys():
        settings.update(metricParam['settings'])

    # zscore whole data array if requested
    if 'zscoreDim' in metricParam.keys():
        axisZScore = tuple([i for i, e in enumerate(settings["dim_order"]) if e in metricParam['zscoreDim']])
        dataEff = zscore(data, axisZScore)
    else:
        dataEff = data

    # Determine axis that are going to disappear
    axis = tuple([i for i, e in enumerate(srcDimOrder) if e not in trgDimOrder]) if trgDimOrder != "" else None

    # Calculate metric
    rez = metricDict[metricName](dataEff, axis, settings)

    # Determine final transpose that will be performed after application
    if len(trgDimOrder) < 2:
        return rez
    else:
        postDimOrder = "".join([e for e in srcDimOrder if e in trgDimOrder])

        # The result may be non-scalar, we need to only transpose the loop dimensions, not the result dimensions
        # Thus, add fake dimensions for result dimensions
        lenRez = len(rez) if hasattr(rez, "__len__") else 0
        fakeDim = "".join([str(i) for i in range(lenRez)])

        return rez.transpose(perm_map_str(postDimOrder + fakeDim, trgDimOrder + fakeDim))
