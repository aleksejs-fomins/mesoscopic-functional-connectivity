import os
import h5py
import numpy as np
import pandas as pd

from codes.lib.aux_functions import merge_dicts
from codes.lib.signal_lib import resample
from codes.lib.fc.fc_generic import fc_parallel_multiparam
from codes.lib.analysis.simulated_file_io import write_fc_h5
from codes.lib.models.false_negative_boost import makedata_snr_observational, makedata_snr_occurence


# Downsamples the data, such that the interval between time steps is "factor" times smaller
def downsample_factor(data, factor, paramDS):
    if factor < 1:
        raise ValueError("Downsampling times must be >=1, got", factor)
    elif factor == 1:
        return data
    else:
        nTrial, nTime, nNode = data.shape

        # The resulting number of INTERVALS must be approximately "factor" times smaller than original
        nTimeNew = int(np.round((nTime - 1) / factor + 1))

        # Actual times do not matter, only their ratio
        fakeTimesOrig = np.linspace(0, nTime-1, nTime)
        fakeTimesNew = np.linspace(0, nTime-1, nTimeNew)

        data_downsampled = np.zeros((nTrial, nTimeNew, nNode))

        for iTr in range(nTrial):
            for iNode in range(nNode):
                data_downsampled[iTr, :, iNode] = resample(fakeTimesOrig, data[iTr, :, iNode], fakeTimesNew, paramDS)


'''
param : {
    'library' ::  Library to use for estimation ('corr', 'idtxl')
    'methods' ::  Methods to use for estimation ('corr', 'spr', 'BivariateMi')
    'pTHR'    ::  P-value at which to threshold results for plotting
    'figExt'  ::  Extension for saving figure ('.png', '.svg')
    'parTrg'  ::  Whether to parallelize over targets (True, False)
    'nCore'   ::  Number of cores to use for parallelization (int, or None for automatic detection)
    'serial'  ::  Whether or not to parallelize code (True / False)
    'paramLib' :: Library-specific parameters {
        'min_lag_sources' :: min lag, int
        'min_lag_sources' :: max lag, int
        'dim_order'       :: Order of dimensions ('psr' <=> [nProcesses x nSamples x nRepetitions])
    }
}
'''
def reinit_param(param, minlag=None, maxlag=None):
    paramThis = param.copy()
    if minlag is not None:
        paramThis['paramLib']['min_lag_sources'] = minlag
    if maxlag is not None:
        paramThis['paramLib']['max_lag_sources'] = maxlag
    paramThis.setdefault('pTHR',   0.01)
    paramThis.setdefault('figExt', '.svg')
    paramThis.setdefault('parTrg', True)
    paramThis.setdefault('nCore', None)
    paramThis.setdefault('serial', False)
    return paramThis


# Shorthand to pass all parameters from dict to method
def wrapper_multiparam(dataLst, param):
    return fc_parallel_multiparam(
        dataLst,
        param['library'],
        param['methods'],
        param['paramLib'],
        parTarget=param['parTrg'],
        serial=param['serial'],
        nCore=param['nCore']
    )


# Write calculated FC data to file, making a new group for every method used
def wrapper_filewrite(xLst, rezIDTxl, connTrue, fname_h5, param):
    for iMethod, method in enumerate(param['methods']):
        fcData = rezIDTxl[method].transpose((1,2,3,0))
        write_fc_h5(fname_h5, xLst, fcData, method, connTrue=connTrue)


# Take several subsets of same dataset, with progressively increasing size (width=more time, depth=more trials)
def analysis_width_depth(dataLst, connTrueLst, fname_h5, param):
    paramThis = reinit_param(param)
    nDataEff = [data.shape[0] * data.shape[1] for data in dataLst]
    connTrue = np.array(connTrueLst)

    # Run calculation
    rezIDTxl = wrapper_multiparam(dataLst, paramThis)

    # Save to file
    wrapper_filewrite(nDataEff, rezIDTxl, connTrue, fname_h5, paramThis)


# Progressively add noise or fake trials until true connections disappear from FC
def analysis_snr(data, connTrue, nStep, fname_h5, param, lMin=1, lMax=5, window=6):
    paramThis = reinit_param(param, lMin, lMax)

    # Set parameter ranges
    paramRangesDict = {
        'observational'  : np.arange(nStep) / (nStep),
        'occurence'      : np.arange(nStep) / (nStep - 1)
    }

    # Set functions that will be used for noise generation
    dataNoiseFuncDict = {
        'observational'  : makedata_snr_observational,
        'occurence'      : makedata_snr_occurence
    }

    dataNorm = data / np.std(data)  # Normalize all data to have unit variance

    for flavour, paramRanges in paramRangesDict.items():
        print("- Processing Flavour", flavour)
        fname_prefix = "snr_" + flavour + "_"

        # Add noise to data according to noise flavour
        dataLst = dataNoiseFuncDict[flavour](dataNorm[:, :window, :], paramRanges)

        # Run calculation
        rezIDTxl = wrapper_multiparam(dataLst, paramThis)

        # Save to file
        wrapper_filewrite(paramRanges, rezIDTxl, connTrue, fname_prefix + fname_h5, paramThis)


# For same dataset, keep lags fixed and progressively increase window
def analysis_window(data, connTrue, wMin, wMax, fname_h5, param):
    paramThis = reinit_param(param, 1, 1)
    windowRange = np.arange(wMin, wMax+1)
    fname_prefix = "window_" + str(wMin) + '_' + str(wMax) + '_'

    dataLst = [data[:, :window, :] for window in windowRange]

    # Run calculation
    rezIDTxl = wrapper_multiparam(dataLst, paramThis)

    # Save to file
    wrapper_filewrite(windowRange, rezIDTxl, connTrue, fname_prefix + fname_h5, paramThis)


# For same dataset, keep window fixed and progressively increase maxlag
def analysis_lag(data, connTrue, lMin, lMax, fname_h5, param):
    paramThis = reinit_param(param, minlag=1)
    maxLagRange = np.arange(lMin+1, lMax+1)
    window = lMax + 1
    fname_prefix = "lag_" + str(lMin) + '_' + str(lMax) + '_'

    dataLst = [data[:, :window, :]]

    # Run calculation
    rezIDTxlLst = []
    for maxlag in maxLagRange:
        print(maxlag, type(maxlag))

        paramThis['paramLib']['max_lag_sources'] = int(maxlag)
        rezIDTxlLst += [wrapper_multiparam(dataLst, paramThis)]
    rezIDTxl = merge_dicts(rezIDTxlLst)
    rezIDTxl = {method : np.array([teDataSingle[0] for teDataSingle in rezLst]) for method, rezLst in rezIDTxl.items() }

    # Save to file
    wrapper_filewrite(maxLagRange, rezIDTxl, connTrue, fname_prefix + fname_h5, paramThis)


# For same dataset, keep window and lags fixed, progressively downsample data
def analysis_downsample(data, connTrue, downsampleFactorLst, fname_h5, param, lMin=1, lMax=5, window=6):
    paramThis = reinit_param(param, lMin, lMax)

    paramDS = {'method': 'averaging', 'kind': 'kernel'}
    dsMin = np.min(downsampleFactorLst)
    dsMax = np.max(downsampleFactorLst)
    fname_prefix = "downsample_" + str(dsMin) + '_' + str(dsMax) + '_'

    # Downsample the data, then select only the window of interest
    dataLst = [downsample_factor(data, factor, paramDS)[:, :window, :] for factor in downsampleFactorLst]

    # Run calculation
    rezIDTxl = wrapper_multiparam(dataLst, paramThis)

    # Save to file
    wrapper_filewrite(downsampleFactorLst, rezIDTxl, connTrue, fname_prefix + fname_h5, paramThis)
