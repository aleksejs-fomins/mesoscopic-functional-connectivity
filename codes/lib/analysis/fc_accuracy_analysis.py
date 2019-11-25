import os
import h5py
import numpy as np
import pandas as pd

from codes.lib.aux_functions import merge_dicts
from codes.lib.signal_lib import resample
from codes.lib.fc.fc_generic import fc_parallel_multiparam
from codes.lib.plots.accuracy import fc_accuracy_plots
from codes.lib.models.false_negative_boost import makedata_snr_observational, makedata_snr_occurence
from codes.lib.analysis.simulated_reader import read_data_h5, parse_file_names_pandas



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


def analysis_width_depth(dataFileNames, param):
    fileInfoDf, fileParams = parse_file_names_pandas(dataFileNames)
    paramThis = reinit_param(param)

    for analysis in fileParams['analysis']:
        for modelName in fileParams['model']:
            for nNode in fileParams['nNode']:
                dfThis = fileInfoDf[
                    (fileInfoDf['analysis'] == analysis) &
                    (fileInfoDf['modelname'] == modelName) &
                    (fileInfoDf['nNode'] == nNode)
                ]

                print("For analysis", analysis, "model", modelName, "nNode", nNode, "have", len(dfThis), "files")

                nDataEff = []
                dataLst = []
                for index, row in dfThis.iterrows():
                    fnameThis = dataFileNames[index]
                    expectedShape = (row["nTrial"], row["nTime"], nNode)
                    data, trueConn = read_data_h5(fnameThis, modelName, expectedShape)

                    dataLst += [data]
                    nDataEff += [expectedShape[0] * expectedShape[1]]

                # Run calculation
                rezIDTxl = wrapper_multiparam(dataLst, paramThis)

                for iMethod, method in enumerate(paramThis['methods']):
                    fname_h5 = analysis + "_" + modelName + '_' + str(nNode) + '.h5'
                    fname_fig = os.path.splitext(fname_h5)[0] + '_' + method + paramThis['figExt']
                    teData = rezIDTxl[method].transpose((1,2,3,0))
                    fc_accuracy_plots(nDataEff, teData, trueConn, method, paramThis['pTHR'], logx=True, percenty=True, h5_fname=fname_h5, fig_fname=fname_fig)


def analysis_snr(dataFileNames, modelName, nStep, param):
    window = 6
    paramThis = reinit_param(param, 1, 5)

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

    fileInfoDf, fileParams = parse_file_names_pandas(dataFileNames)
    analysis = 'typical'

    for nNode in fileParams['nNode']:
        dfThis = fileInfoDf[
            (fileInfoDf['analysis'] == analysis) &
            (fileInfoDf['modelname'] == modelName) &
            (fileInfoDf['nNode'] == nNode)
        ]

        print("For analysis", analysis, "model", modelName, "nNode", nNode, "have", len(dfThis), "files")

        for index, row in dfThis.iterrows():
            # Read data
            fnameThis = dataFileNames[index]
            expectedShape = (row["nTrial"], row["nTime"], nNode)
            data, trueConn = read_data_h5(fnameThis, modelName, expectedShape)
            data /= np.std(data)  # Normalize all data to have unit variance

            for flavour, paramRanges in paramRangesDict.items():
                print("- Processing Flavour", flavour)

                # Add noise to data according to noise flavour
                dataLst = dataNoiseFuncDict[flavour](data[:, :window, :], paramRanges)

                # Run calculation
                rezIDTxl = wrapper_multiparam(dataLst, paramThis)

                # Save to h5 and make plots
                for iMethod, method in enumerate(paramThis['methods']):
                    fname_h5 = "snr_" + flavour + '_' + modelName + '_' + str(nNode) + '_' + method
                    fname_fig = os.path.splitext(fname_h5)[0] + '_' + method + paramThis['figExt']
                    teData = rezIDTxl[method].transpose((1, 2, 3, 0))

                    fc_accuracy_plots(paramRanges, teData, trueConn, method, paramThis['pTHR'], logx=True, percenty=True, h5_fname=fname_h5, fig_fname=fname_fig)


def analysis_window(dataFileNames, wMin, wMax, param):
    paramThis = reinit_param(param, 1, 1)
    windowRange = np.arange(wMin, wMax+1)

    analysis = 'typical'
    fileInfoDf, fileParams = parse_file_names_pandas(dataFileNames)

    for modelName in fileParams['model']:
        for nNode in fileParams['nNode']:
            dfThis = fileInfoDf[
                (fileInfoDf['analysis'] == analysis) &
                (fileInfoDf['modelname'] == modelName) &
                (fileInfoDf['nNode'] == nNode)
            ]

            print("For analysis", analysis, "model", modelName, "nNode", nNode, "have", len(dfThis), "files")

            for index, row in dfThis.iterrows():
                # Read data
                fnameThis = dataFileNames[index]
                expectedShape = (row["nTrial"], row["nTime"], nNode)
                data, trueConn = read_data_h5(fnameThis, modelName, expectedShape)

                dataLst = [data[:, :window, :] for window in windowRange]

                # Run calculation
                rezIDTxl = fc_parallel_multiparam(
                    dataLst,
                    paramThis['library'],
                    paramThis['methods'],
                    paramThis['paramLib'],
                    parTarget=paramThis['parTrg'],
                    serial=paramThis['serial'],
                    nCore=paramThis['nCore']
                )

                # Save to h5 and make plots
                for iMethod, method in enumerate(paramThis['methods']):
                    fname_h5 = "window_" + str(wMin) + '_' + str(wMax) + '_' + modelName + '_' + str(nNode) + '_' + method
                    fname_fig = os.path.splitext(fname_h5)[0] + '_' + method + paramThis['figExt']
                    teData = rezIDTxl[method].transpose((1, 2, 3, 0))
                    fc_accuracy_plots(windowRange, teData, trueConn, method, paramThis['pTHR'], logx=False, percenty=True, h5_fname=fname_h5, fig_fname=fname_fig)


def analysis_lag(dataFileNames, lMin, lMax, param):
    paramThis = reinit_param(param, minlag=1)
    lagRange = np.arange(lMin, lMax+1)
    window = lMax + 1

    analysis = 'typical'
    fileInfoDf, fileParams = parse_file_names_pandas(dataFileNames)

    for modelName in fileParams['model']:
        for nNode in fileParams['nNode']:
            dfThis = fileInfoDf[
                (fileInfoDf['analysis'] == analysis) &
                (fileInfoDf['modelname'] == modelName) &
                (fileInfoDf['nNode'] == nNode)
            ]

            print("For analysis", analysis, "model", modelName, "nNode", nNode, "have", len(dfThis), "files")

            for index, row in dfThis.iterrows():
                # Read data
                fnameThis = dataFileNames[index]
                expectedShape = (row["nTrial"], row["nTime"], nNode)
                data, trueConn = read_data_h5(fnameThis, modelName, expectedShape)
                dataLst = [data[:, :window, :]]

                # Run calculation
                rezIDTxlLst = []
                for maxlag in lagRange:
                    paramThis['paramLib']['max_lag_sources'] = maxlag
                    rezIDTxlLst += [wrapper_multiparam(dataLst, paramThis)]
                rezIDTxlDict = merge_dicts(rezIDTxlLst)

                # Save to h5 and make plots
                for iMethod, method in enumerate(paramThis['methods']):
                    fname_h5 = "lag_" + str(lMin) + '_' + str(lMax) + '_' + modelName + '_' + str(nNode) + '_' + method
                    fname_fig = os.path.splitext(fname_h5)[0] + '_' + method + paramThis['figExt']
                    teData = np.array([teDataSingle[0] for teDataSingle in rezIDTxlDict[method]]).transpose((1, 2, 3, 0))

                    fc_accuracy_plots(lagRange, teData, trueConn, method, paramThis['pTHR'], logx=False, percenty=True, h5_fname=fname_h5, fig_fname=fname_fig)


def analysis_downsample(dataFileNames, downsampleFactorLst, param):
    window = 6
    paramThis = reinit_param(param, 1, 5)

    paramDS = {'method': 'averaging', 'kind': 'kernel'}
    dsMin = np.min(downsampleFactorLst)
    dsMax = np.max(downsampleFactorLst)


    analysis = 'typical'
    fileInfoDf, fileParams = parse_file_names_pandas(dataFileNames)

    for modelName in fileParams['model']:
        for nNode in fileParams['nNode']:
            dfThis = fileInfoDf[
                (fileInfoDf['analysis'] == analysis) &
                (fileInfoDf['modelname'] == modelName) &
                (fileInfoDf['nNode'] == nNode)
            ]

            print("For analysis", analysis, "model", modelName, "nNode", nNode, "have", len(dfThis), "files")

            for index, row in dfThis.iterrows():
                # Read data
                fnameThis = dataFileNames[index]
                expectedShape = (row["nTrial"], row["nTime"], nNode)
                data, trueConn = read_data_h5(fnameThis, modelName, expectedShape)

                # Downsample the data, then select only the window of interest
                dataLst = [downsample_factor(data, factor, paramDS)[:, :window, :] for factor in downsampleFactorLst]
                rezIDTxl = wrapper_multiparam(dataLst, paramThis)

                # Save to h5 and make plots
                for iMethod, method in enumerate(paramThis['methods']):
                    fname_h5 = "window_" + str(dsMin) + '_' + str(dsMax) + '_' + modelName + '_' + str(nNode) + '_' + method
                    fname_fig = os.path.splitext(fname_h5)[0] + '_' + method + paramThis['figExt']
                    teData = rezIDTxl[method].transpose((1, 2, 3, 0))

                    fc_accuracy_plots(downsampleFactorLst, teData, trueConn, method, paramThis['pTHR'], logx=False, percenty=True, h5_fname=fname_h5, fig_fname=fname_fig)
