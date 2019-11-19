import os
import h5py
import numpy as np
import pandas as pd

from codes.lib.aux_functions import merge_dicts
from codes.lib.signal_lib import resample
from codes.lib.fc.te_idtxl_wrapper import idtxlParallelCPUMulti, idtxlParallelCPU
from codes.lib.plots.accuracy import fc_accuracy_plots
from codes.lib.models.false_negative_transform import makedata_snr_observational, makedata_snr_occurence


def parse_file_names_pandas(dataFileNames):
    baseNames = [os.path.basename(fname) for fname in dataFileNames]
    fileInfoDf = pd.DataFrame([fname[:-3].split('_') for fname in baseNames],
                              columns = ['analysis', 'modelname', 'nTrial', 'nNode', 'nTime'])
    fileInfoDf = fileInfoDf.astype(dtype={'nTrial': 'int', 'nNode': 'int', 'nTime': 'int'})
    fileParams = {
        "analysis" : set(fileInfoDf['analysis']),
        "model"    : set(fileInfoDf['modelname']),
        "nNode"    : set(fileInfoDf['nNode']),
    }

    return fileInfoDf, fileParams


def read_data_h5(fname, expectedModel=None, expectedShape=None):
    with h5py.File(fname, "r") as h5f:
        trueConn = np.copy(h5f['results']['connTrue'])
        data = np.copy(h5f['results']['data'])

        if (expectedModel is not None) and (expectedModel != str(np.copy(h5f['results']['modelName']))):
            raise ValueError("Data model in the file does not correspond filename")

        if (expectedShape is not None) and (expectedShape != data.shape):
            raise ValueError("Data shape in the file does not correspond filename")

    return data, trueConn


def downsample_times(data, timesDS, paramDS):
    if timesDS < 1:
        raise ValueError("Downsampling times must be >=1, got", timesDS)
    elif timesDS == 1:
        return data
    else:
        nTrial, nTime, nNode = data.shape

        # The resulting number of INTERVALS must be approximately "timesDS" times smaller than original
        nTimeNew = int(np.round((nTime - 1) / timesDS + 1))

        # Actual times do not matter, only their ratio
        fakeTimesOrig = np.linspace(0, nTime-1, nTime)
        fakeTimesNew = np.linspace(0, nTime-1, nTimeNew)

        data_downsampled = np.zeros((nTrial, nTimeNew, nNode))

        for iTr in range(nTrial):
            for iNode in range(nNode):
                data_downsampled[iTr, :, iNode] = resample(fakeTimesOrig, data[iTr, :, iNode], fakeTimesNew, paramDS)


def analysis_width_depth(dataFileNames, idtxlSettings, methods, pTHR=0.01, figExt='.svg', NCore=None):
    fileInfoDf, fileParams = parse_file_names_pandas(dataFileNames)

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
                rezIDTxl = idtxlParallelCPUMulti(dataLst, idtxlSettings, methods, NCore=NCore)

                for iMethod, method in enumerate(methods):
                    fname_h5 = analysis + "_" + modelName + '_' + str(nNode) + '.h5'
                    fname_svg = fname_h5[:-3] + '_' + method + figExt
                    teData = rezIDTxl[method].transpose((1,2,3,0))
                    fc_accuracy_plots(nDataEff, teData, trueConn, method, pTHR, logx=True, percenty=True, h5_fname=fname_h5, fig_fname=fname_svg)


def analysis_snr(dataFileNames, idtxlSettings, methods, modelName, nStep, pTHR=0.01, figExt='.svg', NCore=None):
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
                dataLst = dataNoiseFuncDict[flavour](data, paramRanges)

                # Run calculation
                rezIDTxl = idtxlParallelCPUMulti(dataLst, idtxlSettings, methods, NCore=NCore)

                # Save to h5 and make plots
                for iMethod, method in enumerate(methods):
                    fname_h5 = "snr_" + flavour + '_' + modelName + '_' + str(nNode) + '_' + method
                    fname_svg = fname_h5[:-3] + '_' + method + figExt
                    teData = rezIDTxl[method].transpose((1, 2, 3, 0))

                    fc_accuracy_plots(paramRanges, teData, trueConn, method, pTHR, logx=True, percenty=True, h5_fname=fname_h5, fig_fname=fname_svg)


def analysis_window(dataFileNames, idtxlSettings, methods, wMin, wMax, pTHR=0.01, figExt='.svg', NCore=None):
    idtxlSettingsThis = idtxlSettings.copy()
    idtxlSettingsThis['min_lag_sources'] = 1
    idtxlSettingsThis['max_lag_sources'] = 1
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
                rezIDTxl = idtxlParallelCPUMulti(dataLst, idtxlSettingsThis, methods, NCore=NCore)

                # Save to h5 and make plots
                for iMethod, method in enumerate(methods):
                    fname_h5 = "window_" + str(wMin) + '_' + str(wMax) + '_' + modelName + '_' + str(nNode) + '_' + method
                    fname_svg = fname_h5[:-3] + '_' + method + figExt
                    teData = rezIDTxl[method].transpose((1, 2, 3, 0))

                    fc_accuracy_plots(windowRange, teData, trueConn, method, pTHR, logx=False, percenty=True, h5_fname=fname_h5, fig_fname=fname_svg)


def analysis_lag(dataFileNames, idtxlSettings, methods, lMin, lMax, pTHR=0.01, figExt='.svg', NCore=None):
    idtxlSettingsThis = idtxlSettings.copy()
    idtxlSettingsThis['min_lag_sources'] = lMin
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

                rezIDTxlLst = []
                for maxlag in lagRange:
                    idtxlSettingsThis['max_lag_sources'] = maxlag
                    rezIDTxlLst += [idtxlParallelCPUMulti(dataLst, idtxlSettingsThis, methods, NCore=NCore)]
                rezIDTxlDict = merge_dicts(rezIDTxlLst)

                # Save to h5 and make plots
                for iMethod, method in enumerate(methods):
                    fname_h5 = "lag_" + str(lMin) + '_' + str(lMax) + '_' + modelName + '_' + str(nNode) + '_' + method
                    fname_svg = fname_h5[:-3] + '_' + method + figExt
                    teData = np.array([teDataSingle[0] for teDataSingle in rezIDTxlDict[method]]).transpose((1, 2, 3, 0))

                    fc_accuracy_plots(lagRange, teData, trueConn, method, pTHR, logx=False, percenty=True, h5_fname=fname_h5, fig_fname=fname_svg)


def analysis_downsample(dataFileNames, idtxlSettings, methods, downsampleTimesRange, pTHR=0.01, figExt='.svg', NCore=None):
    idtxlSettingsThis = idtxlSettings.copy()
    idtxlSettingsThis['min_lag_sources'] = 1
    idtxlSettingsThis['max_lag_sources'] = 5
    window = 6
    paramDS = {'method': 'averaging', 'kind': 'kernel'}
    dsMin = np.min(downsampleTimesRange)
    dsMax = np.max(downsampleTimesRange)


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
                dataLst = [downsample_times(data, timesDS, paramDS)[:, :window, :] for timesDS in downsampleTimesRange]
                rezIDTxl = idtxlParallelCPUMulti(dataLst, idtxlSettingsThis, methods, NCore=NCore)

                # Save to h5 and make plots
                for iMethod, method in enumerate(methods):
                    fname_h5 = "window_" + str(dsMin) + '_' + str(dsMax) + '_' + modelName + '_' + str(nNode) + '_' + method
                    fname_svg = fname_h5[:-3] + '_' + method + figExt
                    teData = rezIDTxl[method].transpose((1, 2, 3, 0))

                    fc_accuracy_plots(downsampleTimesRange, teData, trueConn, method, pTHR, logx=False, percenty=True, h5_fname=fname_h5, fig_fname=fname_svg)
