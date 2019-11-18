import os
import h5py
import numpy as np
import pandas as pd

from codes.lib.fc.te_idtxl_wrapper import idtxlParallelCPUMulti, idtxlResultsParse
from codes.lib.plots.accuracy import fc_accuracy_plots
from codes.lib.models.false_negative_transform import makedata_snr_observational, makedata_snr_occurence


def analysis_width_depth(dataFileNames, idtxlSettings, methods, pTHR=0.01):
    baseNames = [os.path.basename(fname) for fname in dataFileNames]
    fileInfoDf = pd.DataFrame([fname[:-3].split('_') for fname in baseNames],
                              columns = ['analysis', 'modelname', 'nTrial', 'nNode', 'nTime'])
    fileInfoDf = fileInfoDf.astype(dtype={'nTrial': 'int', 'nNode': 'int', 'nTime': 'int'})

    analysisNames = set(fileInfoDf['analysis'])
    modelNames = set(fileInfoDf['modelname'])
    nodeNumbers = set(fileInfoDf['nNode'])

    for analysis in analysisNames:
        for modelName in modelNames:
            for nNode in nodeNumbers:
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

                    # Read file here
                    with h5py.File(fnameThis, "r") as h5f:
                        trueConn = np.copy(h5f['results']['connTrue'])
                        dataLst += [np.copy(h5f['results']['data'])]
                        nDataEff += [expectedShape[0] * expectedShape[1]]

                        print(fnameThis)
                        print(expectedShape)
                        print(dataLst[-1].shape)

                        assert modelName == str(np.copy(h5f['results']['modelName'])), "Data shape in the file does not correspond filename"
                        assert dataLst[-1].shape == expectedShape, "Data shape in the file does not correspond filename"

                # Run calculation
                rezIDTxl = idtxlParallelCPUMulti(dataLst, idtxlSettings, methods)

                for iMethod, method in enumerate(methods):
                    fname_h5 = analysis + "_" + modelName + '_' + str(nNode) + '.h5'
                    fname_svg = fname_h5[:-3] + '_' + method + '.svg'
                    teData = rezIDTxl[method].transpose((1,2,3,0))
                    fc_accuracy_plots(nDataEff, teData, trueConn, method, pTHR, logx=True, percenty=True, h5_fname=fname_h5, fig_fname=fname_svg)


def analysis_snr(fname, modelName, idtxlSettings, nStep):
    # Read file here
    with h5py.File(fname, "r") as h5f:
        trueConn = np.copy(h5f['results']['connTrue'])
        data = np.copy(h5f['results']['data'])
        data /= np.std(data)  # Normalize all data to have unit variance
        nTrial, nTime, nNode = data.shape

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

    for flavour, paramRanges in paramRangesDict.items():
        print("- Processing Flavour", os.path.basename(fname))
        dataLst = dataNoiseFuncDict[flavour](data, paramRanges)

        # Run calculation
        rez = idtxlParallelCPUMulti(dataLst, idtxlSettings, "snr_" + flavour + "_" + modelName)

        for iMethod, method in enumerate(idtxlSettings['methods']):
            fname = "snr_" + flavour + '_' + modelName + '_' + str(nNode) + '_' + method
            teData = np.full((3, nNode, nNode, nStep), np.nan)

            # Parse Data
            for iStep in range(nStep):
                teData[..., iStep] = np.array(
                    idtxlResultsParse(rez[iMethod][iStep], nNode, method=method, storage='matrix'))

            fc_accuracy_plots(paramRanges, teData, trueConn, method, logx=True, percenty=True, pTHR=0.01, h5_fname=fname + '.h5',
                              fig_fname=fname + '.png')


def analysis_window(fname, idtxlSettings, windowRange):
    pass


def anaylsis_lag(fname, idtxlSettings, window):
    pass


def analysis_downsample(fname, idtxlSettings, downsampleRange):
    pass