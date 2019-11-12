import os
import h5py
import numpy as np

from codes.lib.fc.te_idtxl_wrapper import idtxlParallelCPUMulti, idtxlResultsParse
from codes.lib.plots.accuracy import fc_plots
from codes.lib.models.false_negative_transform import makedata_snr_observational, makedata_snr_occurence


def analysis_width_depth(dataFileNames, idtxlSettings):
    for analysis in ["width", "depth"]:
        analysisFileNames = [fname for fname in dataFileNames if analysis in fname]
        print("Performing",analysis,"tests on", len(analysisFileNames), "files")

        for modelName in ["purenoise", "lpfsubnoise", "dynsys"]:
            modelFileNames = np.sort([fname for fname in analysisFileNames if modelName in fname])
            print("- For model", modelName, len(modelFileNames), "files")

            nFile = len(modelFileNames)
            te_results = { key : np.zeros((3, nNode, nNode, nFile)) for key in idtxlSettings['methods'] }

            nDataEff = []
            dataLst = []
            for iFile, fname in enumerate(modelFileNames):
                print("-- Reading Data File", os.path.basename(fname))

                # Read file here
                with h5py.File(fname, "r") as h5f:
                    modelName = str(h5f['results']['modelName'])
                    trueConn = np.copy(h5f['results']['connTrue'])
                    dataLst += [np.copy(h5f['results']['data'])]
                    nTrial, nTime, nNode = dataLst[-1].shape
                    nDataEff += [nTrial * nTime]


            # Run calculation
            rez = idtxlParallelCPUMulti(dataLst, idtxlSettings, analysis+"_"+modelName)

            for iMethod, method in enumerate(idtxlSettings['methods']):
                fname = analysis + "_" + modelName + '_' + str(nNode) + '_' + method
                teData = np.full((3, nNode, nNode, nFile), np.nan)

                # Parse Data
                for iFile in range(nFile):
                    teData[..., iFile] = np.array(idtxlResultsParse(rez[iMethod][iFile], nNode, method=method, storage='matrix'))

                fc_plots(nDataEff, teData, trueConn, method, logx=True, percenty=True, pTHR=0.01, h5_fname=fname + '.h5', fig_fname=fname + '.png')


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

            fc_plots(paramRanges, teData, trueConn, method, logx=True, percenty=True, pTHR=0.01, h5_fname=fname + '.h5',
                      fig_fname=fname + '.png')


def analysis_window(fname, idtxlSettings, windowRange):
    pass


def anaylsis_lag(fname, idtxlSettings, window):
    pass


def analysis_downsample(fname, idtxlSettings, downsampleRange):
    pass