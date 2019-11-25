import os, sys
import numpy as np
import h5py
import pandas as pd


def parse_file_names_pandas(dataFileNames):
    baseNamesBare = [os.path.splitext(os.path.basename(fname))[0] for fname in dataFileNames]
    fileInfoDf = pd.DataFrame([fname.split('_') for fname in baseNamesBare],
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


def data_sweep_generator(fileInfoDf, dataFileNames, analysisLst, modelLst, nNodeLst):
    for analysis in analysisLst:
        for modelName in modelLst:
            for nNode in nNodeLst:
                dfThis = fileInfoDf[
                    (fileInfoDf['analysis'] == analysis) &
                    (fileInfoDf['modelname'] == modelName) &
                    (fileInfoDf['nNode'] == nNode)
                ]

                print("For analysis", analysis, "model", modelName, "nNode", nNode, "have", len(dfThis), "files")

                dataLst = []
                for index, row in dfThis.iterrows():
                    fnameThis = dataFileNames[index]
                    expectedShape = (row["nTrial"], row["nTime"], nNode)
                    data, trueConn = read_data_h5(fnameThis, modelName, expectedShape)

                    dataLst += [data]
