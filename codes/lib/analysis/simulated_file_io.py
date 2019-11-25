import os, sys
import numpy as np
import h5py
import pandas as pd

# Reads data file
def read_data_h5(fname, expectedModel=None, expectedShape=None):
    with h5py.File(fname, "r") as h5f:
        trueConn = np.copy(h5f['results']['connTrue'])
        data = np.copy(h5f['results']['data'])

        if (expectedModel is not None) and (expectedModel != str(np.copy(h5f['results']['modelName']))):
            raise ValueError("Data model in the file does not correspond filename")

        if (expectedShape is not None) and (expectedShape != data.shape):
            raise ValueError("Data shape in the file does not correspond filename")

    return data, trueConn


# Write FC results to file
def write_fc_h5(h5_fname, xparam, fcData, method, connTrue=None):
    filemode = "a" if os.path.isfile(h5_fname) else "w"
    with h5py.File(h5_fname, filemode) as h5f:
        if "metadata" not in h5f.keys():
            grp_rez = h5f.create_group("metadata")
            grp_rez['xparam'] = xparam
            if connTrue is not None:
                grp_rez['connTrue'] = connTrue

        if method in h5f.keys():
            raise ValueError("Already have data for method", method)

        grp_method = h5f.create_group(method)
        grp_method['TE_table']    = fcData[0]
        grp_method['delay_table'] = fcData[1]
        grp_method['p_table']     = fcData[2]


# Reads FC results into a dictionary
def read_fc_h5(h5_fname, methods):
    with h5py.File(h5_fname, "r") as h5f:
        rezDict = {}

        rezDict['xparam'] = np.copy(h5f['metadata']['xparam'])
        rezDict['connTrue'] = np.copy(h5f['metadata']['connTrue'])

        for method in methods:
                rezDict[method] = [
                    np.copy(h5f[method]['TE_table']),
                    np.copy(h5f[method]['delay_table']),
                    np.copy(h5f[method]['p_table'])
                ]

        return rezDict


# Extract parameters from data filenames into pandas table
def parse_data_file_names_pandas(dataFileNames):
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


# Construct a generator to sweep over combinations of analysis types, model names and numbers of nodes.
# For every selects and reads all files corresponding to current sweep combination
def sweep_data_generator(fileInfoDf, dataFileNames, analysisLst, modelLst, nNodeLst):
    for analysis in analysisLst:
        for modelName in modelLst:
            for nNode in nNodeLst:
                dfThis = fileInfoDf[
                    (fileInfoDf['analysis'] == analysis) &
                    (fileInfoDf['modelname'] == modelName) &
                    (fileInfoDf['nNode'] == nNode)
                ]

                nFilesThis = len(dfThis)
                sweepKey = analysis + "_" + modelName + '_' + str(nNode) + '.h5'

                if nFilesThis > 0:
                    print("For analysis", analysis, "model", modelName, "nNode", nNode, "have", nFilesThis, "files")

                    dataLst = []
                    trueConnLst = []
                    for index, row in dfThis.iterrows():
                        fnameThis = dataFileNames[index]
                        expectedShape = (row["nTrial"], row["nTime"], nNode)
                        data, trueConn = read_data_h5(fnameThis, modelName, expectedShape)

                        dataLst += [data]
                        trueConnLst += [trueConn]

                    yield sweepKey, dataLst, trueConnLst