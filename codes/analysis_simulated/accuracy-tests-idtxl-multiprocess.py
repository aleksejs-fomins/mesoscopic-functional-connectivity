import os, sys
import h5py
import numpy as np

from codes.lib.data_io.qt_wrapper import gui_fnames
from codes.lib.fc.te_idtxl_wrapper import idtxlParallelCPUMulti, idtxlResultsParse
from codes.lib.plots.accuracy_plots import testplots

'''
  Width/Depth Test Code:
    Load all width files
    Compute TE
    Make Plots, Store TE

  SNR Code:
    Load Typical File
    Loop over added noise / freq
    Compute TE
    Make Plots, Store TE

  Win/Lag/DS Code:
    Load Typical File
    Loop over windows, lag, ds
    Compute TE
    Make Plots, Store TE
'''

nNode = 12
dataFileNames = gui_fnames("Get simulated data files", "./", "hdf5 (*.h5)")

#############################
# IDTxl parameters
#############################
idtxl_settings = {
    'dim_order'       : 'rsp',
    'methods'         : ['BivariateMI', 'MultivariateMI', 'BivariateTE', 'MultivariateTE'],
    'cmi_estimator'   : 'JidtGaussianCMI',
    'max_lag_sources' : 1,
    'min_lag_sources' : 1}

#############################
# Width / Depth Tests
#############################

for analysis in ["width", "depth"]:
    analysisFileNames = [fname for fname in dataFileNames if analysis in fname]
    print("Performing",analysis,"tests on", len(analysisFileNames), "files")

    for modelName in ["purenoise", "lpfsubnoise", "dynsys"]:
        modelFileNames = np.sort([fname for fname in analysisFileNames if modelName in fname])
        print("- For model", modelName, len(modelFileNames), "files")

        nFile = len(modelFileNames)
        te_results = { key : np.zeros((3, nNode, nNode, nFile)) for key in idtxl_methods }

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
        rez = idtxlParallelCPUMulti(dataLst, idtxl_settings, analysis+"_"+modelName)

        for iMethod, method in enumerate(idtxl_settings['methods']):
            fname = analysis + "_" + modelName + '_' + str(nNode) + '_' + method
            te_data = np.full((3, nNode, nNode, nFile), np.nan)

            # Parse Data
            for iFile in range(nFile):
                te_data[..., iFile] = np.array(idtxlResultsParse(rez[iMethod][iFile], nNode, method=method, storage='matrix'))

            testplots(nDataEff, te_data, trueConn, logx=True, percenty=True, pTHR=0.01, h5_fname=fname + '.h5', fig_fname=fname + '.png')

#############################
# SNR Tests
#############################

#############################
# Param Tests
#############################