'''
1. Eliminate upper right triangle for BTE, introducing extra metric
2. Overplot BTE vs MTE
  * Load all FC files, parse names into pandas
  * Group together files where only METHOD is diff
  * Plot 1: FP-freq by method
  * Plot 2: FN-freq by method
3. Overplot DS, Lpfsub, noise
  * Load all FC files, parse names into pandas
  * Group together files where only MODELNAME is diff
  * Plot 1: FP-freq by method
  * Plot 2: FN-freq by method
4. Overplot DS, RealData for nData, Lag, DS - one mouse. See if already done in mtp15 - I long forgot
  * Load analysis-paired FC files - (DS vs RealData)
  * Plot 1: FP-DS, NConn-Real (BTE)
  * Plot 2: FP-DS, NConn-Real (MTE)
'''


import os, sys
import numpy as np

# Export library path
rootname = "mesoscopic-functional-connectivity"
thispath   = os.path.dirname(os.path.abspath(__file__))
rootpath = os.path.join(thispath[:thispath.index(rootname)], rootname)
print("Appending project path", rootpath)
sys.path.append(rootpath)

from codes.lib.data_io.qt_wrapper import gui_fnames
from codes.lib.plots.accuracy import bte_accuracy_special_fromfile
from codes.lib.analysis.simulated_file_io import read_fc_h5


#############################
# Files
#############################
dataFileNames = gui_fnames("Get simulated data files", "./", "hdf5 (*.h5)")

#############################
# Params
#############################
pTHR = 0.01     # P-value for IDTxl thresholding
fExt = '.svg'   # Request vector graphics. Can also use .png
methods = ['BivariateTE']

dataParamsDict = {
    "analysis"  : ["width", "depth", "snr", "window", "lag", "downsample"],
    "logx"      : [True, True, False, False, False, False],
}

# Determine analysis type and associated params
def parse_analysis_type(fname):
    basename = os.path.basename(fname)
    testType = [key in basename for key in dataParamsDict['analysis']]
    if np.sum(testType) != 1:
        raise ValueError(fname, "matched", np.sum(testType), "analysis types")

    testIdx = np.where(testType)[0][0]
    analysis = dataParamsDict["analysis"][testIdx]
    logx = dataParamsDict["logx"][testIdx]
    return analysis, logx


for fname in dataFileNames:
    # Determine analysis type and associated params
    analysis, logx = parse_analysis_type(fname)

    # Read the file
    rezDict = read_fc_h5(fname, methods)

    # Make a plot for every method in the file
    for method in methods:
        if method in rezDict.keys():
            fig_fname_method = os.path.splitext(fname)[0] + "_" + method + "_Excl"  + fExt

            print(fig_fname_method)

            connUndecided = np.triu(np.ones(rezDict['connTrue'].shape[0]), 2).astype(bool)

            bte_accuracy_special_fromfile(rezDict['xparam'], rezDict[method], rezDict['connTrue'], method, pTHR, connUndecided, logx=logx, percenty=True, fig_fname=fig_fname_method)
