import os, sys
import numpy as np

# Export library path
rootname = "mesoscopic-functional-connectivity"
thispath   = os.path.dirname(os.path.abspath(__file__))
rootpath = os.path.join(thispath[:thispath.index(rootname)], rootname)
print("Appending project path", rootpath)
sys.path.append(rootpath)

from codes.lib.data_io.qt_wrapper import gui_fnames
from codes.lib.plots.accuracy import fc_accuracy_plots_fromfile


#############################
# Files
#############################
dataFileNames = gui_fnames("Get simulated data files", "./", "hdf5 (*.h5)")

#############################
# Params
#############################
pTHR = 0.01     # P-value for IDTxl thresholding
fExt = '.svg'   # Request vector graphics. Can also use .png
methods = ['BivariateMI', 'MultivariateMI', 'BivariateTE', 'MultivariateTE']

dataParamsDict = {
    "analysis"  : ["width", "depth", "snr", "window", "lag", "downsample"],
    "logx"      : [True, True, True, False, False, False],
}

def testAnalysisType(fname):
    testType = [key in fname for key in dataParamsDict['analysis']]
    if np.sum(testType) != 1:
        raise ValueError(fname, "matched", np.sum(testType), "analysis types")

    testIdx = np.where(testType)[0][0]
    analysis = dataParamsDict["analysis"][testIdx]
    logx = dataParamsDict["logx"][testIdx]
    return analysis, logx


for fname in dataFileNames:
    fig_fname = os.path.splitext(fname)[0] + fExt

    print(fig_fname)

    analysis, logx = testAnalysisType(fname)

    fc_accuracy_plots_fromfile(fname, methods, pTHR, logx=logx, percenty=True, fig_fname=fig_fname)
