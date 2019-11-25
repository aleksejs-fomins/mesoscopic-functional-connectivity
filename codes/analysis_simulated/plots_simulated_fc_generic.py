import os, sys
import numpy as np

# Export library path
rootname = "mesoscopic-functional-connectivity"
thispath   = os.path.dirname(os.path.abspath(__file__))
rootpath = os.path.join(thispath[:thispath.index(rootname)], rootname)
print("Appending project path", rootpath)
sys.path.append(rootpath)

from codes.lib.data_io.qt_wrapper import gui_fnames
from codes.lib.plots.accuracy import fc_accuracy_plots
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
methods = ['BivariateMI', 'MultivariateMI', 'BivariateTE', 'MultivariateTE']

dataParamsDict = {
    "analysis"  : ["width", "depth", "snr", "window", "lag", "downsample"],
    "logx"      : [True, True, True, False, False, False],
}

# Determine analysis type and associated params
def parse_analysis_type(fname):
    testType = [key in fname for key in dataParamsDict['analysis']]
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
            fig_fname_method = os.path.splitext(fname)[0] + "_" + method + fExt

            print(fig_fname_method)

            fc_accuracy_plots(rezDict['xparam'], rezDict[method], rezDict['connTrue'], method, pTHR, logx=logx, percenty=True, fig_fname=fig_fname_method)
