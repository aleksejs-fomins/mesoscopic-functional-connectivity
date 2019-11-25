import os, sys

# Export library path
rootname = "mesoscopic-functional-connectivity"
thispath   = os.path.dirname(os.path.abspath(__file__))
rootpath = os.path.join(thispath[:thispath.index(rootname)], rootname)
print("Appending project path", rootpath)
sys.path.append(rootpath)

from codes.lib.data_io.qt_wrapper import gui_fnames, gui_fname
from codes.lib.analysis import fc_accuracy_analysis
from codes.lib.analysis.simulated_file_io import sweep_data_generator, parse_data_file_names_pandas


dataFileNames = gui_fnames("Get simulated data files", "./", "hdf5 (*.h5)")
#typicalFileName = gui_fname("Get typical data file", os.path.dirname(dataFileNames[0]), "hdf5 (*.h5)")

#############################
# Simulation parameters
#############################
param = {
    'library' :  "idtxl",
    'methods' :  ['BivariateMI', 'MultivariateMI', 'BivariateTE', 'MultivariateTE'],
    'pTHR'    :  0.01,
    'figExt'  :  '.svg',
    'parTrg'  :  True,
    'nCore'   :  4,
    'serial'  :  False,
    'paramLib' : {  # IDTxl parameters
        'dim_order'       : 'rsp',
        'cmi_estimator'   : 'JidtGaussianCMI',
        'max_lag_sources' : 1,
        'min_lag_sources' : 1
    }
}

#############################
# Parse all filenames
#############################

fileInfoDf, fileParams = parse_data_file_names_pandas(dataFileNames)

# #############################
# # Width / Depth Tests
# #############################
# data_gen = data_sweep_generator(fileInfoDf, dataFileNames, fileParams['analysis'], fileParams['model'], fileParams['nNode'])
#
# for sweepKey, dataLst, trueConnLst in data_gen:
#     fc_accuracy_analysis.analysis_width_depth(dataLst, trueConnLst, sweepKey + '.h5', param)

#############################
# SNR Tests
#############################
nStep = 40  # Number of different data sizes to pick
data_gen = sweep_data_generator(fileInfoDf, dataFileNames, ['typical'], ['dynsys'], fileParams['nNode'])

for sweepKey, dataLst, trueConnLst in data_gen:
    assert len(dataLst) == 1, "Criteria expected to match only one file at a time"
    fc_accuracy_analysis.analysis_snr(dataLst[0], trueConnLst[0], nStep, sweepKey + '.h5', param)

# ################
# # Window
# ################
# wMin = 2
# wMax = 10
# data_gen = data_sweep_generator(fileInfoDf, dataFileNames, ['typical'], fileParams['model'], fileParams['nNode'])
#
# for sweepKey, dataLst, trueConnLst in data_gen:
#     assert len(dataLst) == 1, "Criteria expected to match only one file at a time"
#     fc_accuracy_analysis.analysis_window(dataLst[0], trueConnLst[0], wMin, wMax, sweepKey + '.h5', param)
#
#
# ################
# # Lag
# ################
# lMin = 1
# lMax = 5
# data_gen = data_sweep_generator(fileInfoDf, dataFileNames, ['typical'], fileParams['model'], fileParams['nNode'])
#
# for sweepKey, dataLst, trueConnLst in data_gen:
#     assert len(dataLst) == 1, "Criteria expected to match only one file at a time"
#     fc_accuracy_analysis.analysis_lag(dataLst[0], trueConnLst[0], lMin, lMax, sweepKey + '.h5', param)
#
# ################
# # Downsample
# ################
# downsampleFactors = [1,2,4,6,8,10,12]
# data_gen = data_sweep_generator(fileInfoDf, dataFileNames, ['typical'], fileParams['model'], fileParams['nNode'])
#
# for sweepKey, dataLst, trueConnLst in data_gen:
#     assert len(dataLst) == 1, "Criteria expected to match only one file at a time"
#     fc_accuracy_analysis.analysis_downsample(dataLst[0], trueConnLst[0], downsampleFactors, sweepKey + '.h5', param)