# Export library path
import os, sys
thispath = os.path.dirname(os.path.abspath(__file__))
p1path = os.path.abspath(os.path.join(thispath, os.pardir))
p2path = os.path.abspath(os.path.join(p1path, os.pardir))
sys.path.append(os.path.join(p1path, 'lib/'))

# Locate results path
rezPath = os.path.join(os.path.join(p2path, 'data/'), 'sim_ds_h5')

# Standard libraries
import numpy as np
import matplotlib.pyplot as plt
import h5py

# IDTxl libraries
from idtxl.multivariate_te import MultivariateTE
from idtxl.data import Data

# Local libraries
from idtxl_wrapper import idtxlResults2matrix
from plots.plt_imshow_mat import plotImshowMat



fname_lst = [
    'sim_noise_pure_1.h5',
    'sim_noise_lpf_1.h5',
    'sim_cycle_1.h5',
    'sim_dynsys_1.h5',
    'sim_noise_pure_trial_1.h5',
    'sim_noise_lpf_trial_1.h5',
    'sim_cycle_trial_1.h5',
    'sim_dynsys_trial_1.h5'
]

for filename in fname_lst:
    
    #######################
    #  Load data
    #######################
    print('reading file', filename)
    
    rez_file_h5 = h5py.File(os.path.join(rezPath, filename), "r")
    data = np.copy(rez_file_h5['data'])
    
    #######################
    # Run IDTxl
    #######################
    
    DELAY_MIN = 1
    DELAY_MAX = 6
    
    # a) Convert data to ITDxl format
    if len(data.shape) == 2:
        N_NODE, N_DATA = data.shape
        dataIDTxl = Data(data, dim_order='ps')
    elif len(data.shape) == 3:
        N_TRIAL, N_DATA, N_NODE = data.shape
        #dataIDTxl = Data(data.transpose((2, 0, 1)), dim_order='spr')
        dataIDTxl = Data(data, dim_order='rsp')
    else:
        raise ValueError("Unexpected data shape", data.shape)

    # b) Initialise analysis object and define settings

    network_analysis = MultivariateTE()

    settings = {'cmi_estimator': 'JidtGaussianCMI',
                'max_lag_sources': DELAY_MAX,
                'min_lag_sources': DELAY_MIN}

    # c) Run analysis
    results = network_analysis.analyse_network(settings=settings, data=dataIDTxl)

    # d) Convert results into comfortable form
    results_mat = idtxlResults2matrix(results, N_NODE, method='MultivariateTE')
    
    rez_file_h5.close()

    #######################
    # Plot Results
    #######################

    plotImshowMat(
        results_mat,
        ['MultivariateTE', "Delays", "p-value"],
        'IDTxl analysis of ' + filename,
        (1, 3),
        lims = [[0, 1], [0, DELAY_MAX], [0, 1]],
        draw = True,
        savename = filename.split('.')[0] + '.png'
    )


plt.show()
