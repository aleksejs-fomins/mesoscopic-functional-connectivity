# Export library path
import os, sys
thispath = os.path.dirname(os.path.abspath(__file__))
p1path = os.path.abspath(os.path.join(thispath, os.pardir))
p2path = os.path.abspath(os.path.join(p1path, os.pardir))
sys.path.append(os.path.join(p1path, 'lib/'))

# import standard libraries
import numpy as np
import matplotlib.pyplot as plt

# IDTxl libraries
from idtxl.bivariate_mi import BivariateMI
from idtxl.bivariate_te import BivariateTE
from idtxl.multivariate_te import MultivariateTE
from idtxl.data import Data
from idtxl.visualise_graph import plot_network

# import special libraries
from idtxl_wrapper import idtxlResults2matrix
from plots.plt_imshow_mat import plotImshowMat

'''
   Test 1:
     Generate random data, and shift it by fixed steps for each channel
     Expected outcomes:
     * If shift <= max_delay, corr ~ 1, delay = shift
     * If shift > max_delay, corr ~ 0, delay = rand
     * Delay is the same for all diagonals, because we compare essentially the same data, both cycled by the same amount
'''

#######################
# Generate Data
#######################
    
N_NODE = 5
N_DATA = 4000
DELAY_MIN = 1
DELAY_MAX = 4

#data = np.random.uniform(0, 1, N_NODE*N_DATA).reshape((N_NODE, N_DATA))
# Generate progressively more random data
data = np.zeros((N_NODE, N_DATA))
data[0] = np.random.normal(0, 1, N_DATA)
for i in range(1, N_NODE):
    data[i] = np.hstack((data[i-1][1:], data[i-1][0]))
    
# Have to add some noise to data or IDTxl will blow up :D
data += np.random.normal(0, 0.1, N_NODE*N_DATA).reshape((N_NODE, N_DATA))

#######################
# Run IDTxl
#######################
# a) Convert data to ITDxl format
dataIDTxl = Data(data, dim_order='ps')

# b) Initialise analysis object and define settings
title_lst = [
    "BivariateMI for shift-based data",
    "BivariateTE for shift-based data",
    "MultivariateTE for shift-based data"]
method_lst = ['BivariateMI', 'BivariateTE', 'MultivariateTE']
network_analysis_lst = [BivariateMI(), BivariateTE(), MultivariateTE()]

settings = {'cmi_estimator': 'JidtGaussianCMI',
            'max_lag_sources': DELAY_MAX,
            'min_lag_sources': DELAY_MIN}

# c) Run analysis
results_lst = [net_analysis.analyse_network(settings=settings, data=dataIDTxl) for net_analysis in network_analysis_lst]

# d) Convert results into comfortable form
results_mat_lst = [idtxlResults2matrix(results, N_NODE, method=method) for results, method in zip(results_lst, method_lst)]

#######################
# Plot Results
#######################

for results_matrices, method, title in zip(results_mat_lst, method_lst, title_lst):
    plotImshowMat(
        results_matrices,
        np.array([method, "Delays", "p-value"]),
        title,
        (1, 3),
        lims = [[0, 1], [0, DELAY_MAX], [0, 1]],
        draw = True
    )

plt.show()
