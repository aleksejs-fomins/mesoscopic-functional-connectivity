# import standard libraries
import os, sys
import numpy as np
import matplotlib.pyplot as plt

# import special libraries
from lib.info_metrics.corr_lib import cross_corr_3D
from lib.plots.matrix import plotMatrix

'''
   Test 1:
     Generate random data, and shift it by fixed steps for each channel
     Expected outcomes:
     * If shift <= max_delay, corr ~ 1, delay = shift
     * If shift > max_delay, corr ~ 0, delay = rand
     * Delay is the same for all diagonals, because we compare essentially the same data, both cycled by the same amount
'''
    
nNode = 5
nData = 1000
settings = {
    'min_lag_sources' : 1,
    'max_lag_sources' : 2,
    'dim_order'       : 'ps'
}

#data = np.random.uniform(0, 1, nNode*nData).reshape((nNode, nData))
# Generate progressively more random data
data = np.zeros((nNode, nData))
data[0] = np.random.normal(0, 1, nData)
for i in range(1, nNode):
    data[i] = np.hstack((data[i-1][1:], data[i-1][0]))

rezCorr = cross_corr_3D(data, settings, est='corr')
rezSpr = cross_corr_3D(data, settings, est='spr')

compose = lambda lst1, lst2: [a + "_" + b for a in lst1 for b in lst2]

plotMatrix(
    "Test 1: Channels are shifts of the same data",
    (2, 3),
    [*rezCorr, *rezSpr],
    np.array(compose(["corr", "spr"], ["val", "lag", "p"])),
    lims = [[-1, 1], [0, settings['max_lag_sources']], [0, 1]]*2,
    draw = True
)


'''
   Test 2:
     Generate random data, all copies of each other, each following one a bit more noisy than prev
     Expected outcomes:
     * Correlation decreases with distance between nodes, as they are separated by more noise
     * Correlation should be approx the same for any two nodes given fixed distance between them
'''

nNode = 5
nData = 1000
settings = {
    'min_lag_sources' : 0,
    'max_lag_sources' : 0,
    'dim_order'       : 'ps'
}
alpha = 0.5

data = np.random.normal(0, 1, (nNode, nData))
for i in range(1, nNode):
    data[i] = data[i-1] * np.sqrt(1 - alpha) + np.random.normal(0, 1, nData) * np.sqrt(alpha)

rezCorr = cross_corr_3D(data, settings, est='corr')
rezSpr  = cross_corr_3D(data, settings, est='spr')

plotMatrix(
    "Test 2: Channels are same, but progressively more noisy",
    (2, 3),
    [*rezCorr, *rezSpr],
    np.array(compose(["corr", "spr"], ["val", "lag", "p"])),
    lims = [[-1, 1], [0, settings['max_lag_sources']], [0, 1]]*2,
    draw = True
)


'''
   Test 3:
     Random data structured by trials. Two channels (0 -> 3) connected with lag 6, others unrelated
     Expected outcomes:
     * No structure, except for (0 -> 3) connection
'''

nNode = 5
lagTrue = 6
settings = {
    'min_lag_sources' : 1,
    'max_lag_sources' : 6,
    'dim_order'       : 'rsp'
}
nData = settings['max_lag_sources']+1
nTrial = 200

data = np.random.normal(0, 1, (nTrial,nData,nNode))
data[0, lagTrue:, :] = data[3, :-lagTrue, :]

rezCorr = cross_corr_3D(data, settings, est='corr')
rezSpr  = cross_corr_3D(data, settings, est='spr')

plotMatrix(
    "Test 3: Random trial-based cross-correlation",
    (2, 3),
    [*rezCorr, *rezSpr],
    np.array(compose(["corr", "spr"], ["val", "lag", "p"])),
    lims = [[-1, 1], [0, settings['max_lag_sources']], [0, 1]]*2,
    draw = True
)

plt.show()
