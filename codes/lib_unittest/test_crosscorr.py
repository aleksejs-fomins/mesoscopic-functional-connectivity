# Export library path
import os, sys
thispath = os.path.dirname(os.path.abspath(__file__))
p1path = os.path.abspath(os.path.join(thispath, os.pardir))
p2path = os.path.abspath(os.path.join(p1path, os.pardir))
sys.path.append(os.path.join(p1path, 'lib/'))

# import standard libraries
import numpy as np
import matplotlib.pyplot as plt

# import special libraries
from corr_lib import crossCorr
from plots.plt_imshow_mat import plotImshowMat

'''
   Test 1:
     Generate random data, and shift it by fixed steps for each channel
     Expected outcomes:
     * If shift <= max_delay, corr ~ 1, delay = shift
     * If shift > max_delay, corr ~ 0, delay = rand
     * Delay is the same for all diagonals, because we compare essentially the same data, both cycled by the same amount
'''
    
N_NODE = 5
N_DATA = 1000
DELAY_MIN = 1
DELAY_MAX = 2

#data = np.random.uniform(0, 1, N_NODE*N_DATA).reshape((N_NODE, N_DATA))
# Generate progressively more random data
data = np.zeros((N_NODE, N_DATA))
data[0] = np.random.normal(0, 1, N_DATA)
for i in range(1, N_NODE):
    data[i] = np.hstack((data[i-1][1:], data[i-1][0]))

corrMat, corrDelMat = crossCorr(data, DELAY_MIN, DELAY_MAX, est='corr')
sprMat, sprDelMat = crossCorr(data, DELAY_MIN, DELAY_MAX, est='spr')

plotImshowMat(
    [corrMat, corrDelMat, sprMat, sprDelMat],
    np.array(["Correlation", "Corr. Delays", "Spearmann", "Spr. Delays"]),
    "Test 1: Channels are shifts of the same data",
    (2, 2),
    lims = [[-1, 1], [0, DELAY_MAX], [-1, 1], [0, DELAY_MAX]],
    draw = True
)


'''
   Test 2:
     Generate random data, all copies of each other, each following one a bit more noisy than prev
     Expected outcomes:
     * Correlation decreases with distance between nodes, as they are separated by more noise
     * Correlation should be approx the same for any two nodes given fixed distance between them
'''

N_NODE = 5
N_DATA = 1000
DELAY_MIN = 0
DELAY_MAX = 0
alpha = 0.5

data = np.random.normal(0, 1, N_NODE*N_DATA).reshape((N_NODE, N_DATA))
for i in range(1, N_NODE):
    data[i] = data[i-1] * np.sqrt(1 - alpha) + np.random.normal(0, 1, N_DATA) * np.sqrt(alpha)

corrMat, corrDelMat = crossCorr(data, DELAY_MIN, DELAY_MAX, est='corr')
sprMat, sprDelMat = crossCorr(data, DELAY_MIN, DELAY_MAX, est='spr')

plotImshowMat(
    [corrMat, corrDelMat, sprMat, sprDelMat],
    np.array(["Correlation", "Corr. Delays", "Spearmann", "Spr. Delays"]),
    "Test 2: Channels are same, but progressively more noisy",
    (2, 2),
    lims = [[-1, 1], [0, DELAY_MAX], [-1, 1], [0, DELAY_MAX]],
    draw = True
)


'''
   Test 3:
     Random data structured by trials. Two channels (0 -> 3) connected with lag 6, others unrelated
     Expected outcomes:
     * No structure, except for (0 -> 3) connection
'''

N_NODE = 5
DELAY_MIN = 1
DELAY_MAX = 6
N_DATA = DELAY_MAX+1
N_TRIAL = 200

data = np.random.normal(0, 1, N_TRIAL*N_DATA*N_NODE).reshape((N_TRIAL,N_DATA,N_NODE))
data[:, -1, 3] = data[:, 0, 0]

corrMat, corrDelMat = crossCorr(data, DELAY_MIN, DELAY_MAX, est='corr')
sprMat, sprDelMat = crossCorr(data, DELAY_MIN, DELAY_MAX, est='spr')

plotImshowMat(
    [corrMat, corrDelMat, sprMat, sprDelMat],
    np.array(["Correlation", "Corr. Delays", "Spearmann", "Spr. Delays"]),
    "Test 3: Random trial-based cross-correlation",
    (2, 2),
    lims = [[-1, 1], [0, DELAY_MAX], [-1, 1], [0, DELAY_MAX]],
    draw = True
)




plt.show()
