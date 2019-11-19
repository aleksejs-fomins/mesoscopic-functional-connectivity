# Standard libraries
import os, sys
import numpy as np
import matplotlib.pyplot as plt
import h5py

# Export library path
rootname = "mesoscopic-functional-connectivity"
thispath   = os.path.dirname(os.path.abspath(__file__))
rootpath = os.path.join(thispath[:thispath.index(rootname)], rootname)
print("Appending project path", rootpath)
sys.path.append(rootpath)

# Local libraries
from codes.lib.fc.corr_lib import crossCorr
from codes.lib.plots.matrix import plotMatrix


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
    data = rez_file_h5['data']

    #######################
    #  Cross-correlation
    #######################

    DELAY_MIN = 1
    DELAY_MAX = 6

    corrMat, corrDelMat = crossCorr(data, DELAY_MIN, DELAY_MAX, est='corr')
    sprMat,  sprDelMat = crossCorr(data, DELAY_MIN, DELAY_MAX, est='spr')

    rez_file_h5.close()

    #######################
    #  Plot
    #######################

    plotMatrix(
        [corrMat, corrDelMat, sprMat,  sprDelMat],
        ['Corr', 'Spr', 'CorrDel', 'SprDel'],
        "Cross-correlation for " + filename,
        shape = (2, 2),
        lims = [[-1,1], [0, DELAY_MAX], [-1,1], [0, DELAY_MAX]],
        draw=True,
        savename = filename.split('.')[0] + '.png'
    )




########################
##  Cross-correlation with velocities
########################

#data2 = np.vstack((DS1.data[:, :-1], DS1.data[:, 1:] - DS1.data[:, :-1]))

#print(data2.shape)

#corrMat, corrDelMat = crossCorr(data2, DELAY_MIN, DELAY_MAX, est='corr')
#sprMat,  sprDelMat = crossCorr(data2, DELAY_MIN, DELAY_MAX, est='spr')

#plotMatrix(
    #[[corrMat, corrDelMat], [sprMat,  sprDelMat]],
    #np.array([['Corr', 'Spr'],['CorrDel', 'SprDel']]),
    #"Cross-correlation for DynSys - with velocities",
    #lims = [[[-1,1], [0, DELAY_MAX]], [[-1,1], [0, DELAY_MAX]]],
    #draw=True
#)


plt.show()
