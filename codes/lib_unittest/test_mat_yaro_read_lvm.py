import sys
import numpy as np
import matplotlib.pyplot as plt

# Export library path
import os, sys
thispath = os.path.dirname(os.path.abspath(__file__))
parpath = os.path.abspath(os.path.join(thispath, os.pardir))
sys.path.append(os.path.join(parpath, 'lib/'))

from matlab.matlab_yaro_lib import read_lvm

# Read LVM file from command line
inputpath = sys.argv[1]
data2D = read_lvm(inputpath)

# Plot it
plt.figure()
for i in range(data2D.shape[0]):
    plt.plot(data2D[i], label="channel_"+str(i))
plt.legend()
plt.show()
