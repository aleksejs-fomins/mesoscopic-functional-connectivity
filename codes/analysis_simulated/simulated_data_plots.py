import os, sys
import numpy as np
import matplotlib.pyplot as plt

# Export library path
rootname = "mesoscopic-functional-connectivity"
thispath   = os.path.dirname(os.path.abspath(__file__))
rootpath = os.path.join(thispath[:thispath.index(rootname)], rootname)
print("Appending project path", rootpath)
sys.path.append(rootpath)

from codes.lib.data_io.qt_wrapper import gui_fname
from codes.lib.analysis.simulated_reader import read_data_h5

# Open data filename
fname = gui_fname("Open simulated data file", "./", "HDF5 (*.h5)")

# Read file
data, trueConn = read_data_h5(fname)
nTrial, nNode, nData = data.shape


# Plot results
iTrial = 10
fig, ax = plt.subplots()
ax[0].set_title("Node activities during trial " + str(iTrial))
ax[1].set_title("True Connectivity")

for iNode in range(nNode):
    ax[0].plot(data[iTrial, iNode], label=str(iNode))

ax[1].legend()
ax[1].imshow(trueConn)
plt.show()