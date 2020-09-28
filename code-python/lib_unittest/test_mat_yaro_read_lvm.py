import os, sys
import numpy as np
import matplotlib.pyplot as plt

# Export library path
rootname = "mesoscopic-functional-connectivity"
thispath = os.path.dirname(os.path.abspath(__file__))
rootpath = os.path.join(thispath[:thispath.index(rootname)], rootname)
print("Appending project path", rootpath)
sys.path.append(rootpath)

from codes.lib.data_io.yaro.yaro_data_read import read_lvm

# Read LVM file from command line
inputpath = sys.argv[1]
data2D = read_lvm(inputpath)

# Plot it
plt.figure()
for i in range(data2D.shape[0]):
    plt.plot(data2D[i], label="channel_"+str(i))
plt.legend()
plt.show()
