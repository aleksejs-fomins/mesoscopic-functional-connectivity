import sys
import matplotlib.pyplot as plt

from lib.data_io.data_read import read_lvm

# Read LVM file from command line
inputpath = sys.argv[1]
data2D = read_lvm(inputpath)

# Plot it
plt.figure()
for i in range(data2D.shape[0]):
    plt.plot(data2D[i], label="channel_"+str(i))
plt.legend()
plt.show()
