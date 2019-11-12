# import standard libraries
import sys
from os.path import dirname, abspath, join
import numpy as np
import matplotlib.pyplot as plt

# Export library path
thispath   = dirname(abspath(__file__))
parentpath = dirname(thispath)
rootpath = dirname(parentpath)
sys.path.append(rootpath)

# import special libraries
from codes.lib.signal_lib import approxDelayConv

# Create signal
DT =  0.001   # s
TAU = 0.1     # s
t = np.arange(0, 10, DT)
y = (np.sin(1 * t)**2 > 0.7).astype(float)

yc = approxDelayConv(y, TAU, DT)

plt.figure()
plt.plot(t, y)
plt.plot(t, yc)
plt.show()