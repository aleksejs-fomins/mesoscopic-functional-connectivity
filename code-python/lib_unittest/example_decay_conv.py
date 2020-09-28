# import standard libraries
import os, sys
import numpy as np
import matplotlib.pyplot as plt

# Export library path
rootname = "mesoscopic-functional-connectivity"
thispath   = os.path.dirname(os.path.abspath(__file__))
rootpath = os.path.join(thispath[:thispath.index(rootname)], rootname)
print("Appending project path", rootpath)
sys.path.append(rootpath)

# import special libraries
from codes.lib.signal_lib import approx_decay_conv

# Create signal
DT =  0.001   # s
TAU = 0.1     # s
t = np.arange(0, 10, DT)
y = (np.sin(1 * t)**2 > 0.7).astype(float)

yc = approx_decay_conv(y, TAU, DT)

plt.figure()
plt.plot(t, y)
plt.plot(t, yc)
plt.show()