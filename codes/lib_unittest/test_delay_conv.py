import os, sys
import numpy as np
import matplotlib.pyplot as plt

# Export library path
thispath = os.path.dirname(os.path.abspath(__file__))
p1path = os.path.abspath(os.path.join(thispath, os.pardir))
p2path = os.path.abspath(os.path.join(p1path, os.pardir))
sys.path.append(os.path.join(p1path, 'lib/'))

# Locate results path
rezPath = os.path.join(os.path.join(p2path, 'data/'), 'sim-ds-py')

from signal_lib import approxDelayConv


# Create signal
DT = 0.001
TAU = 0.1
t = np.arange(0, 1, DT)
y = (np.sin(10 * t)**2 > 0.9).astype(float)

yc = approxDelayConv(y, TAU, DT)

plt.figure()
plt.plot(t, y)
plt.plot(t, yc)
plt.show()