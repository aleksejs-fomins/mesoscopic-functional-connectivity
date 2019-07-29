import os, sys
import numpy as np
import matplotlib.pyplot as plt

# Export library path
thispath = os.path.dirname(os.path.abspath(__file__))
parpath = os.path.abspath(os.path.join(thispath, os.pardir))
sys.path.append(os.path.join(parpath, 'lib/'))

from signal_lib import downsample #, downsample_interp

##########################
# Generate Markov Data
##########################

T_RANGE = [0.0, 1.0]
T = T_RANGE[1] - T_RANGE[0]
DT1 = 0.01 # Times
DT2 = 0.05 # Bins
N1 = int(T / DT1) + 1
N2 = 1 + int((N1-1) * DT1 / DT2)

print(N1, N2)

# Create data
t1 = np.linspace(T_RANGE[0], T_RANGE[1], N1)
y1 = np.random.normal(0, 1, N1)
for i in range(1, N1):
    y1[i] += y1[i-1]

##########################
# Downsample
##########################
    
# t2, y2 = downsample_interp(t1, y1, DT2)
t2, y2 = downsample(t1, y1, N2, method="window")
t3, y3 = downsample(t1, y1, N2, method="kernel",     ker_sig2 = DT2**2)
t4, y4 = downsample(t1, y1, N2, method="kernel", ker_sig2 = (DT2/2)**2)
t5, y5 = downsample(t1, y1, N2, method="kernel", ker_sig2 = (DT2/4)**2)

##########################
# Plot
##########################

plt.figure()
plt.plot(t1, y1, '.-', label='orig')
plt.plot(t2, y2, '.-', label='window')
plt.plot(t3, y3, '.-', label="ker, s2=d2^2")
plt.plot(t4, y4, '.-', label="ker, s2=(d2/2)^2")
plt.plot(t5, y5, '.-', label="ker, s2=(d2/4)^2")

plt.legend()
plt.show()