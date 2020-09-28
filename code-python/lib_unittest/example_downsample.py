'''
VERY IMPORTANT:
DO NOT USE SAME PROCEDURE FOR UPSAMPLING AND DOWNSAMPLING

* When we upsample, we want to interpolate data
* When we downsample, we want to sample the trendline, not individual fluctuations

[] Maybe impl FFT-based resampling, that joins both
'''

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
from codes.lib.signal_lib import resample

##########################
# Downsampling
##########################

# Create data
T  = 10     # s
DT = 0.001  # s
t1 = np.arange(0, T, DT)
y1 = np.random.normal(0, 1, t1.size)
for i in range(1, t1.size):
    y1[i] += y1[i-1]

# Resample
DT2 = 0.1   # s
t2 = np.arange(0, t1[-1], DT2)
y2 = resample(t1, y1, t2, {"method" : "smooth", "kind" : "window"})
y3 = resample(t1, y1, t2, {"method" : "smooth", "kind" : "kernel", "ker_sig2" : DT2**2})
y4 = resample(t1, y1, t2, {"method" : "smooth", "kind" : "kernel", "ker_sig2" : (DT2/2)**2})
y5 = resample(t1, y1, t2, {"method" : "smooth", "kind" : "kernel", "ker_sig2" : (DT2/4)**2})

# Plot
plt.figure()
plt.title('Downsampling using window and kernel estimators')
plt.plot(t1, y1, '.-', label='orig')
plt.plot(t2, y2, '.-', label='window')
plt.plot(t2, y3, '.-', label="ker, s2=d2^2")
plt.plot(t2, y4, '.-', label="ker, s2=(d2/2)^2")
plt.plot(t2, y5, '.-', label="ker, s2=(d2/4)^2")
plt.legend()

##########################
# Upsampling
##########################

# Create Data
T  = 10     # s
DT = 0.1    # s
t1 = np.arange(0, T, DT)
y1 = np.sin(10*t1) * np.sin(2*t1)

# Resample
DT2 = 0.01  # s
t2 = np.arange(0, t1[-1], DT2)
y2 = resample(t1, y1, t2, {"method" : "interpolative", "kind" : "linear"})
y3 = resample(t1, y1, t2, {"method" : "interpolative", "kind" : "quadratic"})
y4 = resample(t1, y1, t2, {"method" : "interpolative", "kind" : "cubic"})

# Plot
plt.figure()
plt.title('Upsampling using interpolation')
plt.plot(t1, y1, 'o', label='orig')
plt.plot(t2, y2, '.-', label='linear')
plt.plot(t2, y3, '.-', label="quadratic")
plt.plot(t2, y4, '.-', label="cubic")
plt.legend()

plt.show()