import os, sys
import numpy as np
import matplotlib.pyplot as plt

# Export library path
rootname = "mesoscopic-functional-connectivity"
thispath = os.path.dirname(os.path.abspath(__file__))
rootpath = os.path.join(thispath[:thispath.index(rootname)], rootname)
print("Appending project path", rootpath)
sys.path.append(rootpath)

from codes.lib.data_io.yaro.yaro_data_read import read_neuro_perf

# Read LVM file from command line
inputpath = sys.argv[1]
data, behaviour, performance = read_neuro_perf(inputpath)

nTrials, nTimes, nChannels = data.shape

print("Loaded neuronal data with (nTrials, nTimes, nChannels)=", data.shape)

tlst = 50*np.linspace(0, nTimes, nTimes)

fig, ax = plt.subplots(ncols=2, tight_layout=True)
for i in range(nChannels):
    act = np.mean(data[:,:,i], axis=0)
    err = np.std(data[:,:,i], axis=0, ddof=1) / np.sqrt(nTrials)
    ax[0].plot(tlst, act, label="channel_"+str(i))
    ax[0].fill_between(tlst, act-err, act+err, alpha=0.2)

ax[0].set_title('Mean activity per channel')
ax[0].set_ylabel('mean activity')
ax[0].set_xlabel('time, ms')
ax[0].legend()

behaviour_keys = ['iGO', 'iNOGO', 'iFA', 'iMISS']
for bkey in behaviour_keys:
    keydata = data[behaviour[bkey]-1,:,0][0]
    nTrialsThis = keydata.shape[0]
    act = np.mean(keydata, axis=0)
    err = np.std(keydata, axis=0, ddof=1) / np.sqrt(nTrialsThis)
    ax[1].plot(tlst, act, label=bkey)
    ax[1].fill_between(tlst, act-err, act+err, alpha=0.2)
    
ax[1].set_title('Mean activity of channel 0 per behaviour')
ax[1].set_ylabel('mean activity')
ax[1].set_xlabel('time, ms')
ax[1].legend()
plt.show()
