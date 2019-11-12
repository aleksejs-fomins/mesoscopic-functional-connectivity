import numpy as np
import matplotlib.pyplot as plt

'''
Shifted Noise
'''

def shifted_noise(nT):
    data = np.zeros((nT, 2), dtype=float)
    data[:, 0] = np.random.normal(0, 1, nT)
    data[1:, 1] = data[:-1, 0] + 0.2 * np.random.normal(0, 1, nT-1)
    return data

def shifted_noise_sin(nT):
    data = np.zeros((nT, 2), dtype=float)
    data[:, 0] = np.random.uniform(0, 1, nT)
    data[1:, 1] = np.sin(2 * np.pi * data[:-1, 0]) + 0.2 * np.random.normal(0, 1, nT-1)
    return data

nTest = 2

testsDict = {
    "ShiftedNoise"   : shifted_noise,
    "ShiftedNoiseSin" : shifted_noise_sin
}


nT = 1000
for iTest, (testName, testFunc) in enumerate(testsDict.items()):
    data = testFunc(nT)

    fig, ax = plt.subplots(ncols=5)
    fig.suptitle(testName)
    ax[0].plot(data[:-1, 0], data[1:, 0], '.')
    ax[1].plot(data[:-1, 1], data[1:, 1], '.')
    ax[2].plot(data[:, 0],   data[:, 1], '.')
    ax[3].plot(data[:-1, 0], data[1:, 1], '.')
    ax[4].plot(data[:-1, 1], data[1:, 0], '.')

    ax[0].set_title("Self-prediction, data1")
    ax[1].set_title("Self-prediction, data2")
    ax[2].set_title("cross-prediction, lag=0")
    ax[3].set_title("cross-prediction 1->2, lag=1")
    ax[4].set_title("cross-prediction 2->1, lag=1")

plt.show()