import numpy as np

# Construct data with progressively more noise
# Data must be normalized to maximum prior to this transform
def makedata_snr_observational(data, paramRanges):
    return [snr * data + (1 - snr) * np.random.normal(0, 1, data.shape) for snr in paramRanges]


# Construct data with progressively more pure noise trials
# Data must be normalized to maximum prior to this transform
# First data axis must be trials
def makedata_snr_occurence(data, paramRanges):
    rez = np.zeros((len(paramRanges),) + data.shape)
    nTrial = data.shape[0]

    for i, freq in enumerate(paramRanges):
        nTrialTrue = int(freq * nTrial)
        nTrialNoise = nTrial - nTrialTrue
        dataPart = data[:nTrialTrue]
        randPart = np.random.normal(0, 1, (nTrialNoise,) + data.shape[1:])
        rez[i] = np.concatenate((dataPart,randPart), axis=0)

    return rez