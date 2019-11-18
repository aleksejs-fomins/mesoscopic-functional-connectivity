import numpy as np

from codes.lib.metrics.graph_lib import offdiag_idx


# Given number of elements in a subset and total set, return their ratio
# If both are zero, return zero
# Complain if values are inadequate
def rate_0_protected(nArrSub, nArrTot):
    assert np.sum(nArrSub < 0) == 0,       "Only non-negative arrays allowed"
    assert np.sum(nArrTot < 0) == 0,       "Only non-negative arrays allowed"
    assert np.sum(nArrTot < nArrSub) == 0, "Subset can't have greater size than the whole set"

    nTotZeroIdx = nArrTot == 0
    rez = np.copy(nArrSub)
    rez[nTotZeroIdx] /= nArrTot[nTotZeroIdx]
    return rez


def accuracyIndices(yHat, y):
    if yHat.shape == y.shape:
        yReal = y
    elif (yHat.ndim == 2) and (y.ndim == 1) and (yHat.shape[0] == y.shape[0]):
        # If correct output is the same for all tests, extrude it along the second dimensions
        yReal = np.repeat([y], yHat.shape[1], axis=0).T
    else:
        raise ValueError("Unexpected shapes", yHat.shape, y.shape)

    idxsDict = {
        'TP' :  yHat &  yReal,  # True Positives
        'FP' :  yHat & ~yReal,  # False Positives
        'FN' : ~yHat &  yReal,  # False Negatives
        'TN' : ~yHat & ~yReal   # True Negatives
    }

    return yReal, idxsDict



def accuracyTests(yHat, y, axis=0):
    yReal, idxsDict = accuracyIndices(yHat, y)

    freqTrue     = np.mean(yReal, axis=axis)      # Frequency of true outcomes
    freqFalse    = np.mean(yHat, axis=axis)       # Frequency of false outcomes
    freqHatTrue  = np.mean(yReal, axis=axis)      # Frequency of true predictions
    freqHatFalse = np.mean(yHat, axis=axis)       # Frequency of false predictions

    freqTP = np.mean(idxsDict['TP'], axis=axis)   # Frequency of True Positives
    freqFP = np.mean(idxsDict['FP'], axis=axis)   # Frequency of False Positives
    freqFN = np.mean(idxsDict['FN'], axis=axis)   # Frequency of False Negatives
    freqTN = np.mean(idxsDict['TN'], axis=axis)   # Frequency of True Negatives

    TPR = rate_0_protected(freqTP, freqTrue)   # True Positive Rate - number of TP out of all that were possible
    FPR = rate_0_protected(freqFP, freqFalse)  # False Positive Rate - number of FP out of all that were possible

    return {
        "TP frequency": freqTP,
        "FP frequency": freqFP,
        "FN frequency": freqFN,
        "TN frequency": freqTN,
        "TPR": TPR,
        "FPR": FPR
    }