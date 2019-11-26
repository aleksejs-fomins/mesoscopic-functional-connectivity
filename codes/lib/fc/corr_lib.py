import numpy as np
import scipy.stats

from codes.lib.aux_functions import perm_map_str
from codes.lib.stat_lib import bonferroni_correction

def corr_significance(c, nData):
    t = c * np.sqrt((nData - 2) / (1 - c**2))
    t[t == np.nan] = np.inf
    return scipy.stats.t(nData).pdf(t)


# Correlation. Requires leading dimension to be channels
# If y2D not specified, correlation computed between channels of x
# If y2D is specified, correlation computed for x-x and x-y in a composite matrix
def corr(x2D, y2D=None, est='corr'):
    nChannel, nData = x2D.shape
    if est == 'corr':
        c = np.corrcoef(x2D, y2D)
        p = corr_significance(c, nData)
        return [c, p]
    elif est == 'spr':
        spr, p = scipy.stats.spearmanr(x2D, y2D, axis=1)

        # SPR has this "great" idea of only returning 1 number if exactly 2 channels are used
        if (nChannel == 2) and (y2D is None):
            coeff2mat = lambda d, c: np.array([[d, c],[c, d]])
            return [coeff2mat(1, spr), coeff2mat(0, p)]
        else:
            return [spr, p]
    else:
        raise ValueError('unexpected estimator type', est)


# If data has trials, concatenate trials into single timeline when computing correlation
# Data - 3D matrix [channel x time x trial]
def corr3D(x3D, y3D=None, est='corr'):
    shape2D = (x3D.shape[0], np.prod(x3D.shape[1:]))
    x2D = x3D.reshape(shape2D)
    y2D = y3D.reshape(shape2D) if y3D is not None else None
    return corr(x2D, y2D, est=est)


def crossCorr(data, settings, est='corr'):
    '''
    Compute cross-correlation of multivariate dataset for a range of lags

    :param data: 2D or 3D matrix
    :param settings: A dictionary. 'min_lag_sources' and 'max_lag_sources' determine lag range.
    :param est: Estimator name. Can be 'corr' or 'spr' for cross-correlation or spearmann-rank
    :return: A matrix [3 x nSource x nTarget], where 3 stands for [Corr Value, Max Lag, p-value]
    '''

    # Parse settings
    haveTrials = len(data.shape) > 2
    lagMin = settings['min_lag_sources']
    lagMax = settings['max_lag_sources']

    # Transpose dataset into comfortable form
    if haveTrials:
        dataOrd = data.transpose(perm_map_str(settings['dim_order'], 'psr'))
    else:
        dataOrd = data.transpose(perm_map_str(settings['dim_order'], 'ps'))

    # Extract dimensions
    nNode, nTime = dataOrd.shape[:2]

    # Check that number of timesteps is sufficient to estimate lagMax
    if nTime <= lagMax:
        raise ValueError('max lag', lagMax, 'cannot be estimated for number of timesteps', nTime)

    rezMat = np.zeros((3, nNode, nNode))
    
    for lag in range(lagMin, lagMax+1):
        xx = dataOrd[:, lag:]
        yy = dataOrd[:, :nTime-lag]

        if haveTrials:
            corrThis, pThis = corr3D(xx, yy, est=est)
        else:
            corrThis, pThis = corr(xx, yy, est=est)

        # Only interested in x-y correlations, crop x-x and y-y
        corrThis = corrThis[nNode:, :nNode]
        pThis    = pThis[nNode:, :nNode]

        # Keep estimate and lag if it has largest absolute value so far
        idxCorr = np.abs(corrThis) > np.abs(rezMat[0])
        rezMat[0, idxCorr] = corrThis[idxCorr]
        rezMat[1, idxCorr] = lag
        rezMat[2, idxCorr] = pThis[idxCorr]

    # Apply bonferroni correction due to multiple lags test
    nHypothesesLag = lagMax - lagMin + 1
    rezMat[2] = bonferroni_correction(rezMat[2], nHypothesesLag)
                        
    return rezMat