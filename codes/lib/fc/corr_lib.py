import numpy as np
import scipy.stats

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


# Compute cross-correlation for all shifts in [delay_min, delay_max]
# For each pair return largest corr by absolute value, and shift value
# * Data - 2D matrix [channel x time]
# * Data - 3D matrix [channel x time x trial]
#   For fixed lag, trial and channel pair:
#    - shift time by lag and get paired overlap
#    - concatenate overlaps for all trials
#    - compute corr over concatenation
def crossCorr(data, lagMin, lagMax, est='corr'):
    nNode = data.shape[0]
    nTime = data.shape[1]
    haveTrials = len(data.shape) > 2
        
    # Check that number of timesteps is sufficient to estimate lagMax
    if nTime <= lagMax:
        raise ValueError('max lag', lagMax, 'cannot be estimated for number of timesteps', nTime)
        
    matCorr = np.eye(nNode)
    matLag = np.zeros((nNode, nNode))
    matP = np.zeros((nNode, nNode))
    
    for lag in range(lagMin, lagMax+1):
        xx = data[:, lag:]
        yy = data[:, :nTime-lag]

        if haveTrials:
            corrThis, pThis = corr3D(xx, yy, est=est)
        else:
            corrThis, pThis = corr(xx, yy, est=est)

        # Only interested in x-y correlations, crop x-x and y-y
        corrThis = corrThis[nNode:, :nNode]
        pThis    = pThis[nNode:, :nNode]

        # Keep estimate and lag if it has largest absolute value so far
        idxCorr = np.abs(corrThis) > np.abs(matCorr)
        matCorr[idxCorr] = corrThis[idxCorr]
        matLag[idxCorr] = lag
        matP[idxCorr] = pThis[idxCorr]
                        
    return matCorr, matLag, matP