import numpy as np
import scipy.stats

# Correlation. Requires leading dimension to be channels
def corr(x2D, y2D=None, est='corr'):
    if est == 'corr':
        return np.corrcoef(x2D, y2D)
    elif est == 'spr':
        return scipy.stats.spearmanr(x2D, y2D, axis=1)[0]
    else:
        raise ValueError('unexpected estimator type', est)

# Compute cross-correlation for all shifts in [delay_min, delay_max]
# For each pair return largest corr by absolute value, and shift value
# * Data - 2D matrix [channel x time]
# * Data - 3D matrix [channel x time x trial]
#   For fixed delay, trial and channel pair:
#    - shift time by delay and get paired overlap
#    - concatenate overlaps for all trials
#    - compute corr over concatenation
def crossCorr(data, delayMin, delayMax, est='corr'):
    nNode = data.shape[0]
    nTime = data.shape[1]
    haveTrials = len(data.shape) > 2
        
    # Check that number of timesteps is sufficient to estimate delayMax
    if nTime <= delayMax:
        raise ValueError('max delay', delayMax, 'cannot be estimated for number of timesteps', nTime)
        
    corrMat = np.zeros((nNode, nNode))
    delayMat = np.zeros((nNode, nNode))
    
    for delay in range(delayMin, delayMax+1):
        xx = data[:, :nTime-delay]
        yy = data[:, delay:]

        if haveTrials:
            shape2D = (nNode, np.prod(xx.shape[1:]))
            xx = xx.reshape(shape2D)
            yy = yy.reshape(shape2D)

        # Choose between Correlation and Spearman Rank estimators
        corrThis = corr(xx, yy, est=est)[nNode:, :nNode]

        # Keep estimate and delay if it has largest absolute value so far
        idxCorr = np.abs(corrThis) > np.abs(corrMat)
        corrMat[idxCorr] = corrThis[idxCorr]
        delayMat[idxCorr] = delay
                        
    return corrMat, delayMat