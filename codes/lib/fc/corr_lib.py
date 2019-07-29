import numpy as np
import scipy.stats

# Compute correlation matrix of 3D data
def corr3D(data):
    nTrials, nTimes, nChannels = data.shape
    cr = np.zeros((nChannels, nChannels))
    for iCh in range(nChannels):
        for jCh in range(iCh + 1, nChannels):
            cr[iCh, jCh] = np.corrcoef(data[:,:,iCh].flatten(), data[:,:,jCh].flatten())[0, 1]

    cr += cr.T
    cr += np.diag(np.ones(nChannels))
    return cr

# Spearmann rank matrix
# Data - 2D matrix [channel x time]
def sprMat(data):
    N = data.shape[0]
    rezMat = np.diag(np.ones(N))

    for i in range(N):
        for j in range(i+1, N):
            spr = scipy.stats.spearmanr(data[i], data[j])[0]
            rezMat[i][j] = spr
            rezMat[j][i] = spr
    return rezMat


# Compute cross-correlation for all shifts in [delay_min, delay_max]
# For each pair return largest corr by absolute value, and shift value
# * Data - 2D matrix [channel x time]
# * Data - 3D matrix [trial x max_delay+1 x channel]
#   For fixed delay, trial and channel pair:
#    - shift time by delay and get paired overlap
#    - concatenate overlaps for all trials
#    - compute corr over concatenation
def crossCorr(data, delayMin, delayMax, est='corr'):
    if len(data.shape) == 2:
        nNode, nTime = data.shape
        haveTrials = False
    elif len(data.shape) == 3:
        nTrial, nTime, nNode = data.shape
        haveTrials = True
    else:
        raise ValueError('unexpected data shape', data.shape)
        
    # Check that number of timesteps is sufficient to estimate delayMax
    if nTime <= delayMax:
        raise ValueError('max delay', delayMax, 'cannot be estimated for number of timesteps', nTime)
        
    corrMat = np.zeros((nNode, nNode))
    delayMat = np.zeros((nNode, nNode))
    
    for i in range(nNode):
        for j in range(nNode):
            for delay in range(delayMin, delayMax+1):
                # Based on data type, select lagged variables to cross-correlate
                l_sh = delay
                r_sh = -delay if delay > 0 else nTime
                
                if haveTrials:
                    x = data[:, :r_sh, i].flatten()
                    y = data[:, l_sh:, j].flatten()
                else:
                    x = data[i][:r_sh]
                    y = data[j][l_sh:]
                
                # Choose between Correlation and Spearman Rank estimators
                if est == 'corr':
                    corrThis = np.corrcoef(x, y)[0, 1]
                elif est == 'spr':
                    corrThis = scipy.stats.spearmanr(x, y)[0]
                else:
                    raise ValueError('unexpected estimator type', est)
                    
                # Keep estimate and delay if it has largest absolute value so far
                if np.abs(corrThis) > np.abs(corrMat[i][j]):
                    corrMat[i][j] = corrThis
                    delayMat[i][j] = delay
                        
    return corrMat, delayMat