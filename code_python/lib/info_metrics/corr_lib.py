import numpy as np
import scipy.stats

from lib.signal_lib import zscore
from lib.array_lib import numpy_merge_dimensions, numpy_transpose_byorder, test_have_dim
from lib.stat.stat_lib import mu_std
from lib.stat.graph_lib import offdiag_1D


def corr_significance(c, nData):
    t = c * np.sqrt((nData - 2) / (1 - c**2))
    t[t == np.nan] = np.inf
    return scipy.stats.t(nData).pdf(t)


# Correlation. Requires leading dimension to be channels
# If y2D not specified, correlation computed between channels of x
# If y2D is specified, correlation computed for x-x and x-y in a composite matrix
def corr_2D(x2D, y2D=None, est='corr'):
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
def corr_3D(data, settings, est='corr'):
    # Convert to canonical form
    test_have_dim("corr3D", settings['dim_order'], "p")
    dataCanon = numpy_transpose_byorder(data, settings['dim_order'], 'psr', augment=True)
    dataFlat = numpy_merge_dimensions(dataCanon, 1, 3)
    return corr_2D(dataFlat, est=est)


# Compute average absolute value off-diagonal correlation (synchr. coeff)
def avg_corr_3D(data, settings, est='corr'):
    M = corr_3D(data, settings, est=est)
    return np.nanmean(np.abs(offdiag_1D(M[0])))


# Calculates the 1-sided autocorrelation of a discrete 1D dataset
# Returned dataset has same length as input, first value normalized to 1.
def autocorr_1D(x):
    N = len(x)

    # FIXME Cheat - replace NAN's with random normal numbers with same mean and variance
    xEff = np.copy(x)
    nanIdx = np.isnan(x)
    xEff[nanIdx] = np.random.normal(*mu_std(x), np.sum(nanIdx))

    #return np.array([np.nanmean(x[iSh:] * x[:N-iSh]) for iSh in range(N)])
    # Note: As of time of writing np.correlate does not correctly handle nan division
    return np.correlate(xEff, xEff, 'full')[N - 1:] / N


# Calculates autocorrelation. Any dimensions
# TODO: Currently autocorrelation is averaged over other provided dimensions. Check if there is a more rigorous way
def autocorr_3D(data, settings):
    test_have_dim("autocorr_3D", settings['dim_order'], "s")

    # Convert to canonical form
    dataCanon = numpy_transpose_byorder(data, settings['dim_order'], 'rps', augment=True)
    dataFlat = numpy_merge_dimensions(dataCanon, 0, 2)
    dataThis = zscore(dataFlat)
    return np.nanmean(np.array([autocorr_1D(d) for d in dataThis]), axis=0)


# Calculates autocorrelation of unit time shift. Can handle nan's
def autocorr_d1_3D(data, settings):
    test_have_dim("autocorr_d1_3D", settings['dim_order'], "s")

    # Convert to canonical form
    dataCanon = numpy_transpose_byorder(data, settings['dim_order'], 'srp', augment=True)
    dataZ = zscore(dataCanon)

    dataZpre  = dataZ[:-1].flatten()
    dataZpost = dataZ[1:].flatten()
    return np.nanmean(dataZpre * dataZpost)


# FIXME: Correct all TE-based procedures, to compute cross-correlation as a window sweep externally
def cross_corr_3D(data, settings, est='corr'):
    '''
    Compute cross-correlation of multivariate dataset for a fixed lag

    :param data: 2D or 3D matrix
    :param settings: A dictionary. 'min_lag_sources' and 'max_lag_sources' determine lag range.
    :param est: Estimator name. Can be 'corr' or 'spr' for cross-correlation or spearmann-rank
    :return: A matrix [3 x nSource x nTarget], where 3 stands for [Corr Value, Max Lag, p-value]
    '''

    # Test that necessary dimensions have been provided
    test_have_dim("autocorr_3D", settings['dim_order'], "p")
    test_have_dim("autocorr_3D", settings['dim_order'], "s")

    # Transpose dataset into comfortable form
    dataOrd = numpy_transpose_byorder(data, settings['dim_order'], 'psr', augment=True)  # add trials dimension for simplicity

    # Extract parameters
    nNode, nTime = dataOrd.shape[:2]
    lagMin = settings['min_lag_sources']
    lagMax = settings['max_lag_sources']

    # Check that number of timesteps is sufficient to estimate lagMax
    if nTime <= lagMax:
        raise ValueError('lag', lagMax, 'cannot be estimated for number of timesteps', nTime)

    nLag = lagMax - lagMin + 1
    rezMat = np.zeros((2, nLag, nNode, nNode))

    for iLag in range(nLag):
        lag = lagMin + iLag
        xx = numpy_merge_dimensions(dataOrd[:, lag:], 1, 3)
        yy = numpy_merge_dimensions(dataOrd[:, :nTime-lag], 1, 3)
        corrThis, pThis = corr_2D(xx, yy, est=est)

        # Only interested in x-y correlations, crop x-x and y-y
        rezMat[0, iLag] = corrThis[nNode:, :nNode]
        rezMat[1, iLag] = pThis[nNode:, :nNode]
                        
    return rezMat


# Correlation that works if some values in the dataset are NANs
def corr_nan(x2D):
    pass
    # z2D = zscore(x2D, axis=1)
    # nChannel, nData = x2D.shape
    # rez = np.ones((nChannel, nChannel))
    # for i in range(nChannel):
    #     for j in range(i+1, nChannel):
    #         rez[i][j] = np.nanmean(z2D[i] * z2D[j])
    #         rez[j][i] = rez[i][j]
    # return rez