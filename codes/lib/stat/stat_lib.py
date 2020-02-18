import numpy as np
from bisect import bisect_left


def gaussian(mu, s2):
    return np.exp(- mu**2 / (2 * s2))

# Combined estimate of mean and variance. Excludes nan values
def mu_std(x, axis=None):
    return np.nanmean(x, axis=axis), np.nanstd(x, axis=axis)


# Computes 1D empirical tolerance interval by determining (1 - sigma)/2 percentile
def tolerance_interval_1D(x):
    xNoNan = x[~np.isnan(x)]
    p = (1 - 0.68) / 2       #  percentile equivalent to 1-sigma gaussian interval
    return np.percentile(xNoNan, [p, 1-p])


# According to Bonferroni, p-value is a multiple of number of hypotheses that have been tested
# However, p-value may not exceed 1, so crop it to 1. It is probably not precise for large p-values,
#  but that is also irrelevant, because hypotheses with large p-values would be rejected anyway
def bonferroni_correction(pMat, nHypothesis):
    pMatCorr = pMat * nHypothesis
    pMatCorr[pMatCorr > 1] = 1
    return pMatCorr


# Construct a 1D random array that has an exact number of ones and zeroes
def rand_bool_perm(nTrue, nTot):
    rv = np.random.uniform(0, 1, nTot)
    return rv < np.sort(rv)[nTrue]


# Compute log-likelihood from probabilities of independent observations
def log_likelihood(pLst, axis=0):
    return -2 * np.sum(np.log(pLst), axis=axis)


# Construct CDF from discrete distribution
# Has same length as PDF, first probability non-zero, last probability is 1
def discrete_distr_to_cdf(distr):
    keysSorted = sorted(distr.keys())
    cdfP = np.array([distr[k] for k in keysSorted])
    for i in range(1, cdfP.size):
        cdfP[i] += cdfP[i - 1]
    return dict(zip(keysSorted, cdfP))


# Construct a PDF from a discrete sample. No binning - items must match exactly
def discrete_empirical_pdf_from_sample(sample):
    keys, vals = np.unique(sample, return_counts=True)
    vals = vals.astype(float) / np.sum(vals)
    return dict(zip(keys, vals))


# Draw N samples from a discrete probability distribution
def discrete_cdf_sample(cdf, nSample):
    cdfX = np.array(list(cdf.keys()))
    cdfP = np.array(list(cdf.values()))

    urand = np.random.uniform(0, 1, nSample)
    bisectIdxs = [bisect_left(cdfP, p) for p in urand]
    return cdfX[bisectIdxs]
