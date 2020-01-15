import numpy as np
from bisect import bisect_left


# According to Bonferroni, p-value is a multiple of number of hypotheses that have been tested
# However, p-value may not exceed 1, so crop it to 1. It is probably not precise for large p-values,
#  but that is also irrelevant, because hypotheses with large p-values would be rejected anyway
def bonferroni_correction(pMat, nHypothesis):
    pMatCorr = pMat * nHypothesis
    pMatCorr[pMatCorr > 1] = 1
    return pMatCorr


# Construct a 1D random array that has an exact number of ones and zeroes
def rand_bool(nTrue, nTot):
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
def discrete_resample(cdf, nSample):
    cdfX = np.array(list(cdf.keys()))
    cdfP = np.array(list(cdf.values()))

    urand = np.random.uniform(0, 1, nSample)
    bisectIdxs = [bisect_left(cdfP, p) for p in urand]
    return cdfX[bisectIdxs]


# Resample a function of 1D data array using bootstrap method
def bootstrap_resample_function(f, data1D, nSample):
    nData = len(data1D)
    fRes = [f(data1D[np.random.randint(0, nData, nData)]) for i in range(nSample)]
    return np.mean(fRes), np.std(fRes)