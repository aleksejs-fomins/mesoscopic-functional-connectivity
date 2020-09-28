'''
This library is concerned with computing probabilities of the number of elements shared by two binary arrays
under index permutation
'''

import numpy as np

from codes.lib.stat.stat_lib import rand_bool_perm, log_likelihood, discrete_cdf_sample, discrete_distr_to_cdf
from codes.lib.stat.comb_lib import log_comb

# Find a probability distribution of the number of entries shared by two binary arrays
# under random permutation of their entries
def emp_distr_binary_shared(nSrc, nTrg, nTot, keyTrue=None):
    if keyTrue is not None and keyTrue>np.min([nSrc, nTrg]):
        raise ValueError(keyTrue, "shared links unreachable for nSrc, nTrg =", nSrc, nTrg)

    distr = np.zeros(nTot + 1)  # There are anywhere between 0 and nTot shared links
    nStep = 0
    while True:
        nStep += 1
        srcConnRand = rand_bool_perm(nSrc, nTot)
        trgConnRand = rand_bool_perm(nTrg, nTot)
        nSharedRand = np.sum(srcConnRand & trgConnRand)

        distrNew = np.copy(distr)
        distrNew[nSharedRand] += 1

        haveConv = np.linalg.norm(distr / nStep - distrNew / (nStep + 1)) < 0.001
        haveTrue = (keyTrue is None) or (distrNew[keyTrue] > 0)

        if haveConv and haveTrue:
            print("converged after", nStep, "steps")
            idxNonZero = distrNew > 0
            distrX = np.where(idxNonZero)[0]  # Values that have non-zero probability
            distrP = distrNew[idxNonZero] / np.sum(distrNew)
            return dict(zip(distrX, distrP))
        distr = distrNew

# Calculate log(sum(exp(arr))) if arr << 0, s.t. exp(arr) is effectively 0 due to floating point roundoff
# https://stackoverflow.com/questions/26436242/calculating-logsum-of-expterms-when-terms-are-very-small
def logexpsum(arr):
    arrFinite = arr[np.isfinite(arr)]
    if len(arrFinite) == 0:
        return -np.inf
    else:
        arrSortDescend = np.sort(arrFinite)[::-1]                 # Sort negative numbers in descending order
        leadTerm = arrSortDescend[0]                              # Select largest (least negative) number
        residual = np.sum(np.exp(arrSortDescend[1:] - leadTerm))  # Subtract that number from all others

        if residual > 1.0E-6:  # If residual is large, it is more accurate to use exact expression
            return leadTerm + np.log(1 + residual)
        else:                  # If residual is small, it is more accurate to use first term of taylor expansion
            return leadTerm + residual


# Find a probability distribution of the number of entries shared by two binary arrays
# under random permutation of their entries
def log_prob_binary_shared(n, nX, nY):
    assert (nX >= 0) and (nY >= 0) and (n >= nX) and (n >= nY)

    # Find range for number of shared elements that are theoretically possible (p > 0)
    nShMin = np.max([0, nX + nY - n])
    nShMax = np.min([nX, nY])

    # Pre-compute number of combinations independent of nSh
    logPnnX = log_comb(n, nX)

    # Compute all non-zero probabilities using combinations
    logP = np.full(n+1, -np.inf, dtype=float)
    for nSh in range(nShMin, nShMax+1):
        logP[nSh] = log_comb(nY, nSh) + log_comb(n - nY, nX - nSh) - logPnnX

    return logP


def log_pval_H0_shared_random(nTotLst, nSrcLst, nTrgLst, nSharedTrueLst):
    rezThis = []
    rezL = []
    rezR = []
    for nTot, nSrc, nTrg, nShTrue in zip(nTotLst, nSrcLst, nTrgLst, nSharedTrueLst):
        logProb = log_prob_binary_shared(nTot, nSrc, nTrg)
        logPThis = logProb[nShTrue]                # Log Prob of getting exactly nSh
        logPValL = logexpsum(logProb[:nShTrue+1])  # Log Prob of getting at most nSh
        logPValR = logexpsum(logProb[nShTrue:])    # Log Prob of getting at least nSh
        rezThis += [logPThis]
        rezL += [logPValL]
        rezR += [logPValR]
    return np.array(rezThis), np.array(rezL), np.array(rezR)



# A certain function computes the number nSh of elements shared by two binary arrays given their length and number of ones in each
# The NULL HYPOTHESIS is that nSh is computed from a random permutation of source and target arrays
#
# Input: Number of total, source, and target points for each observation, as well as the true observed nSh
# Output: p-value of the likelihood of data under H0
#
# p-value is computed as P[Ltrue > Lrand], where L is the negative log-likelihood
# It is evaluated by constructing empirical probability distribution of Lrand and resampling it
# def pval_H0_shared_random_window(nSrcLst, nTrgLst, nTotLst, nSharedTrueLst, nResample=2000):
#     nPoint = len(nSrcLst)
#
#     # 1. Construct empirical distributions for each pair
#     distrLst = [distr_binary_shared(nSrc, nTrg, nTot) for nSrc, nTrg, nTot in zip(nSrcLst, nTrgLst, nTotLst)]
#     cdfLst = [discrete_distr_to_cdf(distr) for distr in distrLst]
#
#     # 2. Find probabilities of true shared links
#     pTrueLst = [distr[nSharedTrue] for distr, nSharedTrue in zip(cdfLst, nSharedTrueLst)]
#
#     # 3. Compute log-likelihood of true observation under null hypothesis
#     logLikelihoodTrue = log_likelihood(pTrueLst)
#
#     # 4. Resample each point, get resampled probabilities
#     pRandResampled = np.array([discrete_resample(cdf, nResample) for cdf in cdfLst])
#
#     # 5. Compute resampled log-likelihoods
#     logLikelihoodResampled = log_likelihood(pRandResampled, axis=1)
#
#     # 6. Find p-value by determining how frequently random LL is smaller than the true one
#     return np.mean((logLikelihoodResampled < logLikelihoodTrue).astype(float))
