'''
This library is concerned with computing probabilities of the number of elements shared by two binary arrays
under index permutation
'''

import numpy as np

from codes.lib.stat.stat_lib import rand_bool, log_likelihood, discrete_sample, discrete_distr_to_cdf

# Find a probability distribution of the number of entries shared by two binary arrays
# under random permutation of their entries
def emp_distr_binary_shared(nSrc, nTrg, nTot):
    distr = np.zeros(nTot + 1)  # There are anywhere between 0 and nTot shared links
    nStep = 0
    while True:
        nStep += 1
        srcConnRand = rand_bool(nSrc, nTot)
        trgConnRand = rand_bool(nTrg, nTot)
        nSharedRand = np.sum(srcConnRand & trgConnRand)

        distrNew = np.copy(distr)
        distrNew[nSharedRand] += 1

        if np.linalg.norm(distr / nStep - distrNew / (nStep + 1)) < 0.001:
            print("converged after", nStep, "steps")
            idxNonZero = distrNew > 0
            distrX = np.where(idxNonZero)[0]  # Values that have non-zero probability
            distrP = distrNew[idxNonZero] / np.sum(distrNew)
            return dict(zip(distrX, distrP))
        distr = distrNew


# A certain function computes the number nSh of elements shared by two binary arrays given their length and number of ones in each
# The NULL HYPOTHESIS is that nSh is computed from a random permutation of source and target arrays
#
# Input: Number of total, source, and target points for each observation, as well as the true observed nSh
# Output: p-value of the likelihood of data under H0
#
# p-value is computed as P[Ltrue > Lrand], where L is the negative log-likelihood
# It is evaluated by constructing empirical probability distribution of Lrand and resampling it
def pval_H0_shared_random(nSrcLst, nTrgLst, nTotLst, nSharedTrueLst, nResample=2000):
    nPoint = len(nSrcLst)

    # 1. Construct empirical distributions for each pair
    empDistrLst = [emp_distr_binary_shared(nSrc, nTrg, nTot) for nSrc, nTrg, nTot in zip(nSrcLst, nTrgLst, nTotLst)]
    empCDFLst = [discrete_distr_to_cdf(empDistr) for empDistr in empDistrLst]

    # 2. Find probabilities of true shared links
    pTrueLst = [empDistr[nSharedTrue] for empDistr, nSharedTrue in zip(empDistrLst, nSharedTrueLst)]

    # 3. Compute log-likelihood of true observation under null hypothesis
    logLikelihoodTrue = log_likelihood(pTrueLst)

    # 4. Resample each point, get resampled probabilities
    pRandResampled = np.array([discrete_sample(cdf, nResample) for cdf in empCDFLst])

    # 5. Compute resampled log-likelihoods
    logLikelihoodResampled = log_likelihood(pRandResampled, axis=1)

    # 6. Find p-value by determining how frequently random LL is smaller than the true one
    return np.mean((logLikelihoodResampled < logLikelihoodTrue).astype(float))
