import numpy as np


def bonferroni_correction(pMat, nHypothesis):
    # According to Bonferroni, p-value is a multiple of number of hypotheses that have been tested
    # However, p-value may not exceed 1, so crop it to 1. It is probably not precise for large p-values,
    #  but that is also irrelevant, because hypotheses with large p-values would be rejected anyway
    pMatCorr = pMat * nHypothesis
    pMatCorr[pMatCorr > 1] = 1
    return pMatCorr