import numpy as np



# Sum of ranks with bias towards largest values
# Can be used to detect if a distribution has longer tail than another
def cranksum(x, y):
    M, N = len(x), len(y)
    jointData = np.hstack([x, y])
    jointSign = np.array([1.0]*M + [-1.0]*N)
    jointNorm = np.array([1.0/M]*M + [1.0/N]*N)
    idxOrder = np.argsort(jointData)
    ranks = np.arange(1, M+N+1).astype(float)**2
    normRanks = ranks * jointNorm[idxOrder]
    signedRanks = normRanks * jointSign[idxOrder]
    signedRankSum = np.sum(signedRanks)

    # Normalize Ranksum by its maximum, which depends on number of datapoints
    if signedRankSum > 0:
        signedRankSum /= np.sum(normRanks[-M:]) - np.sum(normRanks[:N])
    else:
        signedRankSum /= np.sum(normRanks[-N:]) - np.sum(normRanks[:M])

    return signedRankSum