
import numpy as np
import matplotlib.pyplot as plt

A = np.abs(np.random.normal(0,1,100))
B = np.abs(np.random.normal(0,1,100))

def CDF(x):
    return np.sort(x), np.linspace(0,1,len(x))

def cranksum(x, y):
    M, N = len(x), len(y)
    jointData = np.hstack([x, y])
    jointSign = np.array([1.0/M]*M + [-1.0/N]*N)
    idxOrder = np.argsort(jointData)
    ranks = np.arange(1, M+N+1).astype(float)
    ranks *= jointSign[idxOrder]

    rankSum = np.flip(ranks)
    for i in range(1, M+N):
        rankSum[i] += rankSum[i-1]
    return rankSum

xa, ca = CDF(A)
xb, cb = CDF(B)

fig, ax = plt.subplots(ncols=2)
ax[0].plot(xa, ca)
ax[0].plot(xb, cb)
ax[1].plot(cranksum(A, B))
plt.show()
