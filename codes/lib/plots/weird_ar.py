import numpy as np
import matplotlib.pyplot as plt
from codes.lib.stat.stat_lib import bootstrap_resample_function


def _calc_ar_mle(dataArr):
    x = dataArr[:-1]
    y = dataArr[1:]
    Cxx = np.sum(x*x)
    Cyx = np.sum(x*y)
    Sx = np.sum(x)
    Sy = np.sum(y)
    Nx = np.prod(x.shape)

    return np.linalg.solve(
        np.array([[Cxx, Sx],[Sx, Nx]]),
        np.array([Cyx, Sy])
    )


# Data shape [nTrial, nTime]
def test_ar_decay_const(data, nSample=1000, alphaThr=0.9):
    alphaArr, betaArr = np.array(bootstrap_resample_function(_calc_ar_mle, data, nSample)).T
    print("For nSample =", nSample, "P[a >=", alphaArr, "] =", np.sum(alphaArr > alphaThr))

    fig, ax = plt.subplots(ncols=2, figsize=(10, 5))
    ax[0].hist(alphaArr, bins='auto')
    ax[1].hist(betaArr, bins='auto')
    plt.show()
