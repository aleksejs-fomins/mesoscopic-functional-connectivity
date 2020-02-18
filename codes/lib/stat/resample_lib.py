import numpy as np

def bootstrap_1_sample(x):
    nData = x.shape[0]
    return x[np.random.randint(0, nData, nData)]

def permutation_1_sample(x):
    nData = x.shape[0]
    return x[np.random.permutation(nData)]

def _get_1_sample_method(method):
    if method == 'bootstrap':
        return bootstrap_1_sample
    elif method == 'permutation':
        return permutation_1_sample
    else:
        raise ValueError("Unknown method", method)

def resample_monad(f, x, nSample=2000, method="permutation"):
    sampleFunc = _get_1_sample_method(method)
    return np.array([f(sampleFunc(x)) for i in range(nSample)])

# Dyad     ::: f is a function of two variables
# Union    ::: X and Y are resampled from their union
def resample_dyad_union(f,x,y, nSample=2000, method="permutation"):
    M, N = x.shape[0], y.shape[0]
    fmerged = lambda xy : f(xy[:M], xy[M:])
    return resample_monad(fmerged, np.hstack([x, y]), nSample=nSample, method=method)

# Dyad         ::: f is a function of two variables
# Individual   ::: X and Y are resampled from their union
def resample_dyad_individual(f,x,y, nSample=2000, method="permutation"):
    M, N = x.shape[0], y.shape[0]

    if method == 'bootstrap':
        fEff = lambda x, y: f(bootstrap_1_sample(x), bootstrap_1_sample(y))
    elif method == 'permutation':
        # Note: There is no advantage in wasting time permuting both variables, permuting one is sufficient
        fEff = lambda x, y: f(x, permutation_1_sample(y))
    else:
        raise ValueError("Unknown method", method)

    return np.array([fEff(x, y) for i in range(nSample)])
