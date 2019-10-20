import numpy as np
import bisect
from scipy import interpolate



def gaussian(mu, s2):
    return np.exp(- mu**2 / (2 * s2) )


# Compute discretized exponential decay convolution
def approxDelayConv(data, TAU, DT):
    N = data.shape[0]
    alpha = DT / TAU
    beta = 1-alpha
    
    rez = np.zeros(N+1)
    for i in range(1, N+1):
        rez[i] = data[i-1]*alpha + rez[i-1]*beta

    return rez[1:]

# # Imitate geometric sampling, by selecting some neurons 100% and the rest exponentially dropping
# def samplingRangeScale(x, delta, tau):
#     return np.multiply(x < delta, 1.0) + np.multiply(x >= delta, np.exp(-(x-delta)/tau))

def trunc_idx(x1, xmin, xmax):
    l = bisect.bisect_left(x1, xmin)
    r = bisect.bisect_right(x1, xmax)
    return l, r


def resample_kernel(x1, x2, sig2):
    # Each downsampled val is average of all original val weighted by proximity kernel
    xx1 = np.outer(x2, np.ones(len(x1)))
    xx2 = np.outer(np.ones(len(x2)), x1)
    W = gaussian(xx2 - xx1, sig2)

    for i in range(len(x2)):
        W[i] /= np.sum(W[i])

    return W


def resample(x1, y1, x2, param):
    N2 = len(x2)
    y2 = np.zeros(N2)
    DX2 = x2[1] - x2[0]   # step size for final distribution

    # Check that the new data range does not exceed the old one
    rangeX1 = [np.min(x1), np.max(x1)]
    rangeX2 = [np.min(x2), np.max(x2)]
    if (rangeX2[0] < rangeX1[0])or(rangeX2[1] > rangeX1[1]):
        raise ValueError("Requested range", rangeX2, "exceeds the original data range", rangeX1)
    
    # UpSampling: Use if original dataset has lower sampling rate than final
    if param["method"] == "interpolative":
        kind = param["kind"] if "kind" in param.keys() else "cubic"
        y2 = interpolate.interp1d(x1, y1, kind=kind)(x2)
        
    # Downsample uniformly-sampled data by kernel or bin-averaging
    # DownSampling: Use if original dataset has higher sampling rate than final
    else:
        kind = param["kind"] if "kind" in param.keys() else "window"
        # Window-average method
        if kind == "window":
            window_size = param["window_size"] if "window_size" in param.keys() else DX2

            for i2 in range(N2):
                # Find time-window to average
                w_l = x2[i2] - 0.5 * window_size
                w_r = x2[i2] + 0.5 * window_size

                # Find points of original dataset to average
                i1_l, i1_r = trunc_idx(x1, w_l, w_r)
                # i1_l = np.max([int(np.ceil((w_l - x1[0]) / DX1)), 0])
                # i1_r = np.min([int(np.floor((w_r - x1[0]) / DX1)), N1])

                # Compute downsampled values by averaging
                y2[i2] = np.mean(y1[i1_l:i1_r])

        # Gaussian kernel method
        else:
            ker_sig2 = param["ker_sig2"] if "ker_sig2" in param.keys() else (DX2/2)**2
            WKer = param["ker_w"] if "ker_w" in param else resample_kernel(x1, x2, ker_sig2)
            y2 = WKer.dot(y1)

            # # Each downsampled val is average of all original val weighted by proximity kernel
            # w_ker = gaussian(x2[i2] - x1, ker_sig2)
            # w_ker /= np.sum(w_ker)
            # y2[i2] = w_ker.dot(y1)
        
    return y2