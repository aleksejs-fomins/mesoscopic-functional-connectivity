import numpy as np


def cycle(arr, nStep):
    return np.hstack([arr[-nStep:], arr[:-nStep]])


def mix(x, y, frac):
    return (1 - frac) * x + frac * y


def conv_exp(data, dt, tau):
    nTexp = int(5*tau / dt)
    t = dt * np.arange(nTexp)
    exp = np.exp(-t/tau)
    exp /= np.sum(exp)
    nTData = data.shape[0]
    return np.convolve(data, exp)[:nTData]


def two_node_system(nTime, lags=None, trgFracs=None, noiseFrac=0.1, crossXY=0, crossYX=0, convDT=None, convTau=None):
    x = np.random.normal(0, 1, nTime)
    y = np.random.normal(0, 1, nTime)
    
    # Add lagged coupling
    if lags is not None:
        y = (1 - np.sum(trgFracs)) * y + np.sum([frac * cycle(x, lag) for frac, lag in zip(trgFracs, lags)], axis=0)
    
    # Add convolution
    if convDT is not None:
        x = conv_exp(x, convDT, convTau)
        y = conv_exp(y, convDT, convTau)

    # Add cross-talk. NOTE: with symmetric mixing the variables swap for croxxXY
    xMixed = mix(x, y, crossXY)
    yMixed = mix(y, x, crossYX)
        
    # Add observation noise
    xMixed = mix(xMixed, np.random.normal(0, 1, nTime), noiseFrac)
    yMixed = mix(yMixed, np.random.normal(0, 1, nTime), noiseFrac)
    
    return np.array([xMixed, yMixed])


def three_node_system(nTime, lags=None, trgFracs=None, noiseFrac=0.1, crossZX=0, convDT=None, convTau=None):
    x = np.random.normal(0, 1, nTime)
    y = np.random.normal(0, 1, nTime)
    z = np.random.normal(0, 1, nTime)

    # Add lagged coupling
    if lags is not None:
        y = (1 - np.sum(trgFracs)) * y + np.sum([frac * cycle(x, lag) for frac, lag in zip(trgFracs, lags)], axis=0)

    # Add convolution
    if convDT is not None:
        x = conv_exp(x, convDT, convTau)
        y = conv_exp(y, convDT, convTau)
        z = conv_exp(z, convDT, convTau)

    # Add cross-talk. NOTE: with symmetric mixing the variables swap for croxxXY
    zMixed = mix(z, x, crossZX)

    # Add observation noise
    xMixed = mix(x, np.random.normal(0, 1, nTime), noiseFrac)
    yMixed = mix(y, np.random.normal(0, 1, nTime), noiseFrac)
    zMixed = mix(zMixed, np.random.normal(0, 1, nTime), noiseFrac)

    return np.array([xMixed, yMixed, zMixed])
