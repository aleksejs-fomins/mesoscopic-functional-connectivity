import numpy as np

class AR1D:

    def __init__(self, nHist):
        self.nHist = nHist

    # Convert a 2D dataset to predicted values (current timesteps flattened) and predictors (past timesteps flattened)
    # Afterwards, drop all rows where at least one value is NAN
    # NOTE: currently, past dimensions are sorted from oldest to newest, so data[t-1] = x[-1]
    def data2xy(self, d):
        nHist = self.nHist
        nTime, nTrial = d.shape

        y = d[nHist:].flatten()
        x = np.array([d[iTime:iTime + nHist] for iTime in range(nTime - nHist)])
        x = x.transpose((0, 2, 1)).reshape((y.shape[0], nHist))

        # Truncate all datapoints that have at least one NAN in them
        nanY = np.isnan(y)
        nanX = np.any(np.isnan(x), axis=1)

        goodIdx = (~nanX) & (~nanY)

        return x[goodIdx], y[goodIdx]


    # Fit AR(1) coefficients to data
    def fit(self, x, y):
        v = y.dot(x)
        M = x.T.dot(x)
        self.alpha = np.linalg.solve(M, v)
        return self.alpha


    # Compute prediction
    def predict(self, x):
        return x.dot(self.alpha)


    # Compute relative prediction error
    def rel_err(self, y, yhat):
        return np.linalg.norm(y - yhat) / np.linalg.norm(y)


    # Estimate timescale under hypothesis of convolution with exponential kernel
    def tau_exp(self, fps):
        dt = 1 / fps
        return -dt / np.log(self.alpha[-1])