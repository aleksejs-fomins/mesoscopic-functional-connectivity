import numpy as np

from scipy.special import loggamma
from scipy.optimize import minimize

##################################
# All routines in this file assume shape [nTrial, nTime, nChannel]
##################################

def zscore(x):
    return (x - np.mean(x)) / np.std(x)

def autocorr(x):
    N = len(x)
    return np.correlate(x, x, 'full')[N - 1:] / N

def mu_std(x, axis=None):
    return np.mean(x, axis=axis), np.std(x, axis=axis)

def confint_1D(x):
    n = len(x)
    delta = int(n * (1 - 0.68) / 2)
    return np.sort(x)[[delta, -delta]]

def plot_mean_variance_bychannel(ax, data):
    nChannel = data.shape[2]
    mu, std = mu_std(data, axis=(0, 1))
    x = np.arange(nChannel)
    ax.fill_between(x, mu-std, mu+std, alpha=0.2)
    ax.plot(x, mu)
    ax.set_title('DF/F average over Trials and Time')
    ax.set_xlabel('Channel index')


def plot_mean_variance_bytrial(ax, data):
    nTrial, nTime, nChannel = data.shape
    mu, std = mu_std(data, axis=(1, 2))
    x = np.arange(nTrial)
    ax.fill_between(x, mu-std, mu+std, alpha=0.2)
    ax.plot(x, mu)
    ax.set_title('DF/F average over Trials')
    ax.set_xlabel('Trial index')


def plot_mean_variance_bytime(ax, data, fps=1):
    nTrial, nTime, nChannel = data.shape
    mu, std = mu_std(data, axis=(0, 2))
    x = np.arange(nTime) / fps
    ax.fill_between(x, mu-std, mu+std, alpha=0.2)
    ax.plot(x, mu)
    ax.set_title('DF/F average over Time')
    ax.set_xlabel('Time')




def plot_autocorrelation_bychannel(ax, data, fps=1):
    nTrial, nTime, nChannel = data.shape

    x = np.arange(nTime) / fps
    for iCh in range(nChannel):
        dataThis = zscore(data[:, :, iCh])
        autoCorr = [autocorr(dataThis[iTr]) for iTr in range(nTrial)]
        mu, std = mu_std(autoCorr, axis=0)
        ax.fill_between(x, mu - std, mu + std, alpha=0.2)
        ax.plot(x, mu)
    ax.legend()
    ax.set_title('Autocorrelation average by channel')
    ax.set_xlabel('Time shift')


def plot_ar(ax, data, histMax=10, testFrac = 0.1, fps=1):
    nTrial, nTime, nChannel = data.shape
    histLst = np.arange(1, histMax)
    trainErr = np.zeros((nChannel, histMax - 1))
    testErr = np.zeros((nChannel, histMax - 1))
    ar1Lst = []

    def data2xy(d, nTime, nHist):
        y = d[nHist:].flatten()
        x = np.array([d[iTime:iTime + nHist] for iTime in range(nTime - nHist)])
        x = x.transpose((0, 2, 1)).reshape((y.shape[0], nHist))
        return x, y

    def rel_err(x, y, alpha):
        return np.linalg.norm(x.dot(alpha) - y) / np.linalg.norm(y)

    def ar_solve(x ,y):
        v = y.dot(x)
        M = x.T.dot(x)
        return np.linalg.solve(M, v)

    for iCh in range(nChannel):
        dataThis = zscore(data[:, :, iCh]).T  # Time goes first for convenience
        for iHist in histLst:
            x, y = data2xy(dataThis, nTime, iHist)

            trainIdxs = np.random.uniform(0, 1, len(y)) < testFrac
            xTrain, yTrain = x[trainIdxs], y[trainIdxs]
            xTest, yTest = x[~trainIdxs], y[~trainIdxs]

            alpha = ar_solve(xTrain, yTrain)
            trainErr[iCh, iHist-1] = rel_err(xTrain, yTrain, alpha)
            testErr[iCh, iHist-1] = rel_err(xTest, yTest, alpha)

            # Record the first autoregression coefficient
            if iHist == 1:
                ar1Lst += [alpha[0]]

    dt = 1 / fps
    tau_func = lambda x : -dt / np.log(x)

    muAr1, stdAr1 = mu_std(ar1Lst)
    muTau, stdTau = mu_std(tau_func(np.array(ar1Lst)))

    print("AR(1) coeff =", muAr1, "+/-", stdAr1)
    print("Given dt =", dt, "that translates to tau =", np.round(muTau, 2), "+/-", np.round(stdTau, 2))

    muTrain, stdTrain = mu_std(trainErr, axis=0)
    muTest, stdTest = mu_std(testErr, axis=0)

    ax.fill_between(histLst, muTrain - stdTrain, muTrain + stdTrain, alpha=0.2)
    ax.fill_between(histLst, muTest - stdTest, muTest + stdTest, alpha=0.2)
    ax.plot(histLst, muTrain, label='Train')
    ax.plot(histLst, muTest, label='Test')
    ax.set_title('Autoregression error averaged over channel')
    ax.set_xlabel('History depth')
    ax.legend()


def plot_fit_gamma(ax, data):
    x = data[:, :-1, :].flatten()  # Previous time step
    y = data[:, 1:, :].flatten()   # Next time step

    # heatmap, xedges, yedges = np.histogram2d(x, y-x, bins=200)
    # extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]
    #
    # ax.imshow(heatmap.T, extent=extent, origin='lower', cmap='jet')

    ax.plot(x, y-x, '.', alpha=0.1)

    z = x-y

    xIter = np.linspace(np.min(x), np.max(x), 100)
    muArr = np.zeros(100)
    confLArr = np.zeros(100)
    confRArr = np.zeros(100)

    # stdArr = np.zeros(100)
    for i, xI in enumerate(xIter):
        idx = (x > xI - 0.5) & (x < xI + 0.5)
        if np.sum(idx) < 10:
            muArr[i], std = np.nan, np.nan
            confLArr[i], confRArr[i] = np.nan, np.nan
        else:
            muArr[i], std = mu_std(z[idx])
            confLArr[i], confRArr[i] = confint_1D(z[idx])
            confint_1D(x)

    ax.fill_between(xIter, confLArr, confRArr, alpha=0.2, color='r')
    ax.plot(xIter, muArr, color='r')



    # A = np.mean(x)
    # B = np.mean(y)
    # C = np.mean(x**2)
    # D = np.mean(x*y)
    #
    # M = np.array([[1, A], [A, C]])
    # v = np.array([B, D])
    #
    # mu, alpha = np.linalg.solve(M, v)
    #
    # print(mu, alpha, mu / (alpha-1))

    # ax.plot(x, y - mu - alpha*x, '.')

    # ax.plot(x, (y + 20) / (x + 20), '.')

    # def gamma_log_likelihood(k, a, b):
    #     return -a * np.log(b) - (a - 1) * np.mean(np.log(k)) + b * np.mean(k) + loggamma(a)
    #
    # func_gamma_wrap = lambda theta : gamma_log_likelihood(y - theta[2] * x, theta[0], theta[1])
    # theta0 = np.array([5, 1, 0.1])
    #
    # print(func_gamma_wrap(theta0))

    # minClass = minimize(norm_wrap, theta0, method='Nelder-Mead')


