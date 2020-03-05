import numpy as np

from scipy.special import loggamma
from scipy.optimize import minimize

from codes.lib.stat.graph_lib import offdiag_1D
from codes.lib.stat.stat_lib import mu_std, tolerance_interval_1D
from codes.lib.stat.ml_lib import split_train_test
from codes.lib.signal_lib import zscore
from codes.lib.info_metrics.corr_lib import corr_nan, autocorr1D
from codes.lib.info_metrics.autoregression1D import AR1D
from codes.lib.info_metrics.npeet_wrapper import entropy, predictive_info


##################################
# All routines in this file assume shape [nTrial, nTime, nChannel]
##################################

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


def plot_autocorrelation_bychannel(ax, data, fps=1, withFill=True):
    nTrial, nTime, nChannel = data.shape

    x = np.arange(nTime) / fps
    for iCh in range(nChannel):
        dataThis = zscore(data[:, :, iCh])
        autocorrArr = np.array([autocorr1D(dataThis[iTr]) for iTr in range(nTrial)])

        mu, std = mu_std(autocorrArr, axis=0)
        confL, confR = np.array([tolerance_interval_1D(x) for x in autocorrArr.T]).T

        if withFill:
            ax.fill_between(x, confL, confR, alpha=0.2)
            # ax.fill_between(x, mu - std, mu + std, alpha=0.2)
        ax.plot(x, mu, label=str(iCh))
    #ax.legend()
    ax.set_title('autocorr1Delation average by channel')
    ax.set_xlabel('Time shift')


def plot_ar(ax0, ax1, data, histMax=10, testFrac = 0.1, fps=1):
    nTrial, nTime, nChannel = data.shape
    histLst = np.arange(1, histMax)
    trainErr = np.zeros((nChannel, histMax - 1))
    testErr = np.zeros((nChannel, histMax - 1))
    ar1Lst = []

    for iCh in range(nChannel):
        dataThis = zscore(data[:, :, iCh]).T  # Time goes first for convenience
        for iHist in histLst:
            ar1D = AR1D(iHist)

            x, y = ar1D.data2xy(dataThis)
            xTrain, yTrain, xTest, yTest = split_train_test(x, y, testFrac)
            alpha = ar1D.fit(xTrain, yTrain)

            yHatTrain = ar1D.predict(xTrain)
            yHatTest = ar1D.predict(xTest)

            trainErr[iCh, iHist-1] = ar1D.rel_err(yTrain, yHatTrain)
            testErr[iCh, iHist-1] = ar1D.rel_err(yTest, yHatTest)

            # Record the first autoregression coefficient
            if iHist == 1:
                ar1Lst += [alpha[0]]

        ax0.semilogy(histLst, testErr[iCh], label=str(iCh))

    ax1.plot(ar1Lst, '.')

    ax0.set_title('Autoregression error by channel')
    ax0.set_xlabel('History depth')
    ax1.set_title('AR(1) coefficient')
    ax1.set_xlabel('Channel index')
    # ax0.legend()


def plot_fit_gamma(ax, data):
    x = data[:, :-1, :].flatten()  # Previous time step
    y = data[:, 1:, :].flatten()   # Next time step

    # heatmap, xedges, yedges = np.histogram2d(x, y-x, bins=200)
    # extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]
    #
    # ax.imshow(heatmap.T, extent=extent, origin='lower', cmap='jet')

    ax.plot(x, y-x, '.', alpha=0.1)
    ax.set_title("Neuronal 1-step signal change")
    ax.set_xlabel("x(t-1)")
    ax.set_ylabel("x(t) - x(t-1)")

    # z = x-y
    #
    # xIter = np.linspace(np.min(x), np.max(x), 100)
    # muArr = np.zeros(100)
    # confLArr = np.zeros(100)
    # confRArr = np.zeros(100)
    #
    # # stdArr = np.zeros(100)
    # for i, xI in enumerate(xIter):
    #     idx = (x > xI - 0.5) & (x < xI + 0.5)
    #     if np.sum(idx) < 10:
    #         muArr[i], std = np.nan, np.nan
    #         confLArr[i], confRArr[i] = np.nan, np.nan
    #     else:
    #         muArr[i], std = mu_std(z[idx])
    #         confLArr[i], confRArr[i] = confint_1D(z[idx])
    #         confint_1D(x)
    #
    # ax.fill_between(xIter, confLArr, confRArr, alpha=0.2, color='r')
    # ax.plot(xIter, muArr, color='r')



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


# Patch Nan's for each channel with mean over that channel
def _patch_nan(data3D):
    nTrial, nTime, nChannel = data3D.shape
    dataPatch = np.copy(data3D)
    for iChannel in range(nChannel):
        dataPatch[:,:,iChannel][np.isnan(dataPatch[:,:,iChannel])] = np.nanmean(dataPatch[:,:,iChannel])

    assert ~np.any(np.isnan(dataPatch))

    return dataPatch


def plot_mean_correlation_bytime(ax, data, fps=1):
    nTrial, nTime, nChannel = data.shape

    # FIXME
    # dataNoNan = _patch_nan(data)
    dataNoNan = np.copy(data)

    x = np.arange(nTime) / fps
    synchrony_coeff = lambda data2D: np.nanmean(np.abs(offdiag_1D(np.corrcoef(data2D))))
    sCoeff = [synchrony_coeff(dataNoNan[:, iTime, :]) for iTime in range(nTime)]

    ax.plot(x, sCoeff)
    ax.set_ylim([0,1])
    ax.set_xlabel('Time steps')
    ax.set_title('Synchronization coefficient')


def plot_entropy_ND_bytime(ax1, ax2, ax3, data, fps=1):
    nTrial, nTime, nChannel = data.shape

    # FIXME
    # dataNoNan = _patch_nan(data)

    dataScored = np.array([zscore(data[:,:,iChannel]) for iChannel in range(nChannel)]).transpose((1, 2, 0))

    xEntr = np.arange(nTime) / fps
    xPI = np.arange(nTime-1) / fps

    entropy_1D = lambda data1D: entropy(data1D[..., None, None], {"dim_order": "rps"})
    entropy_2D = lambda data2D: entropy(data2D[..., None], {"dim_order" : "rps"})
    pi_1D      = lambda data1D: predictive_info(data1D[..., None], {"dim_order": "rsp", "max_lag": 1})
    pi_2D      = lambda data2D: predictive_info(data2D[...], {"dim_order": "rsp", "max_lag":1})


    e1D = [np.mean([entropy_1D(dataScored[:, iTime, iChannel]) for iChannel in range(nChannel)]) for iTime in range(nTime)]
    pi1D = [np.mean([pi_1D(dataScored[:, iTime:iTime + 2, iChannel]) for iChannel in range(nChannel)]) for iTime in range(nTime - 1)]

    e2D = [entropy_2D(dataScored[:, iTime, :]) for iTime in range(nTime)]
    pi2D = [pi_2D(dataScored[:, iTime:iTime+2, :]) for iTime in range(nTime-1)]

    e1D = np.array(e1D)
    e2D = np.array(e2D)[:, 0]
    pi1D = np.array(pi1D)
    pi2D = np.array(pi2D)[:, 0]

    ax1.plot(xEntr, e1D, label="Max Entropy")
    ax1.plot(xEntr, e2D / nChannel, label="ND Entropy")
    ax2.plot(xEntr, (e1D - e2D / nChannel) / 2)
    ax3.plot(xPI, pi1D, label='avg1D')
    ax3.plot(xPI, pi2D, label='ND')
    ax1.legend()
    ax3.legend()
    ax1.set_xlabel('Time steps')
    ax2.set_xlabel('Time steps')
    ax3.set_xlabel('Time steps')
    ax1.set_title('Normalized Entropy')
    ax2.set_title('Normalized Total Correlation')
    ax3.set_title('Predictive Information')