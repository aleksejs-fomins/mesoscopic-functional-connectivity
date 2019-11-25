import numpy as np
import matplotlib.pyplot as plt

from codes.lib.metrics.graph_lib import is_conn, offdiag_1D, diag_idx
from codes.lib.metrics.accuracy import accuracyTests, accuracyIndices
from codes.lib.plots.matrix import imshowAddColorBar
from codes.lib.plots.matplotblib_lib import set_percent_axis_y, hist_int


# Create test plots directily from data, maybe saving to h5
def fc_accuracy_plots(xparam, fcData, connTrue, method, pTHR, logx=True, percenty=False, fig_fname=None):
    #####################################
    # Analysis
    #####################################

    # Copy data to avoid modifying originals
    te, lag, p = np.copy(fcData)
    nNode, _, nStep = te.shape

    #  Find which connections have high confidence
    isConnConf     = is_conn(p, pTHR)
    isConnConfTrue = ~np.isnan(connTrue)
    isConnConfOffdiag1D     = offdiag_1D(isConnConf)
    isConnConfOffdiag1DTrue = offdiag_1D(isConnConfTrue)

    freqTrue = np.mean(isConnConfOffdiag1DTrue)

    # Delete all connections that have too high p-value
    # Set values of all other connections to NAN
    te[~isConnConf] = np.nan
    lag[~isConnConf] = np.nan
    p[~isConnConf] = np.nan

    # Compute statistics
    _, errIdxsDict = accuracyIndices(isConnConfOffdiag1D, isConnConfOffdiag1DTrue)
    errTestDict = accuracyTests(isConnConfOffdiag1D, isConnConfOffdiag1DTrue)
    connFreq = np.mean(isConnConf, axis=2)

    # Lag statistics
    minLag = 0
    maxLag = np.nanmax(lag)
    lag1DoffDiag = offdiag_1D(lag)
    lagsTP = lag1DoffDiag[errIdxsDict['TP']]  # Lags of TP connections
    lagsFP = lag1DoffDiag[errIdxsDict['FP']]  # Lags of FP connections
    # Note that for FN and TN lags don't exist, since no connections were found

    # P-value statistics
    p1DoffDiag = offdiag_1D(p)
    pvalTPmean = [np.nanmean(p1DoffDiag[:, i][errIdxsDict['TP'][:, i]]) for i in range(nStep)]
    pvalFPmean = [np.nanmean(p1DoffDiag[:, i][errIdxsDict['FP'][:, i]]) for i in range(nStep)]
    #pvalTPstd = [np.nanstd(pvalTP[:, i]) for i in range(nStep)]
    #pvalFPstd = [np.nanstd(pvalFP[:, i]) for i in range(nStep)]
    # Note that for FN and TN p-values don't exist, since no connections were found

    #####################################
    # Plots
    #####################################

    plotfuncLinY = lambda ax: ax.semilogx if logx else ax.plot
    plotfuncLogY = lambda ax: ax.loglog if logx else ax.semilogy

    fig, ax = plt.subplots(nrows=2, ncols=3, figsize=(15, 10))

    ###############################
    # Error rates
    ###############################
    ax[0][0].set_title("Error frequencies")
    plotfuncLinY(ax[0][0])(xparam, errTestDict["FP frequency"], '.-', label='FP')
    plotfuncLinY(ax[0][0])(xparam, errTestDict["FN frequency"], '.-', label='FN')

    if freqTrue > 0:  # Add max possible TP frequency
        ax[0][0].axhline(y=freqTrue, linestyle='--', label='P')

    if percenty:  # Write y-axis as percent
        set_percent_axis_y(ax[0][0])
    ax[0][0].legend()

    ###############################
    # Connectivity
    ###############################
    ax[0][1].set_title("True connections")
    img01 = ax[0][1].imshow(connTrue, vmin=0, vmax=1)
    imshowAddColorBar(fig, ax[0][1], img01)

    ax[1][1].set_title("Frequencies of connections")
    img11 = ax[1][1].imshow(connFreq, vmin=0, vmax=1)
    imshowAddColorBar(fig, ax[1][1], img11)

    ###############################
    # TE values
    ###############################
    ax[1][0].set_title("TE for each connection")
    for i in range(nNode):
        for j in range(nNode):
            plotfuncLogY(ax[1][0])(xparam, te[i, j, :], '.-')

    ###############################
    # P-values
    ###############################
    ax[1][2].set_title("Mean p-value")
    plotfuncLinY(ax[1][2])(xparam, pvalTPmean, '.-', label='TP')
    plotfuncLinY(ax[1][2])(xparam, pvalFPmean, '.-', label='FP')
    ax[1][2].legend()

    ###############################
    # Lags
    ###############################
    ax[0][2].set_title("lag distribution")
    hist_int(ax[0][2], [lagsTP, lagsFP], labels=['TP', 'FP'], xmin=minLag, xmax=maxLag)

    ###############################
    # Save figure
    ###############################
    if fig_fname is not None:
        plt.savefig(fig_fname, dpi=300)
    else:
        plt.show()


# Create type-1 plots for BTE with extra curve for false positives in the upper quadrant
def bte_accuracy_special_fromfile(xparam, fcData, connTrue, method, pTHR, connUndecided, logx=True, percenty=False, fig_fname=None):
    # Copy data to avoid modifying originals
    te, lag, p = fcData
    nNode, _, nStep = te.shape

    #  Find which connections have high confidence
    connExclude = connUndecided & diag_idx(nNode)

    isConn1DeffH1 = is_conn(offdiag_1D(p), pTHR)
    isConn1DeffH2 = is_conn(p[~connExclude], pTHR)
    isConn1DeffH1True = ~np.isnan(offdiag_1D(connTrue))
    isConn1DeffH2True = ~np.isnan(connTrue[~connExclude])

    freqTrueH1 = np.mean(isConn1DeffH1True)
    freqTrueH2 = np.mean(isConn1DeffH2True)

    # Compute statistics
    errTestDictH1 = accuracyTests(isConn1DeffH1, isConn1DeffH1True)
    errTestDictH2 = accuracyTests(isConn1DeffH2, isConn1DeffH2True)

    #####################################
    # Plots
    #####################################

    plotfuncLinY = lambda ax: ax.semilogx if logx else ax.plot

    fig, ax = plt.subplots()
    ax.set_title("Error frequencies")

    plotfuncLinY(ax)(xparam, errTestDictH1["FP frequency"], '.-', label='FP-H1')
    plotfuncLinY(ax)(xparam, errTestDictH2["FP frequency"], '.-', label='FP-H2')
    plotfuncLinY(ax)(xparam, errTestDictH2["FN frequency"], '.-', label='FN')

    if freqTrueH2 > 0:
        ax.axhline(y=freqTrueH2, linestyle='--', label='P')

    # Write y-axis as percent
    if percenty:
        set_percent_axis_y(ax)
    ax.legend()

    if fig_fname is not None:
        plt.savefig(fig_fname, dpi=300)
    else:
        plt.show()


def fc_accuracy_plots_bymethod(xparam, fcDataDict, pTHR, logx=True, percenty=False, fig_fname=None):

    plotfuncLinY = lambda ax: ax.semilogx if logx else ax.plot

    fig, ax = plt.subplots(figsize=(5, 5))
    ax.set_title("Type 1 error")

    for method, fcData in fcDataDict.items():
        te, lag, p = fcData
        nNode, _, nStep = te.shape

        isConn1D = is_conn(offdiag_1D(p), pTHR)
        nConn = np.sum(isConn1D, axis=0)

        plotfuncLinY(ax)(xparam, nConn, '.-', label=method)

    if percenty:
        set_percent_axis_y(ax)
    ax.legend()

    if fig_fname is not None:
        plt.savefig(fig_fname, dpi=300)
    else:
        plt.show()


# Would compare type 1 accuracy for every model. Currently low priority
#def fc_accuracy_plots_bymodel():


def fc_accuracy_plots_vsreal(xparam, fcDataDictSim, fcDataDictReal, pTHR, logx=True, percenty=False, fig_fname=None):
    pass