import os
import h5py
import numpy as np
import matplotlib.pyplot as plt

from codes.lib.metrics.graph_lib import is_conn, offdiag_1D, diag_idx
from codes.lib.metrics.accuracy import accuracyTests, accuracyIndices
from codes.lib.plots.matrix import imshowAddColorBar
from codes.lib.plots.matplotblib_lib import set_percent_axis_y, hist_int


# Create test plots from previously saved file
def fc_accuracy_plots_fromfile(h5_fname, methods, pTHR, logx=True, percenty=False, fig_fname=None):
    with h5py.File(h5_fname, "r") as h5f:
        xparam = np.copy(h5f['metadata']['xparam'])
        connTrue = np.copy(h5f['metadata']['connTrue'])

        for method in methods:
            if method in h5f:
                fcData = [
                    np.copy(h5f[method]['TE_table']),
                    np.copy(h5f[method]['delay_table']),
                    np.copy(h5f[method]['p_table'])]

                if fig_fname is not None:
                    fig_fname_bare, fig_ext = os.path.splitext(fig_fname)
                    fig_fname_method = fig_fname_bare + "_" + method + fig_ext
                else:
                    fig_fname_method = None

                fc_accuracy_plots(xparam, fcData, connTrue, method, pTHR, logx=logx, percenty=percenty, h5_fname=None, fig_fname=fig_fname_method)


# Create test plots directily from data, maybe saving to h5
def fc_accuracy_plots(xparam, fcData, connTrue, method, pTHR, logx=True, percenty=False, h5_fname=None, fig_fname=None):
    te3D, lag3D, p3D = fcData

    #####################################
    # Save data
    #####################################
    if h5_fname is not None:
        filemode = "a" if os.path.isfile(h5_fname) else "w"
        with h5py.File(h5_fname, filemode) as h5f:
            if "metadata" not in h5f.keys():
                grp_rez = h5f.create_group("metadata")
                grp_rez['xparam'] = xparam
                grp_rez['connTrue'] = connTrue

            if method in h5f.keys():
                raise ValueError("Already have data for method", method)

            grp_method = h5f.create_group(method)
            grp_method['TE_table'] = te3D
            grp_method['delay_table'] = lag3D
            grp_method['p_table'] = p3D

    #####################################
    # Analysis
    #####################################

    # Copy data to avoid modifying originals
    te, lag, p = np.copy(te3D), np.copy(lag3D), np.copy(p3D)
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


# Create type-1 plots for BTE with extra curve for false positives in the upper quadrant
def bte_accuracy_special_fromfile(h5_fname, methods, pTHR, connUndecided, logx=True, percenty=False, fig_fname=None):
    with h5py.File(h5_fname, "r") as h5f:
        xparam = np.copy(h5f['metadata']['xparam'])
        connTrue = np.copy(h5f['metadata']['connTrue'])

        for method in methods:
            if method in h5f:
                fcData = [
                    np.copy(h5f[method]['TE_table']),
                    np.copy(h5f[method]['delay_table']),
                    np.copy(h5f[method]['p_table'])]

                if fig_fname is not None:
                    fig_fname_bare, fig_ext = os.path.splitext(fig_fname)
                    fig_fname_method = fig_fname_bare + "_" + method + fig_ext
                else:
                    fig_fname_method = None

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