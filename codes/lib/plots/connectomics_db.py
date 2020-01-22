import os
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict

from codes.lib.plots.matplotblib_lib import bins_multi
from codes.lib.plots import connectomics
from codes.lib.metrics import graph_lib
from codes.lib.stat.stat_lib import bootstrap_resample_function
from codes.lib.stat.stat_shared import log_pval_H0_shared_random


def plot_generic(plotFunc, dataDB, pTHR, outpath, plotPrefix, rangesSec=None, ext='.svg'):
    # Get timestep
    timestep = dataDB.summaryTE['timestep']

    datasetSuffix = '_'.join([
        dataDB.summaryTE["downsampling"],
        str(dataDB.summaryTE["max_lag"]),
        str(dataDB.summaryTE["window"])]
    )

    for sweepDict, rows in dataDB.mouse_iterator():
        sweepSuffix = '_'.join(sweepDict.values())
        print("--", sweepSuffix)

        idxs = rows.index()
        times = dataDB.dataTEtimes[idxs]
        data  = dataDB.dataTEFC[idxs]
        labels = rows['mousekey']

        outfname = os.path.join(outpath, '_'.join([plotPrefix, datasetSuffix, sweepSuffix]) + ext)
        if rangesSec is None:
            plotFunc(outfname, times, data, labels, pTHR, timestep)
        else:
            plotFunc(outfname, times, data, labels, rangesSec, pTHR, timestep)


def plot_te_binary_metrics_bytime(dataDB, pTHR, outpath, ext='.svg'):
    plotFunc = connectomics.plot_te_binary_metrics_bytime
    plot_generic(plotFunc, dataDB, pTHR, outpath, 'metrics_connmat_bytime', ext=ext)


def plot_te_float_metrics_bytime(dataDB, pTHR, outpath, ext='.svg'):
    plotFunc = connectomics.plot_te_float_metrics_bytime
    plot_generic(plotFunc, dataDB, pTHR, outpath, 'metrics_te_bytime', ext=ext)


def plot_te_binary_metrics_rangebydays(dataDB, pTHR, rangesSec, outpath, ext='.svg'):
    plotFunc = connectomics.plot_te_binary_metrics_rangebydays
    plot_generic(plotFunc, dataDB, pTHR, outpath, 'metrics_connmat_rangebytime', rangesSec=rangesSec, ext=ext)


def plot_te_float_metrics_rangebydays(dataDB, pTHR, rangesSec, outpath, ext='.svg'):
    plotFunc = connectomics.plot_te_float_metrics_rangebydays
    plot_generic(plotFunc, dataDB, pTHR, outpath, 'metrics_te_rangebytime', rangesSec=rangesSec, ext=ext)


def plot_te_avgnconn_rangebydays(dataDB, pTHR, rangesSec, outpath, ext='.svg'):
    plotFunc = connectomics.plot_te_avgnconn_rangebydays
    plot_generic(plotFunc, dataDB, pTHR, outpath, 'metrics_avgnconn_rangebytime', rangesSec=rangesSec, ext=ext)



def plot_te_shared_link_scatter(dataDB, pTHR, rangesSec, outpath, ext='.svg'):
    plotFunc = connectomics.plot_te_shared_link_scatter
    plot_generic(plotFunc, dataDB, pTHR, outpath, 'shared_link_scatter', rangesSec=rangesSec, ext=ext)


def plot_te_distribution(dataDB, pTHR, rangesSec, outpath, ext='.svg'):
    plotFunc = connectomics.plot_te_distribution
    plot_generic(plotFunc, dataDB, pTHR, outpath, 'te_distr', rangesSec=rangesSec, ext=ext)


def plot_te_distribution_avgbyperformance(dataDB, pTHR, outname=None, show=True):
    '''
    TODO:
       Scatter nConn,TE vs Performance + expertLine
       2xBin mean(nConn), mean(TE) naive&expert + (* from wilcoxon rank-sum)
       Scatter nConn, TE vs nTrial
       Test if nTrial fully explains prev plot
    '''

    perfLst = defaultdict(list)
    meanTE = defaultdict(list)
    meanNConnConf = defaultdict(list)

    for idx, row in dataDB.metaDataFrames['TE'].iterrows():
        # Extract parameters from dataframe
        trial = row['trial']
        method = row['method']
        mousekey = row['mousekey']

        # Find corresponding data file, extract performance
        dataRowsFiltered = dataDB.get_rows('neuro', {'mousekey' : mousekey})
        nRows = dataRowsFiltered.shape[0]
        if nRows == 0:
            print("Warning:", mousekey, "does not have an associated data entry, skipping")
        elif nRows > 1:
            print("Warning: have", len(dataRowsFiltered), "original matches for TE data", mousekey)
            raise ValueError("Unexpected")
        else:
            # Set multiplex key
            key = (trial, method)

            idxData = dataRowsFiltered.index[0]
            perfLst[key] += [dataDB.dataPerformance[idxData]]

            # Compute mean TE and connection frequency
            te, lag, p = dataDB.dataTEFC[idx]

            teOffDiag = graph_lib.offdiag_1D(te)
            pOffDiag = graph_lib.offdiag_1D(p)
            isConn1D = graph_lib.is_conn(pOffDiag, pTHR)

            nDataPoint = np.prod(teOffDiag.shape)
            nConnConf = np.sum(isConn1D)

            # Store results
            meanNConnConf[key] += [nConnConf / nDataPoint]
            meanTE[key] += [np.mean(teOffDiag[isConn1D])]


    methods = set(dataDB.metaDataFrames['TE']["method"])
    trials = set(dataDB.metaDataFrames['TE']["trial"])

    for method in methods:
        fig, ax = plt.subplots(nrows=2, ncols=2, figsize=(10, 10))
        fig.suptitle(method)

        multiBinDataNConn = [[], []]
        multiBinDataTE = [[], []]
        for trial in trials:
                key = (trial, method)

                perfThis  = np.array(perfLst[key])
                nConnThis = np.array(meanNConnConf[key])
                teThis    = np.array(meanTE[key])

                idxExpert = perfThis >= 0.7
                multiBinDataNConn[0] += [nConnThis[~idxExpert]]
                multiBinDataNConn[1] += [nConnThis[idxExpert]]
                multiBinDataTE[0] += [teThis[~idxExpert]]
                multiBinDataTE[1] += [teThis[idxExpert]]

                ax[0][0].plot(perfThis, nConnThis, '.', label=trial)
                ax[1][0].semilogy(perfThis, teThis, '.', label=trial)

        bins_multi(ax[0][1], ["naive", "expert"], list(trials), multiBinDataNConn)
        bins_multi(ax[1][1], ["naive", "expert"], list(trials), multiBinDataTE)

        ax[0][0].set_xlabel("performance")
        ax[1][0].set_xlabel("performance")
        ax[0][0].set_ylabel("Connection Frequency")
        ax[1][0].set_ylabel("TE")
        ax[0][0].legend()
        ax[1][0].legend()

    if outname is not None:
        plt.savefig(outname)
    if show:
        plt.show()


def plot_te_distribution_avgbyskill(dataDB, pTHR, outname=None, show=True):
    meanTE = {}
    meanNConnConf = {}
    expertMap = {True: "Expert", False: "Naive"}

    for idx, row in dataDB.metaDataFrames['TE'].iterrows():
        # Extract parameters from dataframe
        trial = row['trial']
        method = row['method']
        mousekey = row['mousekey']

        # Find corresponding data file, extract performance
        dataRowsFiltered = dataDB.get_rows('neuro', {'mousekey' : mousekey})
        nRows = dataRowsFiltered.shape[0]
        if nRows == 0:
            print("Warning:", mousekey, "does not have an associated data entry, skipping")
        elif nRows > 1:
            print("Warning: have", len(dataRowsFiltered), "original matches for TE data", mousekey)
            raise ValueError("Unexpected")
        else:
            skill = expertMap[dataRowsFiltered['isExpert'].values[0]]

            # Set multiplex key
            key = (trial, method, skill)
            if key not in meanTE.keys():
                meanTE[key] = []
            if key not in meanNConnConf.keys():
                meanNConnConf[key] = []

            # Compute mean TE and connection frequency
            te, lag, p = dataDB.dataTEFC[idx]

            teOffDiag = graph_lib.offdiag_1D(te)
            pOffDiag = graph_lib.offdiag_1D(p)
            isConn1D = graph_lib.is_conn(pOffDiag, pTHR)

            nDataPoint = np.prod(teOffDiag.shape)
            nConnConf = np.sum(isConn1D)

            # Store results
            meanNConnConf[key] += [nConnConf / nDataPoint]
            meanTE[key] += [np.mean(teOffDiag[isConn1D])]

    methods = set(dataDB.metaDataFrames['TE']["method"])
    trials = set(dataDB.metaDataFrames['TE']["trial"])
    skills = expertMap.values()

    for method in methods:
        fig, ax = plt.subplots(ncols=2, figsize=(15, 8))
        fig.suptitle(method)

        for trial in trials:
            for skill in skills:
                key = (trial, method, skill)
                label = '_'.join((trial, skill))

                ax[0].hist(meanNConnConf[key], bins='auto', alpha=0.3, label=label)
                ax[1].hist(meanTE[key], bins='auto', alpha=0.3, label=label)
        ax[0].set_xlabel("Connection Frequency")
        ax[1].set_xlabel("TE")
        ax[0].set_xscale('log')
        ax[1].set_xscale('log')
        ax[0].legend()
        ax[1].legend()

    if outname is not None:
        plt.savefig(outname)
    if show:
        plt.show()


def plot_te_shared_link_pval_by_performance(dataDB, pTHR, rangesSec, outname=None, show=True):
    FPS = 20
    RANGE_STEP = (int(FPS * rangesSec[0]), int(FPS * rangesSec[1]))
    nMethod = len(dataDB.summaryTE['method'])

    ##############################
    # Part 1:
    #   * For every mouse, find conn for every session if connconf in at least 1 time bin within interval
    #   * For every mouse, for every two consecutive sessions, find nConnPre, nConnPost, nConnShared
    ##############################

    sharedConnDict = {}
    for method in dataDB.summaryTE['method'].keys():

        sharedConnDict[method] = {
            'nConnTot': [],
            'nConnPre': [],
            'nConnPost': [],
            'nConnShared': [],
            'perf': []
        }

        for mousename in dataDB.mice:
            mouseRows = dataDB.get_rows('TE', {'mousename': mousename})

            isConnAnyRngLst = []
            nConnAnyRng = []
            perfSharedThis = []

            for idx, row in mouseRows.iterrows():
                mousekey = row['mousekey']
                methodThis = row['method']

                if methodThis == method:
                    # Find corresponding data file, extract performance
                    dataRowsFiltered = dataDB.get_rows('neuro', {'mousekey': mousekey})
                    nRows = dataRowsFiltered.shape[0]
                    if nRows == 0:
                        print("Warning:", mousekey, "does not have an associated data entry, skipping")
                    elif nRows > 1:
                        print("Warning: have", len(dataRowsFiltered), "original matches for TE data", mousekey)
                        raise ValueError("Unexpected")
                    else:

                        thisMouseDataIdxs = dataRowsFiltered.index
                        perf = dataDB.dataPerformance[thisMouseDataIdxs]
                        perfSharedThis += [perf]

                        # Compute mean TE and connection frequency
                        te, lag, p = dataDB.dataTEFC[idx]
                        pRng = p[..., RANGE_STEP[0]: RANGE_STEP[1]]

                        nChannel = p.shape[1]
                        isConnRng = graph_lib.is_conn(pRng, pTHR)
                        isConnAnyRngLst += [np.any(isConnRng, axis=2)]
                        nConnAnyRng += [np.sum(isConnAnyRngLst[-1])]

            for i in range(1, len(isConnAnyRngLst)):
                sharedConnDict[method]['nConnTot'] += [nChannel * (nChannel - 1)]
                sharedConnDict[method]['nConnPre'] += [nConnAnyRng[i - 1]]
                sharedConnDict[method]['nConnPost'] += [nConnAnyRng[i]]
                sharedConnDict[method]['nConnShared'] += [np.sum(nConnAnyRng[i - 1] & nConnAnyRng[i])]
                sharedConnDict[method]['perf'] += [perfSharedThis[i]]

        for k, v in sharedConnDict[method].items():
            sharedConnDict[method][k] = np.array(v)

    ##############################
    # Part 2:
    #   * For each two consecutive sessions over all mice, test hypothesis that shared links appeared by random chance via permutation
    #   * Plot p-value of H0 versus performance
    #   * Fit trendline, see if pVal decreases with performance
    ##############################

    fig, ax = plt.subplots(ncols=nMethod, figsize=(10, 5))
    fig.suptitle("P-value of nSharedConn under H0 of random permutation")

    for iMethod, method in enumerate(sharedConnDict.keys()):
        tmpDict = sharedConnDict[method]

        logPvalL, logPvalR = log_pval_H0_shared_random(
            tmpDict['nConnTot'],
            tmpDict['nConnPre'],
            tmpDict['nConnPost'],
            tmpDict['nConnShared'])

        idxL = logPvalL < logPvalR

        THR = -100
        logPvalL[logPvalL < THR] = THR
        logPvalR[logPvalR < THR] = THR

        ax[iMethod].plot(tmpDict['perf'][idxL],  logPvalL[idxL], '.', label='fewer')
        ax[iMethod].plot(tmpDict['perf'][~idxL], logPvalR[~idxL], '.', label='more')
        ax[iMethod].legend()
        ax[iMethod].set_xlim(0, 1)
        ax[iMethod].set_xlabel("Performance")
        ax[iMethod].set_ylabel("Log p-value")
        ax[iMethod].set_title(method)

        logPvalJoint = np.zeros(logPvalL.shape)
        logPvalJoint[idxL] = logPvalL[idxL]
        logPvalJoint[~idxL] = logPvalR[~idxL]
        idxExpert = (tmpDict['perf'] > 0.7)[:, 0]

        muNaive, stdNaive = bootstrap_resample_function(np.median, logPvalJoint[~idxExpert], 10000)
        muExpert, stdExpert = bootstrap_resample_function(np.median, logPvalJoint[idxExpert], 10000)
        print("For method", method, "")
        print("* Naive median", np.round(muNaive, 2), "+/-", np.round(stdNaive, 2))
        print("* Expert median", np.round(muExpert, 2), "+/-", np.round(stdExpert, 2))


    ##############################
    # Part 3: TODO
    #   * Sweep performance with
    ##############################

    if outname is not None:
        plt.savefig(outname)
    if show:
        plt.show()
