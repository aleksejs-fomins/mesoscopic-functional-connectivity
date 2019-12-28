import os
import numpy as np
import matplotlib.pyplot as plt

from codes.lib.plots import connectomics
from codes.lib.metrics import graph_lib


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
        if nRows != 1:
            print("Warning: have", len(dataRowsFiltered), "original matches for TE data", mousekey)
            raise ValueError("Unexpected")

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
