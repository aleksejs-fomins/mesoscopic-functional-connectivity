import os
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
import pandas as pd
import seaborn as sns
from statannot import add_stat_annotation

#from codes.lib.plots.matplotblib_lib import bins_multi
from codes.lib.plots import connectomics
from codes.lib.metrics import graph_lib
from codes.lib.stat.stat_lib import bootstrap_resample_function
from codes.lib.stat.stat_shared import log_pval_H0_shared_random


def plot_generic(dataDB, pTHR, outpath, plotPrefix, rangesSec=None, ext='.svg'):
    connectomicsMetricsDict = {
        "connectivity_nconn_vs_time"         : connectomics.plot_fc_binary_vs_time,
        "connectivity_nconn_vs_days"         : connectomics.plot_fc_binary_vs_days,
        "connectivity_fc_vs_time"            : connectomics.plot_fc_mag_vs_time,
        "connectivity_fc_vs_days"            : connectomics.plot_fc_mag_vs_days,
        "connectivity_avgnconn_vs_days"      : connectomics.plot_fc_binary_avg_vs_time,
        "connectivity_nconn_shared_avg"      : connectomics.plot_fc_binary_shared_avg,
        "connectivity_fc_vs_performance"     : connectomics.plot_fc_vs_performance
    }

    # Get plot function
    plotFunc = connectomicsMetricsDict[plotPrefix]

    datasetSuffix = '_'.join([
        dataDB.summaryTE["downsampling"],
        str(dataDB.summaryTE["max_lag"]),
        str(dataDB.summaryTE["window"])]
    )

    for sweepDict, rows in dataDB.mouse_iterator():
        sweepSuffix = '_'.join(sweepDict.values())
        outfname = os.path.join(outpath, '_'.join([plotPrefix, datasetSuffix, sweepSuffix]) + ext)
        print("--", sweepSuffix)

        dataLabels = rows['mousekey']

        timesLst = []
        dataLst = []
        if rangesSec is None:
            rangeLabels = ["alltimes"]
            times, data = dataDB.get_fc_data(rows, rangesSec)
            timesLst += [times]
            dataLst += [data]
        else:
            rangeLabels = list(rangesSec.keys())
            for rng in rangesSec.values():
                times, data = dataDB.get_fc_data(rows, rng)
                timesLst += [times]
                dataLst += [data]

        plotFunc(outfname, timesLst, dataLst, dataLabels, rangeLabels, pTHR)


def plot_fc_vs_performance(dataDB, pTHR, outname=None, show=True):
    '''
       Scatter nConn,TE vs Performance + expertLine
       2xBin mean(nConn), mean(TE) naive&expert + (* from wilcoxon rank-sum)
       Scatter nConn, TE vs nTrial
       Test if nTrial fully explains prev plot
    '''

    rezDict = {
        "trial" : [],
        "method" : [],
        "performance" : [],
        "mean TE" : [],
        "mean nConnConf" : []
    }

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
            idxData = dataRowsFiltered.index[0]

            # Compute mean TE and connection frequency
            te, lag, p = dataDB.dataTEFC[idx]

            teOffDiag = graph_lib.offdiag_1D(te)
            pOffDiag = graph_lib.offdiag_1D(p)
            isConn1D = graph_lib.is_conn(pOffDiag, pTHR)

            nDataPoint = np.prod(teOffDiag.shape)
            nConnConf = np.sum(isConn1D)

            # Store results
            rezDict['trial'] += [trial]
            rezDict['method'] += [method]
            rezDict['performance'] += [dataDB.dataPerformance[idxData]]
            rezDict['mean nConnConf'] += [nConnConf / nDataPoint]
            rezDict['mean TE'] += [np.mean(teOffDiag[isConn1D])]

    rezDF = pd.DataFrame(rezDict)

    methods = set(dataDB.metaDataFrames['TE']["method"])
    trials = set(dataDB.metaDataFrames['TE']["trial"])
    metrics = ["mean nConnConf", "mean TE"]
    metricYScales = ["linear", "log"]
    skillMap = {False : "Naive", True : "Expert"}

    for method in methods:
        rezDFMehtod = rezDF[rezDF["method"] == method]
        rezDFMehtod.insert(0, 'skill', (rezDFMehtod["performance"] >= 0.7).map(skillMap))

        fig, ax = plt.subplots(nrows=2, ncols=3, figsize=(12,8), tight_layout=True)
        fig.suptitle(method, y=1.05)

        for iMetric, (metric, scaleY) in enumerate(zip(metrics, metricYScales)):
            sns.scatterplot(ax=ax[iMetric][0], data=rezDFMehtod, x="performance", y=metric, hue="trial")
            sns.violinplot(ax=ax[iMetric][2], data=rezDFMehtod, x="trial", y=metric, hue="skill", cut=0.5)

            ax[iMetric][0].set_yscale(scaleY)
            #ax[iMetric][1].set_xscale(scaleY)
            ax[iMetric][1].set_yscale("log")
            ax[iMetric][2].set_yscale(scaleY)

            box_pairs = [[(trial, skill) for skill in skillMap.values()] for trial in trials]
            add_stat_annotation(ax[iMetric][2], data=rezDFMehtod, x="trial", y=metric, hue="skill",
                                box_pairs=box_pairs, test='Mann-Whitney', loc='inside', verbose=0)  # , text_format='full'

            for trial in trials:
                for skill in skillMap.values():
                    dataThis = rezDFMehtod[(rezDFMehtod["skill"] == skill) & (rezDFMehtod["trial"] == trial)]

                    sns.distplot(dataThis[metric], ax=ax[iMetric][1], kde=False, label=trial + "_" + skill)


            # if scaleY == 'log':
            #     ax[iMetric][2].set_ylim(0.1 * np.min(rezDFMehtod[metric]), 2 * np.max(rezDFMehtod[metric]))

            ax[iMetric][0].legend()
            ax[iMetric][1].legend()
            ax[iMetric][2].legend()

    if outname is not None:
        plt.savefig(outname)
    if show:
        plt.show()


def plot_fc_binary_shared_pval_vs_performance(dataDB, pTHR, rangesSec, outname=None, show=True):
    FPS = 20
    RANGE_STEP = (int(FPS * rangesSec[0]), int(FPS * rangesSec[1]))
    nMethod = len(dataDB.summaryTE['method'])

    ##############################
    # Part 1:
    #   * For every mouse, find conn for every session if connconf in at least 1 time bin within interval
    #   * For every mouse, for every two consecutive sessions, find nConnPre, nConnPost, nConnShared
    ##############################

    sharedConnDict = {
        'method' : [],
        'nConnTot': [],
        'nConnPre': [],
        'nConnPost': [],
        'nConnShared': [],
        'Performance': []
    }

    for method in dataDB.summaryTE['method'].keys():
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
                sharedConnDict['method'] += [method]
                sharedConnDict['nConnTot'] += [nChannel * (nChannel - 1)]
                sharedConnDict['nConnPre'] += [nConnAnyRng[i - 1]]
                sharedConnDict['nConnPost'] += [nConnAnyRng[i]]
                sharedConnDict['nConnShared'] += [np.sum(nConnAnyRng[i - 1] & nConnAnyRng[i])]
                sharedConnDict['Performance'] += [perfSharedThis[i]]

    sharedConnDF = pd.DataFrame(sharedConnDict)

    ##############################
    # Part 2:
    #   * For each two consecutive sessions over all mice, test hypothesis that shared links appeared by random chance via permutation
    #   * Plot p-value of H0 versus performance
    #   * Fit trendline, see if pVal decreases with performance
    ##############################
    nConnDeltaMap = {False : "Fewer", True : "More"}
    skillMap = {False: "Naive", True: "Expert"}

    fig, ax = plt.subplots(nrows=2, ncols=nMethod, figsize=(10, 10), tight_layout=True)
    fig.suptitle("P-value of nSharedConn under H0 of random permutation")

    for iMethod, method in enumerate(set(sharedConnDict['method'])):
        rowsThis = sharedConnDF[sharedConnDF["method"] == method]

        logPvalL, logPvalR = log_pval_H0_shared_random(
            rowsThis['nConnTot'],
            rowsThis['nConnPre'],
            rowsThis['nConnPost'],
            rowsThis['nConnShared'])

        rowsThis.insert(0, 'skill', (rowsThis["Performance"] >= 0.7).map(skillMap))
        rowsThis.insert(0, "Log p-value", np.min([logPvalL, logPvalR], axis=0))
        rowsThis.insert(0, "direction", [nConnDeltaMap[l < r] for l, r in zip(logPvalL, logPvalR)])

        sns.scatterplot(ax=ax[0][iMethod], data=rowsThis, x='Performance', y='Log p-value', hue='direction')
        sns.violinplot(ax=ax[1][iMethod], data=rowsThis, x='skill', y='Log p-value', cut=0)

        # ax[iMetric][0].set_yscale(scaleY)
        # # ax[iMetric][1].set_xscale(scaleY)
        # ax[iMetric][1].set_yscale("log")
        # ax[iMetric][2].set_yscale(scaleY)
        #
        box_pairs = [list(skillMap.values())]
        add_stat_annotation(ax[1][iMethod], data=rowsThis, x="skill", y='Log p-value',
                            box_pairs=box_pairs, test='Mann-Whitney', loc='inside', verbose=0)  # , text_format='full'

        for skill in skillMap.values():
            print("For", skill, method, "mean log p-val is", np.mean(rowsThis[rowsThis['skill'] == skill]['Log p-value']))


        # THR = -100
        # logPvalL[logPvalL < THR] = THR
        # logPvalR[logPvalR < THR] = THR

        # ax[iMethod].plot(tmpDict['perf'][idxL],  logPvalL[idxL], '.', label='fewer')
        # ax[iMethod].plot(tmpDict['perf'][~idxL], logPvalR[~idxL], '.', label='more')
        # ax[iMethod].legend()
        # ax[iMethod].set_xlim(0, 1)
        # ax[iMethod].set_ylim(-200, 10)
        # ax[iMethod].set_xlabel("Performance")
        # ax[iMethod].set_ylabel("Log p-value")
        ax[0][iMethod].set_title(method)

        # logPvalJoint = np.zeros(logPvalL.shape)
        # logPvalJoint[idxL] = logPvalL[idxL]
        # logPvalJoint[~idxL] = logPvalR[~idxL]
        # idxExpert = (tmpDict['perf'] > 0.7)[:, 0]
        #
        # muNaive, stdNaive = bootstrap_resample_function(np.median, logPvalJoint[~idxExpert], 10000)
        # muExpert, stdExpert = bootstrap_resample_function(np.median, logPvalJoint[idxExpert], 10000)
        # print("For method", method, "")
        # print("* Naive median", np.round(muNaive, 2), "+/-", np.round(stdNaive, 2))
        # print("* Expert median", np.round(muExpert, 2), "+/-", np.round(stdExpert, 2))


    ##############################
    # Part 3: TODO
    #   * Sweep performance with
    ##############################

    if outname is not None:
        plt.savefig(outname)
    if show:
        plt.show()