import numpy as np
import matplotlib.pyplot as plt
from statsmodels.stats.proportion import proportion_confint


from code_python.lib.null.models import two_node_system, three_node_system


def tp_fp(mat, nTest, nLag):
    nTestTP = np.sum(np.diag(mat)).astype(float)
    nTestFP = np.sum(mat) - nTestTP
    return nTestTP, nTestFP


def tp_fp_te_lag_corrected(mat, nTest, nLag):
    nTestTP = np.sum(mat.diagonal(offset=1)) + mat[-1][-1]  # Correction for TE convolutional lag bias
    nTestFP = np.sum(mat) - nTestTP
    return nTestTP, nTestFP


def list_bernoulli_bounds(nHitLst, nTot, alpha=0.05, method='binom_test'):
    lLst = []
    rLst = []
    for nHit in nHitLst:
        l, r = proportion_confint(nHit, nTot, alpha=alpha, method=method)
        lLst += [l]
        rLst += [r]
    return np.array(lLst), np.array(rLst)


def filled_bernoulli_plot(ax, xLst, nHitLst, nTot, label=None):
    pMu = np.array(nHitLst) / nTot
    pL, pR = list_bernoulli_bounds(nHitLst, nTot, alpha=0.05, method='beta')

    ax.plot(xLst, pMu, label=label)
    ax.fill_between(xLst, pL, pR, alpha=0.3)


def _get_settings(metric, maxLag):
    if metric == 'crosscorr':
        return {
            'metricSettings': {'havePVal': True},
            'sweepSettings': {'lag': np.arange(1, maxLag + 1)}
        }
    elif metric in ['BivariateTE', 'MultivariateTE']:
        return {
            'metricSettings': {
                'dim_order': 'ps',
                'cmi_estimator': 'JidtGaussianCMI',
                'max_lag_sources': maxLag,
                'min_lag_sources': 1,
                'parallelTrg': False
            }
        }


# Test if existing link is correctly found
def crosstalk_test_two_node_negative(mc, methodLst, testConv=False, nCTDiscr=20, nData=2000,
                  nTest=20, maxLag=5, pTHR=0.01, noiseFrac=0.01, convDT=0.05, convTau=0.5, showPlot=False):
    convLst = [False, True] if testConv else [False]
    for haveConv in convLst:
        convID = "raw" if not haveConv else "conv"

        fig, ax = plt.subplots(figsize=(4, 4))
        ax.set_title('False Positive Rate')

        for method in methodLst:
            print('Doing method', method, convID)

            settingsDict = _get_settings(method, maxLag)

            crossTalkLst = np.linspace(0, 1, nCTDiscr)
            fpLst = []
            for crossTalk in crossTalkLst:
                nFP = 0
                for iTest in range(nTest):

                    # 1. Set Data
                    if not haveConv:
                        data = two_node_system(nData, None, None,
                                               noiseFrac=noiseFrac, crossYX=crossTalk)
                    else:
                        data = two_node_system(nData, None, None, noiseFrac=noiseFrac,
                                               crossYX=crossTalk, convDT=convDT, convTau=convTau)
                    mc.set_data(data, 'ps')

                    # 2. Compute metric
                    rez = mc.metric3D(method, '', **settingsDict)

                    # 3. Increase FP if any lag significant
                    if method == 'crosscorr':
                        pval = rez[:, 0, 1, 1]    # [Lagsweep, src, trg, ans(val/pval)]
                        if np.any(pval < pTHR):
                            nFP += 1
                    else:
                        pval = rez[2, 0, 1]
                        if (not np.isnan(pval)) and (pval < pTHR):
                            nFP += 1

                fpLst += [nFP]

            nTestTot = nTest
            filled_bernoulli_plot(ax, crossTalkLst, fpLst, nTestTot, label=method)

        ax.legend()
        ax.set_ylim(-0.05, 1.05)
        plt.savefig("crosstalk_negative_" + convID + ".svg")
        if showPlot:
            plt.show()


# Test if existing link is correctly found
def crosstalk_test_two_node_positive(mc, methodLst, testConv=False, nCTDiscr=20, nData=2000,
                  nTest=20, maxLag=5, pTHR=0.01, noiseFrac=0.01, connStr=0.5, convDT=0.05, convTau=0.5, showPlot=False):
    convLst = [False, True] if testConv else [False]
    for haveConv in convLst:
        convID = "raw" if not haveConv else "conv"

        fig, ax = plt.subplots(ncols=3, figsize=(12, 4))
        ax[0].set_title('True Positive Rate')
        ax[1].set_title('False Positive Rate')
        ax[2].set_title('True Positive Rate - No Lag')

        for method in methodLst:
            print('Doing method', method, convID)

            settingsDict = _get_settings(method, maxLag)

            crossTalkLst = np.linspace(0, 1, nCTDiscr)
            tpLst = []
            fpLst = []
            tpLooseLst = []
            for crossTalk in crossTalkLst:
                nTP = 0
                rezMat = np.zeros((maxLag + 1, maxLag + 1))
                for lagTrue in range(1, maxLag + 1):
                    for iTest in range(nTest):

                        # 1. Set Data
                        if not haveConv:
                            data = two_node_system(nData, [lagTrue], [connStr],
                                                   noiseFrac=noiseFrac, crossYX=crossTalk)
                        else:
                            data = two_node_system(nData, [lagTrue], [connStr], noiseFrac=noiseFrac,
                                                   crossYX=crossTalk, convDT=convDT, convTau=convTau)
                        mc.set_data(data, 'ps')

                        # 2. Compute metric
                        rez = mc.metric3D(method, '', **settingsDict)

                        # 3. If metric significant at any lag, store max significant lag
                        if method == 'crosscorr':
                            # Cross-correlation returns results for each lag, so need to find max
                            mag = np.abs(rez[:, 0, 1, 0])
                            pval = rez[:, 0, 1, 1]
                            psign = pval < pTHR

                            # print(crossTalk, lagTrue, np.round(-np.log10(pval), 2))

                            if np.any(psign):
                                mag[~psign] = -np.inf
                                lagEst = np.argmax(mag) + 1
                                rezMat[lagTrue][lagEst] += 1
                                nTP += 1  # If any is significant, loose estimator is true

                        else:
                            # TE already finds max, just need to test if it is significant
                            pval = rez[2, 0, 1]
                            if (not np.isnan(pval)) and (pval < pTHR):
                                lagEst = int(rez[1, 0, 1])
                                rezMat[lagTrue][lagEst] += 1
                                nTP += 1  # If any is significant, loose estimator is true

                # 4. Compute true positives and false positives
                if haveConv and method in ['BivariateTE', 'MultivariateTE']:
                    tpr, fpr = tp_fp_te_lag_corrected(rezMat[1:, 1:], nTest, maxLag)
                    # tpr, fpr = acc_te_lag_corrected(rezMat, nTest, maxLag)
                else:
                    tpr, fpr = tp_fp(rezMat[1:, 1:], nTest, maxLag)
                    # tpr, fpr = acc(rezMat, nTest, maxLag)
                tpLst += [tpr]
                fpLst += [fpr]
                tpLooseLst += [nTP]

                # print(rezMat, tpr, fpr)
            
            nTestTot = nTest * maxLag
            filled_bernoulli_plot(ax[0], crossTalkLst, tpLst, nTestTot, label=method)
            filled_bernoulli_plot(ax[1], crossTalkLst, fpLst, nTestTot, label=method)
            filled_bernoulli_plot(ax[2], crossTalkLst, tpLooseLst, nTestTot, label=method)

        ax[0].legend()
        ax[1].legend()
        ax[2].legend()
        ax[0].set_ylim(-0.05, 1.05)
        ax[1].set_ylim(-0.05, 1.05)
        ax[2].set_ylim(-0.05, 1.05)
        plt.savefig("crosstalk_positive_" + convID + ".svg")
        if showPlot:
            plt.show()


# Test if existing link is correctly found
def crosstalk_test_three_node_positive(mc, methodLst, testConv=False, nCTDiscr=20, nData=2000,
                  nTest=20, maxLag=5, pTHR=0.01, noiseFrac=0.01, connStr=0.5, convDT=0.05, convTau=0.5, showPlot=False):
    convLst = [False, True] if testConv else [False]
    for haveConv in convLst:
        convID = "raw" if not haveConv else "conv"

        fig, ax = plt.subplots(ncols=2, figsize=(8, 4))
        ax[0].set_title('True Positive Rate X->Y')
        ax[1].set_title('False Positive Rate Z->Y')

        for method in methodLst:
            print('Doing method', method, convID)

            settingsDict = _get_settings(method, maxLag)

            crossTalkLst = np.linspace(0, 1, nCTDiscr)
            tpLst = []
            fpLst = []
            for crossTalk in crossTalkLst:
                nFPZY = 0
                rezMat = np.zeros((maxLag + 1, maxLag + 1))
                for lagTrue in range(1, maxLag + 1):
                    for iTest in range(nTest):

                        # 1. Set Data
                        if not haveConv:
                            data = three_node_system(nData, [lagTrue], [connStr],
                                                     noiseFrac=noiseFrac, crossZX=crossTalk)
                        else:
                            data = three_node_system(nData, [lagTrue], [connStr], noiseFrac=noiseFrac,
                                                     crossZX=crossTalk, convDT=convDT, convTau=convTau)
                        mc.set_data(data, 'ps')

                        # 2. Compute metric
                        rez = mc.metric3D(method, '', **settingsDict)

                        # 3. If metric significant at any lag, store max significant lag
                        if method == 'crosscorr':
                            # Cross-correlation returns results for each lag, so need to find max
                            magXY = np.abs(rez[:, 0, 1, 0])
                            psignXY = rez[:, 0, 1, 1] < pTHR
                            psignZY = rez[:, 2, 1, 1] < pTHR

                            # print(crossTalk, lagTrue, np.round(-np.log10(pval), 2))

                            # 3.1 Test if (0)->(1) true positives are at the right lag
                            if np.any(psignXY):
                                magXY[~psignXY] = -np.inf
                                lagEst = np.argmax(magXY) + 1
                                rezMat[lagTrue][lagEst] += 1

                            # 3.2 Test if (2)->(1) false positives exist
                            if np.any(psignZY):
                                nFPZY += 1

                        else:
                            pvalXY = rez[2, 0, 1]
                            if (not np.isnan(pvalXY)) and (pvalXY < pTHR):
                                lagEstXY = int(rez[1, 0, 1])
                                rezMat[lagTrue][lagEstXY] += 1

                            pvalZY = rez[2, 2, 1]
                            if (not np.isnan(pvalZY)) and (pvalZY < pTHR):
                                nFPZY += 1

                # 4. Compute true positives and false positives
                if haveConv and method in ['BivariateTE', 'MultivariateTE']:
                    tpXY, fpXY = tp_fp_te_lag_corrected(rezMat[1:, 1:], nTest, maxLag)
                    # tpXY, fpXY = acc_te_lag_corrected(rezMat, nTest, maxLag)
                else:
                    tpXY, fpXY = tp_fp(rezMat[1:, 1:], nTest, maxLag)
                    # tpXY, fpXY = acc(rezMat, nTest, maxLag)

                tpLst += [tpXY]
                fpLst += [nFPZY]

                # print(rezMat, tpr, fpr)

            nTestTot = nTest * maxLag
            filled_bernoulli_plot(ax[0], crossTalkLst, tpLst, nTestTot, label=method)
            filled_bernoulli_plot(ax[1], crossTalkLst, fpLst, nTestTot, label=method)

        ax[0].legend()
        ax[1].legend()
        ax[0].set_ylim(-0.05, 1.05)
        ax[1].set_ylim(-0.05, 1.05)
        plt.savefig("crosstalk_3node_" + convID + ".svg")
        if showPlot:
            plt.show()
