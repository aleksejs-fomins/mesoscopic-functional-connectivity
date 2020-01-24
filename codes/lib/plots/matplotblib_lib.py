import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import mannwhitneyu

# Convert y-axis from fractions to percent
def set_percent_axis_y(ax):
    ax.set_yticklabels(['{:,.2%}'.format(x) for x in ax.get_yticks()])


# Plot a histogram from integer data
def hist_int(ax, x, labels=None, xmin=None, xmax=None):
    xmin = np.min(x) if xmin is None else xmin
    xmax = np.max(x) if xmax is None else xmax

    print(xmin, xmax)
    bins = np.arange(xmin, xmax + 1)
    bins_ext = np.arange(xmin - 1, xmax + 1) + 0.5

    if labels is None:
        ax.hist(x, bins=bins_ext, rwidth=0.5, density=True)
    else:
        ax.hist(np.array(x).T, bins=bins_ext, rwidth=0.5, density=True, label=labels) # , alpha=0.5
        ax.legend()

    ax.set_xticks(bins)
    set_percent_axis_y(ax)


def bins_multi(ax, groupLabels, binLabels, data, rankTest=False):

    def autolabel(rects):
        """Attach a text label above each bar in *rects*, displaying its height."""
        for rect in rects:
            height = rect.get_height()
            ax.annotate('{}'.format(height),
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 3),  # 3 points vertical offset
                        textcoords="offset points",
                        ha='center', va='bottom')


    nGroup = len(groupLabels)
    nBin = len(binLabels)

    if rankTest:
        if nBin != 2:
            raise ValueError("Not implemented for number of bins", nBin)

        for iGroup in range(nGroup):
            pVal = mannwhitneyu(data[0][iGroup], data[1][iGroup])
            print("For", groupLabels[iGroup], "Signed-rank test has pval", pVal)


    widthGroupDelta  = 1    # Gap between groups
    rationGroupGap   = 0.5  # Ratio of gap between groups vs group width

    widthGroup = widthGroupDelta / (1 + rationGroupGap)
    widthBin = widthGroup / nBin

    x = np.arange(nGroup) * widthGroupDelta

    groupRects = []
    for iBin, (binData, binLabel) in enumerate(zip(data, binLabels)):
        muBin = [np.mean(v) for v in binData]
        stdMuBin = [np.std(v) / np.sqrt(len(v)) for v in binData]

        shiftBin = -(widthGroup - widthBin)/2 + iBin*widthBin
        groupRects += [ax.bar(x + shiftBin, muBin, widthBin, yerr=stdMuBin, label=binLabel, capsize=5)]
        #autolabel(groupRects[-1])

    ax.set_xticks(x)
    ax.set_xticklabels(groupLabels)
    ax.legend()
    return groupRects
