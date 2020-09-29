import numpy as np

from lib.info_metrics.projector_metrics import metric3D
from lib.array_lib import unique_subtract

from IPython.display import display
from ipywidgets import IntProgress


# A metric to all rows of a query
def collect(dataDB, queryDict, metricName, metricParam):
    metricList = []

    rows = dataDB.get_rows('neuro', queryDict)
    nRows = len(rows)
    progBar = IntProgress(min=0, max=nRows, description='Collecting ' + metricName + ' for data')
    display(progBar)  # display the bar
    for idx, row in rows.iterrows():
        metricList += [metric3D(dataDB.dataNeuronal[idx], metricName, metricParam)]
        progBar.value += 1
    return metricList


# Plot data along one axis
def plot_1D(ax, dataList, axis=None, label=None):
    # Step 1: flatten all individual datasets to 1 axis of choice
    if axis is not None:
        dataFlatList = [np.nanmean(data, axis=unique_subtract(data.shape, (axis,))) for data in dataList]
    else:
        dataFlatList = dataList

    # Step 2: pad ends of all data so that they are comparable
    # TODO: replace this mechanism with parameter alignment (e.g. temporal)
    nData = len(dataFlatList)
    nPointMax = np.max([len(data) for data in dataFlatList])
    tmpArr = np.full((nData, nPointMax), np.nan)
    for i, data in enumerate(dataFlatList):
        tmpArr[i, :len(data)] = data

    # Step 3: Find mean and variance, plot
    x = np.arange(nPointMax)
    mu = np.nanmean(tmpArr, axis=0)
    std = np.nanstd(tmpArr, axis=0)

    ax.fill_between(x, mu-std, mu+std, alpha=0.3)
    ax.plot(x, mu, label=label)

