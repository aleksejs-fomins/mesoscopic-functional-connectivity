import numpy as np
import matplotlib.pyplot as plt

# Convert y-axis from fractions to percent
def set_percent_axis_y(ax):
    ax.set_yticklabels(['{:,.2%}'.format(x) for x in ax.get_yticks()])


# Plot a histogram from integer data
def hist_int(ax, x, labels=None, xmin=None, xmax=None):
    xmin = np.min(x) if xmin is None else xmin
    xmax = np.max(x) if xmax is None else xmax

    bins = np.arange(xmin, xmax + 1)
    bins_ext = np.arange(xmin - 1, xmax + 1) + 0.5

    if labels is None:
        ax.hist(x, bins=bins_ext, rwidth=0.5, density=True)
    else:
        for xThis, label in zip(x, labels):
            ax.hist(x, bins=bins_ext, rwidth=0.5, density=True, alpha=0.5, label=label)
        ax.legend()

    ax.set_xticks(bins)
    set_percent_axis_y(ax)
