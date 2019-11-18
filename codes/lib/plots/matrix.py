from matplotlib import colors
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

def plotMatrix(title, shape, mat_lst, title_lst, lims=None, draw=False, savename=None):
    
    # Create plot matrix
    nRows, nCols = shape
    fig, ax = plt.subplots(nrows=nRows, ncols=nCols, figsize=(5*nRows, 5*nCols))
    fig.suptitle(title)
    
    # Convert plot indices to 1D index
    ax1D = ax.flatten()
    n1D = nRows*nCols
    
    # Plot data
    for i in range(n1D):
        pl = ax1D[i].imshow(mat_lst[i], cmap='jet')
        ax1D[i].set_title(title_lst[i])
        fig.colorbar(pl, ax=ax1D[i])
        
        if lims is not None:
            norm = colors.Normalize(vmin=lims[i][0], vmax=lims[i][1])
            pl.set_norm(norm)

    if savename is not None:
        plt.savefig(savename, dpi=300)

    if draw:
        plt.draw()
    else:
        plt.show()


# Add colorbar to existing imshow
def imshowAddColorBar(fig, ax, img):
    divider = make_axes_locatable(ax)
    cax = divider.append_axes('right', size='5%', pad=0.05)
    fig.colorbar(img, cax=cax, orientation='vertical')