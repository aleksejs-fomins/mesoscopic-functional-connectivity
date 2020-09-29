import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

def polar(r, phi):
    return r * np.array([np.cos(phi), np.sin(phi)])

def makeArrow(pos, col, ax):
    kw = dict(arrowstyle="Simple,tail_width=1,head_width=8,head_length=8", color=col)
    posPrim = [0.95*pos[0] + 0.05*pos[1], 0.05*pos[0] + 0.95*pos[1]]
    arrow123 = patches.FancyArrowPatch(posPrim[0], posPrim[1], connectionstyle="arc3,rad=.5", **kw)
    ax.add_patch(arrow123)

def plotGraph(graph, nPop, savename=None):
    graph_r = 10
    graph_rc = [50, 80]
    graph_color_type = ['red', 'blue']

    ctype = [i % 2 for i in range(nPop)]  # Cell types
    graph_phi = [2 * np.pi * (i // 2) / (nPop / 2) for i in range(nPop)]
    graph_coord = [polar(graph_rc[ctype[i]], graph_phi[i]) for i in range(nPop)]
    graph_color = [graph_color_type[ctype[i]] for i in range(nPop)]

    # Nodes
    fig, ax = plt.subplots(figsize = (10, 10))
    for i in range(nPop):
        circ = plt.Circle(graph_coord[i], graph_r, color=graph_color[i])
        ax.add_patch(circ)

    # Connections

    for i, j, param in graph:
        color = 'k' if np.abs(i - j) == 1 else 'y'
        makeArrow([graph_coord[i], graph_coord[j]], color, ax)

    plt.axis('off')
    ax.set_title('Population Connectivity')
    ax.set_xlim(-(graph_rc[1]+graph_r), (graph_rc[1]+graph_r))
    ax.set_ylim(-(graph_rc[1]+graph_r), (graph_rc[1]+graph_r))
    
    if savename == None:
        print("Showing plot")
        plt.show()
    else:
        print("Saving plot to", savename)
        plt.savefig(savename, bbox_inches='tight')