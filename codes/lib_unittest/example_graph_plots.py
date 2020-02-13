import os
import sys

# Export library path
rootname = "mesoscopic-functional-connectivity"
thispath = os.path.dirname(os.path.abspath(__file__))
rootpath = os.path.join(thispath[:thispath.index(rootname)], rootname)
print("Appending project path", rootpath)
sys.path.append(rootpath)

from codes.lib.plots.directed_graph import plotGraph


N_REGION = 5
LL_CONN_REGIONS = [(0, 1), (1, 2), (3, 1), (4, 2)]
LL_CONN_POPULATIONS = [(2*i, 2*j) for i,j in LL_CONN_REGIONS]
CONN_GRAPH_POPULATIONS = []
    
    
# Construct connectivity within each layer
for i in range(N_REGION):
    CONN_GRAPH_POPULATIONS += [
        (2*i,   2*i,   None),
        (2*i,   2*i+1, None),
        (2*i+1, 2*i,   None),
        (2*i+1, 2*i+1, None)
    ]

# Construct inter-layer connectivity
for i,j in LL_CONN_POPULATIONS:    
    CONN_GRAPH_POPULATIONS += [(i, j, None)]

plotGraph(CONN_GRAPH_POPULATIONS, N_REGION*2, "graph.pdf")