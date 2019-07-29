# Export library path
import os, sys
thispath = os.path.dirname(os.path.abspath(__file__))
parpath = os.path.abspath(os.path.join(thispath, os.pardir))
sys.path.append(os.path.join(parpath, 'lib/'))

from plots.plot_graph import plotGraph


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