import time
import matplotlib.pyplot as plt

# Export library path
import os, sys
thispath = os.path.dirname(os.path.abspath(__file__))
parpath = os.path.abspath(os.path.join(thispath, os.pardir))
sys.path.append(os.path.join(parpath, 'lib/'))

# Import local libraries
import brunelmesoscopic
import spike_postprocess

modelParam = {
    'N_REGIONS'                 : 5,
    'CONN_REGIONS'              : [(0, 1), (1, 2), (3, 1), (4, 2)],

    'N_NEURONS_REGION'          : 1000,
    'PCONN_INTERN'              : [0.2, 0.2, 0.2, 0.2],       # EXC-EXC, EXC-INH, INH-EXC, INH-INH
    'PCONN_EXTERN'              : [0.05, 0.05, 0.05, 0.05],   # One for each mesoscopic connection

    'WSCALE_INTERN'             : 2.0,
    'WSCALE_EXTERN'             : 2.0,
    'SYN_TAU_INTERN'            : 1.5,
    'SYN_TAU_EXTERN'            : 200.0,

    'NOISE_RATE'                : 6000.0,
    'W_NOISE_MAX'               : [10.0] * 10,
    'SYN_TAU_NOISE'             : 0.1,

    'STIM_MAG'                  : 0.0,
    'STIM_FREQ'                 : 50.0
}


# Create the simulation
BM1 = brunelmesoscopic.BrunnelMesoscopic(modelParam)

# Plot Connectivity
BM1.plotConnectivity('conn.pdf')

# Run the simulation
tstop = 1000.    # simulation duration 8sec
simulationParam = {
    'resolution': 0.1,          # time discretization, ms
    'local_num_threads': 8,     # number of parallel threads to use
}

start = time.time()
BM1.run(tstop, simulationParam)
print("Time taken to simulate:", time.time() - start)

# Write results to file
start = time.time()
RESULT_FILE = "data.h5"
BM1.createOutFile(RESULT_FILE)
BM1.saveConnectivity()
BM1.saveTraces()
print("Time taken to write results to file:", time.time() - start)

# # Compute energy transfer and add it to file
# start = time.time()
# spike_postprocess.computeEnergyTransfer(RESULT_FILE)
# print("Time taken compute energy transfer:", time.time() - start)

# # Compute and write GCamp traces to file
# gcampParam = {
#     'TAU_CA'        : 400,  # (ms) Delay of CA indicator
#     'GEOM_RANGE'    : 0.5,  # Ratio of neurons that will be 100% visible to the optical detector
#     'FPS'           : 50.0
# }

# start = time.time()
# BM1.saveGCampTraces(gcampParam, withPlot = False)
# print("Time taken to compute GCamp traces:", time.time() - start)