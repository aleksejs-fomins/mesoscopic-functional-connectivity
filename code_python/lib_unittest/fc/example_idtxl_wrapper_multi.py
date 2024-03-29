import matplotlib.pyplot as plt

from lib.info_metrics.info_metrics_generic import parallel_metric_2d
from lib.models.test_lib import dynsys
from lib.sweep_lib import DataSweep


############################
# Parameters
############################

# DynSys parameters
dynsysParam = {
    'nNode'   : 4,    # Number of variables
    'nData'   : 4000,  # Number of timesteps
    'nTrial'  : 20,    # Number of trials
    'dt'      : 50,    # ms, timestep
    'tau'     : 500,   # ms, timescale of each mesoscopic area
    'inpT'    : 100,   # Period of input oscillation
    'inpMag'  : 0.0,   # Magnitude of the periodic input
    'std'     : 0.2,   # STD of neuron noise
}

# IDTxl parameters
idtxlParam = {
    'dim_order'       : 'rps',
    'cmi_estimator'   : 'JidtGaussianCMI',
    'max_lag_sources' : 5,
    'min_lag_sources' : 1,
    'window' : 50
}

############################
# Data
############################

nSweep = 10
data = dynsys(dynsysParam)   #[trial x channel x time]
print("Generated data of shape", data.shape)

methods = ['BivariateTE', 'MultivariateTE']
dataSweep1 = DataSweep(data, idtxlParam, nSweepMax=nSweep)
timeIdxs = dataSweep1.get_target_time_idxs()

# print(timeIdxs)
#
# from codes.lib.sweep_lib import Sweep2D
#
# sweeper = Sweep2D(dataSweep1.iterator(), methods, idtxlParam["dim_order"], parTarget=True)
#
# for i, (method, data, iTrg) in enumerate(sweeper.iterator()):
#     print(i, method, data.shape, iTrg)


results = parallel_metric_2d(dataSweep1.iterator(), "idtxl", methods, idtxlParam, nCore=None)

fig, ax = plt.subplots(nrows=nSweep, ncols=2)
fig.suptitle("TE computation for several windows of the data")
for iMethod, method in enumerate(methods):
    ax[0][iMethod].set_title(method)

    print(results[method].shape)

    for iSweep in range(nSweep):
        ax[iSweep][0].set_ylabel("time="+str(timeIdxs[iSweep]))
        ax[iSweep][iMethod].imshow(results[method][iSweep][0])
plt.show()