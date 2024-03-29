import matplotlib.pyplot as plt

from lib.info_metrics.info_metrics_generic import parallel_metric_2d
from lib.models.test_lib import dynsys


############################
# Generate Data
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

data = dynsys(dynsysParam)   #[trial x channel x time]
print("Generated data of shape", data.shape)

############################
# Compute Functional Conn
############################

# IDTxl parameters
idtxlParam = {
    'dim_order'       : 'rsp',
    'cmi_estimator'   : 'JidtGaussianCMI',
    'max_lag_sources' : 5,
    'min_lag_sources' : 1
}

dataEff = data.transpose((0, 2, 1))[:, -200:, :]

print(dataEff.shape)

rezDict = parallel_metric_2d([dataEff], "idtxl", ['BivariateTE'], idtxlParam, parTarget=True, serial=False, nCore=None)
te, lag, p = rezDict['BivariateTE'][0]

fig, ax = plt.subplots(ncols=3)
ax[0].imshow(te)
ax[1].imshow(lag)
ax[2].imshow(p, vmin=0, vmax=1)
plt.show()