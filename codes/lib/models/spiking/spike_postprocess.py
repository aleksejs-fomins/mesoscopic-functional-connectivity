import h5py
import numpy as np
import matplotlib.pyplot as plt

# COMPUTE ENERGY TRANSFER and save to HDF5
# TODO: IMPLEMENT TIME_DELAY INTO ENERGY TRANSFER
def computeEnergyTransfer(filepathname, withPlot = False):

    # Open and read the file
    h5f = h5py.File(filepathname, "r+")
    POP_IDX_END = h5f['metadata/POP_IDX_END']
    POPULATION_CONN_EXT = h5f['metadata/POPULATION_CONN_EXT']
    NEURON_CONN = h5f['metadata/NEURON_CONN']
    W_EXC_MAX_EXT = h5f['metadata/W_EXC_MAX_EXT']
    h5f.close()
    
    #######################################
    # 1. Extract inter-layer connectivity
    #######################################

    # Learn to map from neuron index to population index
    # Convert to zero-starting index
    maxNeuronIdx = POP_IDX_END - 1
    N_NEURON_TOT = maxNeuronIdx[-1]  # There are also spike detectors and such that come with larger indices
    N_CONN_EXTERN = len(POPULATION_CONN_EXT)
    
    # Find out how many times each neuron maps to another region
    connMap = np.zeros((N_NEURON_TOT, N_CONN_EXTERN))
    for src, trg in NEURON_CONN:
        # Find indices of connected populations
        popSrc = np.searchsorted(POP_IDX_END, src)
        popTrg = np.searchsorted(POP_IDX_END, trg)
        connPopIdxs = [popSrc, popTrg]
        
        # Find index of this connection among external connections
        connPopIdx = np.where(np.all(POPULATION_CONN_EXT == connPopIdxs, axis=1))[0]
        
        # If it exists (that is, if the connection is indeed external and not internal),
        # increase input from source neuron to target population
        # NOTE: Nest numbers neurons starting from 1
        if len(connPopIdx) != 0:
            connMap[src-1][connPopIdx[0]] += W_EXC_MAX_EXT

    ######################################################
    # 2. Convert spikes into cross-layer energy transfer
    ######################################################

    MIN_TIME = 0
    MAX_TIME = self.tstop
    times_discr = np.arange(MIN_TIME, MAX_TIME + self.dt, self.dt)
    micro_energy = np.zeros((N_CONN_EXTERN, len(times_discr)))

    for iPop in range(self.N_POPULATIONS):
        # Only consider excitatory population data
        if (iPop % 2 == 0):
            tspk = nest.GetStatus(self.neuron_spike_detectors[iPop])[0]['events']['times']
            nspk = nest.GetStatus(self.neuron_spike_detectors[iPop])[0]['events']['senders']

            for t, n in zip(tspk, nspk):
                timeIdx = int((t - MIN_TIME) / self.dt)
                for iLink in range(N_CONN_EXTERN):
                    micro_energy[iLink][timeIdx] += connMap[n - 1][iLink]

    # Write to file
    f = h5py.File(self.basename + ".hdf5", "r+")
    f['ENERGY_TRANSFER_T'] = times_discr
    f['ENERGY_TRANSFER_X'] = micro_energy
    f.close()

    # Plot if requested
    if withPlot:
        plt.figure()
        for i in range(len(micro_energy)):
            plt.plot(times_discr, micro_energy[i])


def saveGCampTraces(self, param, withPlot = False):
    # Coarsen the input to account for finite camera framerate
    # Delete last time point because it is incomplete due to coarsening
    MIN_TIME = 0
    MAX_TIME = self.tstop
    BINSIZE = 1000 / param['FPS']
    tCoarse = np.linspace(MIN_TIME, MAX_TIME, int((MAX_TIME-MIN_TIME)/BINSIZE)+1)
    xCoarseLst = np.zeros((self.N_POPULATIONS, len(tCoarse)-1))

    # Write to file
    f = h5py.File(self.basename + ".hdf5", "r+")
    f['GCAMP_COARSE_T'] = tCoarse[:-1]


    # Plot if requested
    if withPlot:
        plt.figure()

    for iPop in range(self.N_POPULATIONS):
        tspk = nest.GetStatus(self.neuron_spike_detectors[iPop])[0]['events']['times']
        nspk = nest.GetStatus(self.neuron_spike_detectors[iPop])[0]['events']['senders']

        dataFine = spike2ca(tspk, nspk, self.dt, param['TAU_CA'], MIN_TIME, MAX_TIME, param['GEOM_RANGE'])

        xCoarse = np.zeros(len(tCoarse))
        for t, x in zip(dataFine[0], dataFine[1]):
            xCoarse[int((t - MIN_TIME) / BINSIZE)] += x / BINSIZE

        # Write to file
        xCoarseLst[iPop] = xCoarse[:-1]

        # Plot if requested
        if withPlot:
            plt.plot(tCoarse[:-1], xCoarse[:-1])

    # Close HDF5
    f['GCAMP_COARSE_X'] = xCoarseLst
    f.close()