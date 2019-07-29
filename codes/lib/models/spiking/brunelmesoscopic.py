import numpy as np
import matplotlib.pyplot as plt
import scipy.signal
import nest
import h5py

# Export library path
import os, sys
thispath = os.path.dirname(os.path.abspath(__file__))
parpath = os.path.abspath(os.path.join(thispath, os.pardir))
sys.path.append(os.path.join(parpath, 'lib/'))

# Import local libraries
from plot_graph import plotGraph
from spike2ca import spike2ca


class BrunnelMesoscopic():
    def __init__(self, param):
        self.param = param

        # POPULATION PROPERTIES
        self.param['N_POPULATIONS'] = 2 * param['N_REGIONS']
        N_E = int(0.8 * param['N_NEURONS_REGION'])
        N_I = int(0.2 * param['N_NEURONS_REGION'])
        self.N_NEURON_PP = [N_E, N_I] * param['N_REGIONS']   # number of neurons in each population

        # Synapse properties
        self.W_EXC_MAX = 1 * param['WSCALE_INTERN']
        self.W_INH_MAX = -4 * param['WSCALE_INTERN']
        self.W_EXC_MAX_EXT = 1 * param['WSCALE_EXTERN']

        # Connectivity
        # Inter-layer connectivity graph
        self.N_CONN_EXTERN = len(param['CONN_REGIONS'])
        self.CONN_POPULATIONS_EXTERN = [(2*i, 2*j) for i,j in param['CONN_REGIONS']]
        self.CONN_POPULATIONS = []

        # Construct connectivity within each layer
        for i in range(param['N_REGIONS']):
            self.CONN_POPULATIONS += [
                (2*i,   2*i,   {'WMAX' : self.W_EXC_MAX, 'PCONN' : param['PCONN_INTERN'][0], 'DELAY' : param['SYN_TAU_INTERN']}),
                (2*i,   2*i+1, {'WMAX' : self.W_EXC_MAX, 'PCONN' : param['PCONN_INTERN'][1], 'DELAY' : param['SYN_TAU_INTERN']}),
                (2*i+1, 2*i,   {'WMAX' : self.W_INH_MAX, 'PCONN' : param['PCONN_INTERN'][2], 'DELAY' : param['SYN_TAU_INTERN']}),
                (2*i+1, 2*i+1, {'WMAX' : self.W_INH_MAX, 'PCONN' : param['PCONN_INTERN'][3], 'DELAY' : param['SYN_TAU_INTERN']})
            ]

        # Construct inter-layer connectivity
        for iReg in range(self.N_CONN_EXTERN):
            i,j = param['CONN_REGIONS'][iReg]
            self.CONN_POPULATIONS += [(2*i, 2*j,   {'WMAX' : self.W_EXC_MAX_EXT, 'PCONN' : param['PCONN_EXTERN'][iReg], 'DELAY' : param['SYN_TAU_EXTERN']})]


    # Plot connectivity graph
    def plotConnectivity(self, savename=None):
        plotGraph(self.CONN_POPULATIONS, self.param['N_POPULATIONS'], savename)

        
    # Construct the graph and run
    def run(self, tstop, simParam):
        self.dt = simParam['resolution']
        self.tstop = tstop

        # reset kernel and set dt
        nest.ResetKernel()
        nest.SetKernelStatus(simParam)

        # create nodes
        self.neuron_nodes = [nest.Create('iaf_psc_alpha', n) for n in self.N_NEURON_PP]
        self.neuron_spike_detectors = [nest.Create('spike_detector') for i in range(self.param['N_POPULATIONS'])]
        self.noise_generator = nest.Create('poisson_generator', params={'rate': self.param['NOISE_RATE']})

        # Connect input to 0th node
        if self.param['STIM_MAG'] > 0.0:
            ac_generator = nest.Create('ac_generator', params={'amplitude': self.param['STIM_MAG'], 'frequency': self.param['STIM_FREQ']})
            nest.Connect(ac_generator, self.neuron_nodes[0], conn_spec='all_to_all')

        # create connections
        for i in range(self.param['N_POPULATIONS']):
            # Connect detectors to populations
            nest.Connect(self.neuron_nodes[i], self.neuron_spike_detectors[i], conn_spec='all_to_all')

            # Connect noise to populations
            nest.Connect(self.noise_generator, self.neuron_nodes[i],
                         conn_spec='all_to_all',
                         syn_spec={'weight': self.param['W_NOISE_MAX'][i], 'delay': self.param['SYN_TAU_NOISE']})

        for i, j, param in self.CONN_POPULATIONS:
            # Connect populations to each other
            nest.Connect(self.neuron_nodes[i], self.neuron_nodes[j],
                         conn_spec={'rule': 'pairwise_bernoulli', 'p': param['PCONN']},
                         syn_spec={'model': 'static_synapse', 'weight': param['WMAX'], 'delay': param['DELAY']})

        # simulate
        nest.Simulate(tstop)


    # Create output HDF5 file
    def createOutFile(self, outpathname):
        self.outpathname = outpathname
        h5f = h5py.File(self.outpathname, "w")
        h5f.close()
        
        
    # Save connectivity to HDF5
    def saveConnectivity(self):
        # Extract start and end indices of each population
        POP_IDX_START = [nest.GetStatus(self.neuron_nodes[i], 'global_id')[0] for i in range(self.param['N_POPULATIONS'])]
        POP_IDX_END = [nest.GetStatus(self.neuron_nodes[i], 'global_id')[-1] for i in range(self.param['N_POPULATIONS'])]
        
        # Extract neuronal connections. Exclude non-neuron connections (e.g. spike detectors)
        MIN_IDX = POP_IDX_START[0]
        MAX_IDX = POP_IDX_END[-1]
        
        NEURON_CONN = []
        nestConn = nest.GetConnections()
        for arr in nestConn:
            if (arr[0] >= MIN_IDX) and (arr[1] >= MIN_IDX) and (arr[0] <= MAX_IDX) and (arr[1] <= MAX_IDX):
                NEURON_CONN.append([arr[0], arr[1]])
        
        h5f = h5py.File(self.outpathname, "r+")
        grp_meta = h5f.create_group("metadata")
        grp_meta['N_POPULATIONS'] = self.param['N_POPULATIONS']
        grp_meta['POP_IDX_START'] = POP_IDX_START
        grp_meta['POP_IDX_END']   = POP_IDX_END
        grp_meta['POPULATION_CONN_EXT'] = self.CONN_POPULATIONS_EXTERN
        grp_meta['NEURON_CONN'] = NEURON_CONN
        grp_meta['dt'] = self.dt
        grp_meta['T_RANGE'] = (0, self.tstop)
        
        grp_meta['W_EXC_MAX'] = self.W_EXC_MAX
        grp_meta['W_INH_MAX'] = self.W_INH_MAX
        grp_meta['W_EXC_MAX_EXT'] = self.W_EXC_MAX_EXT
        h5f.close()

        
    # Save traces to HDF5
    def saveTraces(self):
        h5f = h5py.File(self.outpathname, "r+")
        grp_spike = h5f.create_group("spikes")
        for i in range(self.param['N_POPULATIONS']):
            grp_spike['POP_' + str(i)] = np.array([
                nest.GetStatus(self.neuron_spike_detectors[i])[0]['events']['times'],
                nest.GetStatus(self.neuron_spike_detectors[i])[0]['events']['senders']
            ]).transpose()
        h5f.close()
