import numpy as np
import matplotlib.pyplot as plt
import h5py

####################################
# Creates a network of consecutively-connected linear first order ODE's
####################################

class DynSys:
    def __init__(self, param):
        self.param = param
        N_DATA, N_NODE, ALPHA = param['N_DATA'], param['N_NODE'], param['ALPHA']
        #ALPHA_RAND = np.linspace(ALPHA, 1.0, N_NODE)

        # Create Interaction matrix
        self.M = np.zeros((N_NODE, N_NODE))
        for i in range(N_NODE):
            self.M[i, i] = ALPHA #ALPHA_RAND[i]

        for i in range(N_NODE-1):
            self.M[i + 1, i] = 1 - ALPHA #ALPHA_RAND[i]

    def compute(self):
        N_DATA, N_NODE = self.param['N_DATA'], self.param['N_NODE']
        MAG, T, STD = self.param['MAG'], self.param['T'], self.param['STD']
        
        self.data = np.zeros((N_NODE, N_DATA))
        for i in range(1, N_DATA):
            self.data[:, i] = self.M.dot(self.data[:, i-1])        # Propagate signal
            self.data[0, i] += MAG * np.sin(2 * np.pi * i / T)     # Input to the first node
            self.data[:, i] += np.random.normal(0, STD, N_NODE)    # Noise to all nodes
            
    def plot(self, draw=False):
        print("DynSys: plotting results")
        fig, ax = plt.subplots(ncols=2, figsize=(15,5))
        ax[0].imshow(self.M)
        ax[0].set_title("Connectivity-matrix")
        for j in range(self.param['N_NODE']):
            ax[1].plot(self.data[j], label=str(j))
        ax[1].legend()
        ax[1].set_title("Dynamics")
        
        if draw:
            plt.draw()
        else:
            plt.show()
    
    def save(self, filename):
        print("DynSys: saving to", filename)
        h5f = h5py.File(filename, "w")
        
        # Write metadata
        grp_meta = h5f.create_group("metadata")
        for key, val in self.param.items():
            grp_meta[key] = val
            
        # Write connectivity matrix
        h5f['conn_mat'] = self.M
        
        # Write dynamics
        h5f['data'] = self.data
            
        h5f.close()
        
