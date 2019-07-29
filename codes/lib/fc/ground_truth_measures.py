import h5py
import matplotlib.pyplot as plt
import numpy as np
import scipy.signal

class metricCalculator():
    def __init__(self, filename):
        self.filename = filename

    def plotEnergyTransfer(self):
        f = h5py.File(self.filename, "r")
        plt.figure()

        LABEL = f['POPULATION_CONN']
        T = f['ENERGY_TRANSFER_T']
        X = f['ENERGY_TRANSFER_X']
        for i in range(len(LABEL)):
            plt.plot(T, X[i], label=str(LABEL[i]))
        plt.legend()
        f.close()

    def plotGCamp(self):
        f = h5py.File(self.filename, "r")
        plt.figure()

        T = f['GCAMP_COARSE_T']
        X = f['GCAMP_COARSE_X']
        for i in range(X.shape[0]):
            plt.plot(T, X[i], label=str(i))
        plt.legend()


MC = metricCalculator('datafile.hdf5')
MC.plotEnergyTransfer()
MC.plotGCamp()
plt.show()