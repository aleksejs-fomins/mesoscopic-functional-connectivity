import numpy as np

from codes.lib.aux_functions import perm_map_str

import npeet.entropy_estimators as ee


# Permute data into canonical form
def data_transpose_canonical(data, settings):
    permShape = perm_map_str(settings['dim_order'], 'psr')
    return data.transpose(permShape)


# Compute entropy individually for each channel
# TODO: Impl parallel version
def entropy1D(data, settings):
    dataCanon = data_transpose_canonical(data, settings)
    nNode, nTime, nTrial = dataCanon.shape()
    taskgen1D = sweep_generator_1D(dataCanon, settings)
    rez = np.array(list(map(ee.entropy, taskgen1D)))
    nSwipe = len(rez) // nNode
    return rez.reshape((nNode, nSwipe))


# Compute entropy of all channels simultaneously
def entropyND(data, settings):
    dataCanon = data_transpose_canonical(data, settings)
    taskgen = sweep_generator(dataCanon, settings)
    return np.array(list(map(ee.entropy, taskgen)))


