import numpy as np

from codes.lib.aux_functions import perm_map_str

import npeet.entropy_estimators as ee


# Add extra dimensions of size 1 to array at given locations
def numpy_reshape_extradim(x, reducedAxis):
    newShape = list(x.shape)
    for axis in reducedAxis:
        newShape.insert(axis, 1)
    return x.reshape(tuple(newShape))



# Transpose data dimensions given permutation of axis labels
def data_transpose_byorder(data, orderSrc, orderTrg):
    if sorted(orderSrc) != sorted(orderTrg):
        raise ValueError('Cannot transform', orderSrc, "to", orderTrg)
    return data.transpose(perm_map_str(orderSrc, orderTrg))


# Reshape array by merging all dimensions between l and r
def numpy_merge_dimensions(data, l, r):
    shOrig = list(data.shape)
    shNew = tuple(shOrig[:l] + [np.prod(shOrig[l:r])] + shOrig[r:])
    return data.reshape(shNew)


# Compute metrics individually for each channel
def npeet_metric_1D_generic(method, data, settings):
    assert data.shape[settings['dim_order'].index("p")] == 1, "Expected only 1 channel for this estimate"
    methods1D = {
        'Entropy' : entropy,
        'PI'      : predictive_info
    }
    return methods1D[method](data, settings)


# Compute 1 metric for all channels
def npeet_metric_ND_generic(method, data, settings):
    methodsND = {
        'Entropy' : entropy,
        'PI'      : predictive_info
    }
    return methodsND[method](data, settings)


def entropy(data, settings):
    dataCanon = data_transpose_byorder(data, settings['dim_order'], 'srp')
    dataFlat = numpy_merge_dimensions(dataCanon, 0, 2)
    rez = ee.entropy(dataFlat)
    return numpy_reshape_extradim(rez, [1])  # Need extra dimension for number of results (1 in this case)


# Predictive information
# Defined as H(Future) - H(Future | Past) = MI(Future : Past)
def predictive_info(data, settings):
    dataCanon = data_transpose_byorder(data, settings['dim_order'], 'srp')
    nTime = dataCanon.shape[0]
    lag = settings['max_lag']

    x = np.array([dataCanon[i:i+lag] for i in range(nTime - lag)])

    # shape transform for x :: (swrp) -> (s*r, w*p)
    x = x.transpose((0,2,1,3))
    x = numpy_merge_dimensions(x, 2, 4)  # p*w
    x = numpy_merge_dimensions(x, 0, 2)  # s*r

    # shape transform for y :: (srp) -> (s*r, p)
    y = numpy_merge_dimensions(dataCanon[lag:], 0, 2)

    rez = ee.mi(x, y)
    return numpy_reshape_extradim(rez, [1])  # Need extra dimension for number of results (1 in this case)