import numpy as np

from lib.array_lib import numpy_merge_dimensions, numpy_transpose_byorder, test_have_dim

import npeet.entropy_estimators as ee


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


# FIXME: Correct info_metrics_generic to account for missing results dimension
def entropy(data, settings):
    dataCanon = numpy_transpose_byorder(data, settings['dim_order'], 'srp', augment=True)
    dataFlat = numpy_merge_dimensions(dataCanon, 0, 2)
    return ee.entropy(dataFlat)


# Predictive information
# Defined as H(Future) - H(Future | Past) = MI(Future : Past)
# FIXME: Correct info_metrics_generic to account for missing results dimension
def predictive_info(data, settings):
    test_have_dim("autocorr_3D", settings['dim_order'], "s")

    dataCanon = numpy_transpose_byorder(data, settings['dim_order'], 'srp')
    nTime = dataCanon.shape[0]
    lag = settings['max_lag']

    x = np.array([dataCanon[i:i+lag] for i in range(nTime - lag)])

    # shape transform for x :: (swrp) -> (s*r, w*p)
    x = x.transpose((0,2,1,3))
    x = numpy_merge_dimensions(x, 2, 4)  # p*w
    x = numpy_merge_dimensions(x, 0, 2)  # s*r

    # shape transform for y :: (srp) -> (s*r, p)
    y = numpy_merge_dimensions(dataCanon[lag:], 0, 2)

    return ee.mi(x, y)


# Compute the total correlation, normalized by number of processes/channels
def total_correlation(data, settings):
    test_have_dim("TC", settings['dim_order'], "p")

    dataCanon = numpy_transpose_byorder(data, settings['dim_order'], 'psr', augment=True)
    nChannel = dataCanon.shape[0]

    e2Davg = entropy(dataCanon, {"dim_order" : "psr"}) / nChannel
    e1Davg = np.nanmean([entropy(d, {"dim_order" : "sr"}) for d in dataCanon], axis=0)

    return e1Davg - e2Davg