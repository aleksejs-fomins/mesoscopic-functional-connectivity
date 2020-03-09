
'''
    Temporal
        Static          history irrelevant, times stay the same
        Dynamic         history used, times truncated by window
    Spatial Input
        Univariate      1 channel at a time
        Multivariate    all channels at a time
    Spatial Output
        Const           1 number regardless of input
        Linear          1D wrt channels
        Quadratic       2D wrt channels




'''


def metric3D(data3D, metricName, axis=(0,1,2)):
    '''
    A procedure to apply an info-theoretic metric to a 3D data

    :param data3D:      Data of the shape [nTrial, nTime, nChannel]
    :param metricName:  Name of the metric to compute
    :param axis:        Axes that will be integrated over to compute the metric
    :param catDim:      Which dimensions to concatenate for evaluation
    :return:            Array of shape (src/axis)
    '''
    pass