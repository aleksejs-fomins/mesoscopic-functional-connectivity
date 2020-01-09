'''
The following functions provide iterators over slices of a dataset

Modifiers:
    * 1D             - (True) sweep over each channel                           (False) Multidimensional over channels
    * window_pooled  - (True) Time-points within a window are pooled together   (False) Multidimensional over window points
    * time_pooled    - (True) All windows from sweep are pooled together        (False) Sweep over windows
'''



# Provide slices of a dataset of shape [nNode, nStep]
# If window parameter is provided, a sweep is performed over time dimension
# Otherwise, trial and time dimensions are concetenated
def sweep_generator(data, settings):
    nNode, nTime, nTrial = data.shape()
    if settings['window'] is None:
        yield data.reshape(nNode, nTime * nTrial)
    else:
        w = settings['window']
        for iTime in range(nTime - w):
            yield data[:, iTime : iTime + w, :].reshape((nNode, w * nTrial))


# Provide slices of a dataset of shape [nStep], individually for each channel
# If window parameter is provided, a sweep is performed over time dimension
# Otherwise, trial and time dimensions are concetenated
def sweep_generator_1D(data, settings):
    nNode, nTime, nTrial = data.shape()
    for iNode in range(nNode):
        if settings['window'] is None:
            yield data[iNode].flatten()
        else:
            w = settings['window']
            for iTime in range(nTime - w):
                yield data[iNode, iTime : iTime + w, :].flatten()