import numpy as np

'''
This library is designed to simulate Ca indicator fluorescent signal,
emitted by a population of cells.

Current Features:
* For each neuron, convolve spikes with decaying exponential
* Weighted-sum signals over all neurons with random coefficients

TODO:
* Noise on final signal
* Baseline-correction error as done in real experiments
* Geometry - sampling only part of the population
* More realistic variability in neuronal expression
* Effects of dendritic and axonal signals, neuropil?
'''

# Take discretized spiking signal, produce macroscopic CA indicator response
# NOTE: NEURON INDICES MUST BE CONTIGUOUS
def spike2ca(spikeTimes, neuronIdxs, p):

    # Simulate variability of Ca indicator response strength,
    # and visibility of individual neurons via random factor
#     neuronVariability = samplingRangeScale(np.random.uniform(0, 1, p['N_NEURON']), p['SAMPLING_RANGE'], 0.1)
    neuronVariability = np.random.normal(1, p['MAX_INTENSITY_STD'], p['N_NEURON'])
    neuronVisibility = np.random.uniform(0, 1, p['N_NEURON'])
    neuronVariability[neuronVisibility > p['RATIO_VISIBLE']] = 0

    # Take average spike trace, considering variability
    times_discr = np.arange(p['MIN_TIME'], p['MAX_TIME'] + p['DT'], p['DT'])
    N_TIME_STEP = len(times_discr)
    signalCaAvg = np.zeros(N_TIME_STEP)

    spikeBins = ((spikeTimes - p['MIN_TIME']) / p['DT']).astype(int)
    neuronIdxsSh = (neuronIdxs - p['MIN_NEUR_ID']).astype(int)
    
    for b, i in zip(spikeBins, neuronIdxsSh):
        signalCaAvg[b] += neuronVariability[i]

    # Convolve time-signal with indicator response
    signalCaAvg = approxDelayConv(signalCaAvg, p['TAU_CA_IND'], p['DT'])

    return times_discr, signalCaAvg

    ###########################################################################
    # Code below is more detailed, for the case of variability in neural response. Much slower though
    ###########################################################################

    # # # Construct indicator response function
    # # indicatorResponse_t = np.arange(0, 6 * TAU_CA_IND, DT)
    # # indicatorResponse_x = np.exp(-indicatorResponse_t / TAU_CA_IND)
    #
    # # Separate data into individual traces
    # times_discr = np.arange(MIN_TIME, MAX_TIME + DT, DT)
    # N_TIME_STEP = len(times_discr)
    #
    # # print(MIN_TIME, MAX_TIME, DT, N_TIME_STEP)
    #
    # signalCaAvg = np.zeros(N_TIME_STEP)
    #
    #
    # for i in range(MIN_NEUR_ID, MAX_NEUR_ID + 1):
    #     # Extract spike times of this neuron
    #     spikeTimesThis = spikeTimes[neuronIdxs == i]
    #     spikeTimeIdxs = ((spikeTimesThis - MIN_TIME) / DT).astype(int)
    #
    #     # Convert them into time-continuous signal
    #     signalDiscr = np.zeros(N_TIME_STEP)
    #     signalDiscr[spikeTimeIdxs] = 1
    #
    #     # Convolve time-signal with indicator response
    #     # signalCa = np.convolve(indicatorResponse_x, signalDiscr)[:N_TIME_STEP]
    #     signalCa = approxDelayConv(signalDiscr, TAU_CA_IND, DT)
    #
    #     # Average over all cells in the area
    #     # Simulate variability of Ca indicator response, and visibility of individual neurons via random factor
    #     signalCaAvg += signalCa / N_NEURON * np.random.uniform(0, 1)
    #
    # return times_discr, signalCaAvg