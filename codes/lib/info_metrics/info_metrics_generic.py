import numpy as np

from codes.lib.sys_lib import mem_now_as_str
from codes.lib.decorators_lib import time_mem_1starg
import codes.lib.info_metrics.idtxl_wrapper as idtxl_wrapper
import codes.lib.info_metrics.npeet_wrapper as npeet_wrapper
from codes.lib.info_metrics.corr_lib import cross_corr_3D
from codes.lib.parallel_lib import GenericMapper
from codes.lib.sweep_lib import Sweep1D, Sweep2D


'''
Keywords
  * network         - analysis is performed on all pairs of sources and targets, returning square matrix
  * single_target   - analysis is performed on only one target, returning a vector for that target
  * parallel        - parallelize using pathos multiprocessing library
  * multiparam      - sweep over a list of datasets and methods, parallelize over all
    * target        - also sweep and parallelize over each target
    * network       - same as target, but all targets are analysed simultaneously by one local method.
    
Note that some methods do not have a single target implementation, so can't parallelize over targets, only over data and methods.
'''

# This method only receives data for 1 channel
@time_mem_1starg
def metric_1d_single_channel(library, method, data, settings):
    if library == 'npeet':
        return npeet_wrapper.npeet_metric_1D_generic(method, data, settings)
    else:
        raise ValueError("Single target not implemented for", library)


# This method receives all data
@time_mem_1starg
def metric_1d_multi_channel(library, method, data, settings):
    if library == 'npeet':
        return npeet_wrapper.npeet_metric_ND_generic(method, data, settings)
    else:
        raise ValueError("Single target not implemented for", library)


@time_mem_1starg
def metric_2d_single_target(iTrg, library, method, data, settings):
    if library == 'idtxl':
        return idtxl_wrapper.analyse_single_target(iTrg, method, data, settings)
    else:
        raise ValueError("Single target not implemented for", library)


@time_mem_1starg
def metric_2d_network(library, method, data, settings):
    if library == 'idtxl':
        return idtxl_wrapper.analyse_network(method, data, settings)
    elif library == 'corr':
        return cross_corr_3D(data, settings, est=method)
    else:
        raise ValueError("Unexpected library", library)


def parallel_metric_1d(dataLst, library, methods, settings, nCh, parCh=True, serial=False, nCore=None):
    print("Mem:", mem_now_as_str(), "- Start of subroutine")

    ###############################
    # Initialize mapper
    ###############################
    mapper = GenericMapper(serial, nCore=nCore)

    ###############################
    # Initialize Sweeper
    ###############################
    sweeper = Sweep1D(dataLst, methods, settings["dim_order"], parCh=parCh, nCh=nCh)

    ###############################
    # Compute estimators in parallel
    ###############################
    if parCh:
        metric_proxy = lambda method, data : metric_1d_single_channel(library, method, data, settings)
    else:
        metric_proxy = lambda method, data : metric_1d_multi_channel(library, method, data, settings)

    rezLst = mapper.mapMultiArg(metric_proxy, sweeper.iterator())

    ###############################
    # Glue computed matrices
    ###############################
    return sweeper.unpack(rezLst)


# Parallelize FC estimate over targets, datasets and methods
# Returns dict {"method" : array of shape [nData x 3 x nSource x nTarget]}
# User can provide parameter whether to force-analyse each target separately or all simultaneously
def parallel_metric_2d(dataLst, library, methods, settings, parTarget=True, serial=False, nCore=None):
    print("Mem:", mem_now_as_str(), "- Start of subroutine")

    ###############################
    # Initialize mapper
    ###############################
    mapper = GenericMapper(serial, nCore=nCore)

    ###############################
    # Initialize Sweeper
    ###############################
    sweeper = Sweep2D(dataLst, methods, settings["dim_order"], parTarget=parTarget)

    ###############################
    # Compute estimators in parallel
    ###############################
    if parTarget:
        metric_proxy = lambda method, data, iTrg : metric_2d_single_target(iTrg, library, method, data, settings)
    else:
        metric_proxy = lambda method, data : metric_2d_network(library, method, data, settings)

    rezLst = mapper.mapMultiArg(metric_proxy, sweeper.iterator())

    ###############################
    # Glue computed matrices
    ###############################
    return sweeper.unpack(rezLst)