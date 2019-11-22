import numpy as np

# IDTxl libraries
from idtxl.bivariate_mi import BivariateMI
from idtxl.bivariate_te import BivariateTE
from idtxl.data import Data
from idtxl.multivariate_mi import MultivariateMI
from idtxl.multivariate_te import MultivariateTE
# import jpype as jp

# Local libraries
from codes.lib.decorators_lib import redirect_stdout
# from codes.lib.decorators_lib import jpype_sync_thread


# Determine label used by IDTxl results class
def get_source_type_label(method):
    if 'TE' in method:
        return 'selected_sources_te'
    elif 'MI' in method:
        return 'selected_sources_mi'
    else:
        raise ValueError('Unexpected method', method)


# Construct an IDTxl analysis class using its name
def get_analysis_class(methodname):
    # Initialise analysis object
    if methodname == "BivariateMI":
        return BivariateMI()
    elif methodname == "MultivariateMI":
        return MultivariateMI()
    elif methodname == "BivariateTE":
        return BivariateTE()
    elif methodname == "MultivariateTE":
        return MultivariateTE()
    else:
        raise ValueError("Unexpected method", methodname)


# Convert data to IDTxl format
def preprocess_data(data, settings):
    return Data(data, dim_order=settings['dim_order'])


# Convert results structure into set of matrices for better usability
# Returns shape [3 x nSource] for one given target
def parse_single_target(resultClass, nNode, iTrg, method):
    # Determine metric name to be extracted
    sourceTypeLabel = get_source_type_label(method)

    # Initialize results
    rezMat  = np.full((3, nNode), np.nan)

    # Parse data
    rezThis = resultClass.get_single_target(iTrg, fdr=False)

    # If any connections were found, get their data  at all was found
    if rezThis[sourceTypeLabel] is not None:
        te_lst = rezThis[sourceTypeLabel]
        p_lst = rezThis['selected_sources_pval']
        lag_lst = [val[1] for val in rezThis['selected_vars_sources']]
        src_lst = [val[0] for val in rezThis['selected_vars_sources']]
        rezThisZip = zip(src_lst, te_lst, lag_lst, p_lst)

        for iSrc, te, lag, p in rezThisZip:
            rezMat[0, iSrc] = te
            rezMat[1, iSrc] = lag
            rezMat[2, iSrc] = p

    return rezMat


# Convert results structure into set of matrices for better usability
# Returns shape [3 x nSource x nTarget]
def parse_results(results, nNode, method):
    # Determine metric name to be extracted
    sourceTypeLabel = get_source_type_label(method)

    # Initialize results
    rezMat  = np.full((3, nNode, nNode), np.nan)

    # Parse data
    for iTrg in range(nNode):
        resultClass = results[iTrg] if isinstance(results, list) else results
        rezMat[:, :, iTrg] = parse_single_target(resultClass, nNode, iTrg, method)

    return rezMat


# Compute FC for single target
# Returns shape [3 x nSource] for one given target. 3 means [FC, lag, p]
#@jpype_sync_thread
@redirect_stdout
def analyse_single_target(iTrg, data, method, settings):
    # Get number of nodes
    idxNodeShape = settings['dim_order'].index("p")
    nNode = data.shape[idxNodeShape]

    # Perform analysis
    analysisClass = get_analysis_class(method)
    resultClass = analysisClass.analyse_single_target(settings=settings, data=data, target=iTrg)

    # Parse results and return them
    return parse_single_target(resultClass, nNode, iTrg, method)


# Compute FC for all targets
# Returns shape [3 x nSource x nTarget] for one given target. 3 means [FC, lag, p]
#@jpype_sync_thread
@redirect_stdout
def analyse_network(data, method, settings):
    # Get number of nodes
    idxNodeShape = settings['dim_order'].index("p")
    nNode = data.shape[idxNodeShape]

    # Perform analysis
    analysisClass = get_analysis_class(method)
    resultClass = analysisClass.analyse_network(settings=settings, data=data)

    # Parse results and return them
    return parse_results(resultClass, nNode, method)
