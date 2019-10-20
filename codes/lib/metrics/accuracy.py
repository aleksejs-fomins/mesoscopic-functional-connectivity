import numpy as np

from codes.lib.metrics.graph_lib import offdiag_idx


def accuracyTests(conn, connTrue):
    n = conn.shape[0]
    nOffDiag = n * (n - 1)
    connOffDiag = conn[offdiag_idx(n)]
    connOffDiagTrue = connTrue[offdiag_idx(n)]
    isconn1D = ~np.isnan(connOffDiag)
    isconnTrue1D = ~np.isnan(connOffDiagTrue)

    TP = np.sum(np.logical_and(isconn1D, isconnTrue1D))
    FP = np.sum(np.logical_and(isconn1D, ~isconnTrue1D))
    FN = np.sum(np.logical_and(~isconn1D, isconnTrue1D))
    TN = np.sum(np.logical_and(~isconn1D, ~isconnTrue1D))

    return {
        "TruePositive": TP,
        "FalsePositive": FP,
        "FalseNegative": FN,
        "TrueNegative": TN,
        "Type1": FP / (FP + TP) if FP + TP != 0 else 0,
        "Type2": FN / (FN + TN) if FN + TN != 0 else 0,
        "FalsePositiveRate": FP / nOffDiag,
        "FalseNegativeRate": FN / nOffDiag
    }