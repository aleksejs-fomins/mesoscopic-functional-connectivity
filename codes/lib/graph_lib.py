import numpy as np
import networkx

# Return index of diagonal elements of a square matrix
def diag_idx(N):
    return np.eye(N, dtype=bool)

# Return index of off-diagonal elements of a square matrix
def offdiag_idx(N):
    return ~np.eye(N, dtype=bool)

# Set diagonal to zero
def offdiag(M):
    return M - np.diag(np.diagonal(M))

# Set diagonal to zero, then normalize
def offdiag_norm(M):
    Mnodiag = offdiag(M)
    Mmax = np.max(Mnodiag)
    return Mnodiag if Mmax == 0 else Mnodiag / Mmax

# Compute ratio of average off-diagonal to average diagonal elements
def diagonal_dominance(M):
    nNode = M.shape[0]
    MDiag1D    = M[diag_idx(nNode)]
    MOffDiag1D = M[offdiag_idx(nNode)]
    
    muDiag     = np.mean(MDiag1D)
    muOffDiag  = np.mean(MOffDiag1D)
    stdOffDiag = np.std(MOffDiag1D)
    
    diagDomMu  = muOffDiag / muDiag
    diagDomStd = stdOffDiag / muDiag
    
    return diagDomMu, diagDomStd

# IN-DEGREE: Number of incoming connections. In weighted version sum of incoming weights
def degree_in(M):
    return np.sum(offdiag_norm(M), axis=0)

# OUT-DEGREE: Number of outcoming connections. In weighted version sum of outcoming weights
def degree_out(M):
    return np.sum(offdiag_norm(M), axis=1)

# TOTAL-DEGREE: Number of connections per node. In weighted version sum of connected weights
def degree_tot(M):
    return degree_in(M) + degree_out(M)

# RECIPROCAL-DEGREE: Number of bidirectional connections. In weighted version sum of geometric averages of both weights
def degree_rec(M):
    Mnrm = offdiag_norm(M)
    return np.sum(np.sqrt(Mnrm*Mnrm.T), axis=0)

def avg_geom(v):
    return np.prod(v) ** (1 / len(v))

# https://networkx.github.io/documentation/stable/reference/algorithms/generated/networkx.algorithms.cluster.clustering.html
def cl_coeff(M, normDegree=True):
    nNode = M.shape[0]
    
    # Check if matrix is nonzero, find maximum, normalize
    Mnorm = offdiag_norm(M)
    if np.max(Mnorm) == 0:
        return np.zeros(nNode)
    
    # Compute dynamic intermediate step
    MnormSqrt3 = Mnorm**(1/3) 
    S2D = MnormSqrt3 + MnormSqrt3.T

    #Cv = 0.5 * np.sum(S2D.dot(S2D) * S2D, axis=0)
    Cv = 0.5 * np.einsum('uv,ui,vj', S2D, S2D, S2D).diagonal()
    
    if normDegree:
        # Compute maximal number of triangles that can be formed
        # with this number of outgoing edges (including factor of
        # 2 for the flipped 3rd edge). Then correct it by
        # subtracting triangles made by reciprocal edges, as
        # those are not really triangles
#         deg_tot = degree_tot(Mnorm)
#         deg_rec = degree_rec(Mnorm)
#         nMaxTriPerNode = deg_tot * (deg_tot - 1) - 2 * deg_rec
        #Cv[Cv > 0] /= nMaxTriPerNode[Cv > 0]  # Avoid dividing by zero
        totDegSq32 = np.sum(S2D, axis=0)**2
        recDegSq32 = np.sum(S2D**2, axis=0)
        norm = totDegSq32 - recDegSq32
        Cv[Cv > 0] /= norm[Cv > 0]  # Avoid dividing by zero
    else:
        # Normalize everything by the same factor - result if
        # all connections were non-zero and had the same magnitude
        nMaxTriPerNode = (nNode - 1) * (nNode - 2) / 2
        Cv /= 8 * nMaxTriPerNode
    
    return Cv
        
        