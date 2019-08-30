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
    return np.sum(offdiag(M), axis=0)

# OUT-DEGREE: Number of outcoming connections. In weighted version sum of outcoming weights
def degree_out(M):
    return np.sum(offdiag(M), axis=1)

# TOTAL-DEGREE: Number of connections per node. In weighted version sum of connected weights
def degree_tot(M):
    return degree_in(M) + degree_out(M)

# RECIPROCAL-DEGREE: Number of bidirectional connections. In weighted version sum of geometric averages of both weights
def degree_rec(M):
    return np.sum(offdiag(np.sqrt(M*M.T)), axis=0)

def avg_geom(v):
    return np.prod(v) ** (1 / len(v))

# https://networkx.github.io/documentation/stable/reference/algorithms/generated/networkx.algorithms.cluster.clustering.html
def cl_coeff(M, normDegree=True):
    deg_tot = degree_tot(M)
    deg_rec = degree_rec(M)
    deg_ind = deg_tot - deg_rec # number of nodes connected to this node in some way
        
    nNode = M.shape[0]
    Cv = np.zeros(nNode)
    maxM = np.max(M)
    
    if maxM == 0:
        return Cv
    
    Mnorm = M / maxM
  
    MnormSqrt3 = (Mnorm)**(1/3) 
    S2D = MnormSqrt3 + MnormSqrt3.T

    for i in range(nNode):
        for j in range(i+1, nNode):
            for k in range(j+1, nNode):
                # Cv[i] += 3*S2D[i,j]*S2D[i,k]*S2D[j,k]
                Cv[i] += S2D[i,j]*S2D[i,k]*S2D[j,k]
                Cv[j] += S2D[j,i]*S2D[j,k]*S2D[i,k]
                Cv[k] += S2D[k,i]*S2D[k,j]*S2D[i,j]
                
                # triangle_value = avg_geom([Mnorm[i,j], Mnorm[i,k], Mnorm[j,k]])
                # triangle_value += avg_geom([Mnorm[i,j], Mnorm[i,k], Mnorm[k,j]])
                # triangle_value += avg_geom([Mnorm[i,j], Mnorm[k,i], Mnorm[j,k]])
                # triangle_value += avg_geom([Mnorm[i,j], Mnorm[k,i], Mnorm[k,j]])
                # triangle_value += avg_geom([Mnorm[j,i], Mnorm[i,k], Mnorm[j,k]])
                # triangle_value += avg_geom([Mnorm[j,i], Mnorm[i,k], Mnorm[k,j]])
                # triangle_value += avg_geom([Mnorm[j,i], Mnorm[k,i], Mnorm[j,k]])
                # triangle_value += avg_geom([Mnorm[j,i], Mnorm[k,i], Mnorm[k,j]])
                # Cv[i] += triangle_value
                # Cv[j] += triangle_value
                # Cv[k] += triangle_value
    
    for i in range(nNode):
        if deg_ind[i] >= 2:
            if normDegree:
                # Compute maximal number of triangles that can be formed
                # with this number of outgoing edges (including factor of
                # 2 for the flipped 3rd edge). Then correct it by
                # subtracting triangles made by reciprocal edges, as
                # those are not really triangles
                nMaxTriPerNode = deg_tot[i] * (deg_tot[i] - 1)
                nMaxTriRecPerNode = 2 * deg_rec[i]
                Cv[i] /= nMaxTriPerNode - nMaxTriRecPerNode
            else:
                nMaxTriPerNode = (nNode - 1) * (nNode - 2) / 2
                Cv[i] /= 8 * nMaxTriPerNode
    
    return Cv
        
        