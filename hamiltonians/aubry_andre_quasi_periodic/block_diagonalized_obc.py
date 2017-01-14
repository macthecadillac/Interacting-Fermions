import numpy as np
from scipy.misc import comb
from spinsys.utils.globalvar import Globals as G
import spinsys.half.block as blk
import spinsys.constructors as cn
from spinsys.half.misc import full_matrix


def aubry_andre_H(N, h, c, phi, J=1):
    """Open boundary conditions"""
    try:
        Sx, Sy, Sz = G['sigmas']
    except KeyError:
        Sx, Sy, Sz = cn.sigmax(), cn.sigmay(), cn.sigmaz()
        G['sigmas'] = [Sx, Sy, Sz]

    if not G.__contains__('full_S'):
        G['full_S'] = {}
    try:
        full_S = G['full_S'][N]
    except KeyError:
        full_S = [[full_matrix(S, k, N) for k in range(N)]
                  for S in [Sx, Sy, Sz]]
        G['full_S'][N] = full_S

    inter_terms = J * sum(full_S[i][j] * full_S[i][j + 1] for j in range(N - 1)
                          for i in range(3))
    field = h * np.cos(2 * np.pi * c * np.arange(1, N + 1) + phi)
    field_terms = sum(field * full_S[2])
    H = inter_terms + field_terms
    return H.real


def block_diagonalized_H(N, h, c, phi, J=1):
    H = aubry_andre_H(N, h, c, phi, J)
    if not G.__contains__('similarity_trans_matrix'):
        G['similarity_trans_matrix'] = {}
    try:
        U = G['similarity_trans_matrix'][N]
    except KeyError:
        U = blk.similarity_trans_matrix(N)
        G['similarity_trans_matrix'][N] = U
    return U * H * U.T


def spin_block(N, h, c, phi, curr_j=0, J=1):
    offset = sum(comb(N, j, exact=True) for j in np.arange(0.5 * N - curr_j))
    blk_size = comb(N, round(0.5 * N + curr_j), exact=True)
    H = block_diagonalized_H(N, h, c, phi, J)
    return H[offset:offset + blk_size, offset:offset + blk_size]
