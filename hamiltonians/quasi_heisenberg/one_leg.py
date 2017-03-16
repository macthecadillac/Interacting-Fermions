"""Provides functions that generate the hamiltonian. Open or
periodic boundary conditions.

1-17-2017
"""

import numpy as np
from scipy.misc import comb
from spinsys.utils.globalvar import Globals as G
import spinsys.constructors as cn
from spinsys import half


def aubry_andre_H(N, h, c, phi, J=1, mode='open'):
    """Generates the full hamiltonian

    Args: "N" number of sites
          "h" disorder strength
          "c" trancendental number in the quasi-periodic field
          "phi" phase
          "J" coupling constant for nearest neighbors
          "mode" "open" or "periodic" boundary conditions
    Returns: csc_matrix
    """
    # try to load Sx, Sy, Sz from the Globals dictionary if they have been
    #  generated
    try:
        Sx, Sy, Sz = G['sigmas']
    except KeyError:
        Sx, Sy, Sz = cn.sigmax(), cn.sigmay(), cn.sigmaz()
        G['sigmas'] = [Sx, Sy, Sz]

    # try to load the full Hilbert space operators from the Globals
    #  dictionary if they have been generated
    if not G.__contains__('full_S'):
        G['full_S'] = {}
    try:
        full_S = G['full_S'][N]
    except KeyError:
        full_S = [[half.full_matrix(S, k, N) for k in range(N)]
                  for S in [Sx, Sy, Sz]]
        G['full_S'][N] = full_S

    lbound = 0 if mode == 'periodic' else 1
    # contributions from nearest neighbor interations
    inter_terms = J * sum(full_S[i][j - 1] * full_S[i][j] for j in range(lbound, N)
                          for i in range(3))
    # contributions from the disorder field
    field = h * np.cos(2 * np.pi * c * np.arange(1, N + 1) + phi)
    field_terms = sum(field * full_S[2])
    H = inter_terms + field_terms
    return H.real


def block_diagonalized_H(N, h, c, phi, J=1, mode='open'):
    """Generates the block diagonalized full hamiltonian

    Args: "N" number of sites
          "h" disorder strength
          "c" trancendental number in the quasi-periodic field
          "phi" phase
          "J" coupling constant for nearest neighbors
          "mode" "open" or "periodic" boundary conditions
    Returns: csc_matrix
    """
    H = aubry_andre_H(N, h, c, phi, J, mode)
    # Try to load the similarity transformation matrix from memory.
    #  Generates the matrix if doesn't yet exist
    if not G.__contains__('similarity_trans_matrix'):
        G['similarity_trans_matrix'] = {}
    try:
        U = G['similarity_trans_matrix'][N]
    except KeyError:
        U = half.similarity_trans_matrix(N)
        G['similarity_trans_matrix'][N] = U
    return U * H * U.T


def spin_block(N, h, c, phi, curr_j=0, J=1, mode='open'):
    """Generates the block in the block diagonalized hamiltonian with
    the specific total <Sz>.

    Args: "N" number of sites
          "h" disorder strength
          "c" trancendental number in the quasi-periodic field
          "phi" phase
          "curr_j" total <Sz>
          "J" coupling constant for nearest neighbors
          "mode" "open" or "periodic" boundary conditions
    Returns: csc_matrix
    """
    # Slice out the center block from the full block diagonalized hamiltonian
    offset = sum(comb(N, j, exact=True) for j in np.arange(0.5 * N - curr_j))
    blk_size = comb(N, round(0.5 * N + curr_j), exact=True)
    H = block_diagonalized_H(N, h, c, phi, J, mode)
    return H[offset:offset + blk_size, offset:offset + blk_size]
