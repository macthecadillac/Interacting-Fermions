"""
This file is part of spinsys.

Spinsys is free software: you can redistribute it and/or modify
it under the terms of the BSD 3-clause license. See LICENSE.txt
for exact terms and conditions.


Provides functions that generate the hamiltonian. Open or
periodic boundary conditions.
"""

from spinsys.utils.cache import Globals as G
import spinsys.constructors as cn
from spinsys import half


def H(N, h=0, J=1, mode='open'):
    """Generates the full hamiltonian

    Args: "N" number of sites
          "h" disorder strength
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
    field_terms = sum(h * full_S[2])
    H = inter_terms + field_terms
    return H.real
