"""
This file is part of spinsys.

Spinsys is free software: you can redistribute it and/or modify
it under the terms of the BSD 3-clause license. See LICENSE.txt
for exact terms and conditions.


Provides functions that generate the hamiltonian. Open or
periodic boundary conditions.

3-28-2017
"""

from spinsys.utils.globalvar import Globals as G
import spinsys.constructors as cn
import numpy as np
from spinsys import half


def flip_Sz(N, g=np.pi/2, eps=0):
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
    if not G.__contains__('full_Sx'):
        G['full_Sx'] = {}
    try:
        full_Sx = G['full_Sx'][N]
    except KeyError:
        full_Sx = [half.full_matrix(Sx, k, N) for k in range(N)]
        G['full_Sx'][N] = full_Sx
    flip_Sz = (g-eps) * sum(2 * full_Sx)
    return flip_Sz