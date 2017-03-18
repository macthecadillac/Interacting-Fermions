"""This module provides functions to generate a N-legged hamiltonian
using tensor products

3-16-2017
"""

import numpy as np
from spinsys import half
from spinsys import constructors as c
from spinsys.utils.globalvar import Globals as G


def H(N, W1, c1, phi1, J1=1, W2=0, c2=0, phi2=0, J2=0, nleg=1, mode='open'):
    """Generates the multi-leg hamiltonian for a 2-D lattice

    Args: "N" total number of sites
          "W1" disorder strength along the horizontal
          "c1" trancendental number in the cosine term in the horizontal
               field term
          "phi1" phase factor for the horizontal field term
          "J1" nearest neighbor coupling constant along the horizontal
          "W2" disorder strength along the vertical
          "c2" trancendental number in the cosine term in the vertical
               field term
          "phi2" phase factor for the vertical field term
          "J2" nearest neighbor coupling constant along the vertical
          "nleg" number of legs i.e. number of horizontal rungs on the
                 lattice
          "mode" "peiodic" or "open" boundary conditions
    Returns: csc_matrix
    """

    # try to load Sx, Sy, Sz from the Globals dictionary if they have been
    #  generated
    try:
        Sx, Sy, Sz = G['sigmas']
    except KeyError:
        Sx, Sy, Sz = c.sigmax(), c.sigmay(), c.sigmaz()
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

    # The following section describes a hamiltonian of a 2-D latice that
    #  we label the following way
    #     0     l2      2*l2    ...  (l1-1)*l2
    #     1     l2+1    2*l2+1  ...  (l1-1)*l2+1
    #     2     l2+2    2*l2+2  ...  (l1-1)*l2+2
    #     ...
    #     l2-1  2*l2-1  3*l2-1  ...  l1*l2-1
    l1 = N // nleg  # number of sites along the "horizontal"
    l2 = nleg       # number of sites along the "vertical" (number of legs)
    # Nearest neighbor contributions along the horizontal direction (along
    #  which J1 is the coupling constant)
    if nleg == N:
        J1 = 0   # interaction=0 if there's only 1 site along direction
    # mode == 'periodic' is self-explanatory. l1==2 and lb1=0 would imply
    #  adjacent sites along adjacent legs interact with each other twice,
    #  not exactly what we are looking for.
    lb1 = 0 if (mode == 'periodic' and not l1 == 2) else l2
    inter_terms1 = J1 * sum(full_S[i][j - l2] * full_S[i][j]
                            for j in range(lb1, N) for i in range(3))

    # Nearest neighbor contributions along the vertical direction (along
    #  which J2 is the coupling constant)
    if nleg == 1:
        J2 = 0   # interaction=0 if there's only 1 site along direction
    # mode == 'periodic' is self-explanatory. l2==2 and js looping over
    #  every index would imply adjacent sites interact with each other
    #  twice, not exactly what we are looking for.
    if (mode == 'periodic' and not l2 == 2):
        # if periodic BC, loop over every site
        js = range(0, l2)
    else:
        # if open BC, skip indices that are at the start of a column
        js = range(1, l2)
    # "vert_os" is the offset -- "distance" in our indices between
    #  adjacent sites along the same rung. It is in general the number
    #  of legs of the lattice.
    inter_terms2 = J2 * sum(full_S[i][vert_os: vert_os + l2][j - 1] *
                            full_S[i][vert_os: vert_os + l2][j]
                            for j in js for vert_os in range(0, N, l2)
                            for i in range(3))

    inter_terms = inter_terms1 + inter_terms2

    # Contributions from the disorder field
    # --- along horizontal direction ---
    if l1 == 1:     # field contrib=0 if length of leg=1 aka 1D
        field1 = np.zeros([1])
    else:
        field1 = W1 * np.cos(2 * np.pi * c1 * np.arange(1, l1 + 1) + phi1)
    # --- along vertical direction ---
    if l2 == 1:      # field contrib=0 if length of leg=1 aka 1D
        field2 = np.zeros([1])
    else:
        field2 = W2 * np.cos(2 * np.pi * c2 * np.arange(1, l2 + 1) + phi2)

    field = field1.repeat(l2) + field2.repeat(l1).reshape(l2, l1).T.flatten()
    field_terms = sum(field * full_S[2])

    # Total
    H = inter_terms + field_terms
    return H.real
