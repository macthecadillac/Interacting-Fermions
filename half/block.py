"""This module provides functions for working with block diagonalized
hamiltonians. Functions provided:
    reorder_basis
    similarity_trans_matrix

1-13-2017
"""

import numpy as np
import scipy as scp
from . import misc


def reorder_basis(N, psi_diag, current_j=0):
    """
    Reorders the basis of a vector from one arranged by their total <Sz>
    to one that results from tensor products.

    Args: "N" System size
          "psi_diag" State in a block diagonalized basis arrangement
          "current_j" Total <Sz>
    Returns: Numpy 2D array (column vector)
    """
    psi_ord = np.zeros([2 ** N, 1], complex)
    to_ord = misc.create_complete_basis(N, current_j)[2]
    for i in psi_diag.nonzero()[0]:
        psi_ord[to_ord[i], 0] = psi_diag[i, 0]
    return psi_ord


def similarity_trans_matrix(N):
    """
    Returns a matrix U such that Uv = v' with v in the tensor product
    basis arrangement and v' in the spin block basis arrangement.

    Args: "N" System size
    Returns: Sparse matrix (CSC matrix)
    """
    offset = 0
    dim = 2 ** N
    data = np.ones(dim)
    row_ind = np.empty(dim)
    col_ind = np.empty(dim)
    current_pos = 0                     # current position along the data array
    for current_j in np.arange(N / 2, -N / 2 - 1, -1):
        spin_ups = round(0.5 * N + current_j)
        blksize = int(round(scp.misc.comb(N, spin_ups)))
        to_diag = misc.create_complete_basis(N, current_j)[1]
        for ord_ind, diag_ind in to_diag.items():
            row_ind[current_pos] = diag_ind + offset
            col_ind[current_pos] = ord_ind
            current_pos += 1
        offset += blksize
    return scp.sparse.csc_matrix((data, (row_ind, col_ind)), shape=(dim, dim))
