"""This module provides functions working on half-spin systems.
Functions included:
    create_complete_basis
    full_matrix
    reorder_basis
    similarity_trans_matrix

1-16-2017
"""

import numpy as np
import scipy as sp
from spinsys import utils
from spinsys.utils.globalvar import Globals as G


def generate_complete_basis(N, current_j):
    """Creates a complete basis for the current total <Sz>"""
    # instantiate a dict if it doesn't exist
    if not G.__contains__('complete_basis'):
        G['complete_basis'] = {}
    if not G['complete_basis'].__contains__(N):
        G['complete_basis'][N] = {}
    # reuse generated results if already exists
    try:
        basis_set, to_diag, to_ord = G['complete_basis'][N][current_j]
    except KeyError:
        dim = 2 ** N
        spin_ups = int(round(0.5 * N + current_j))
        spin_downs = N - spin_ups
        blksize = int(round(sp.misc.comb(N, spin_ups)))
        basis_seed = [0] * spin_downs + [1] * spin_ups
        basis = basis_seed
        # "to_diag" is a dict that maps ordinary indices to block diagonalized
        #  indices. "to_ord" is the opposite.
        basis_set, to_diag, to_ord = [], {}, {}
        for i in range(blksize):
            try:
                basis = utils.misc.binary_permutation(basis)
            except IndexError:            # When current_j is N // 2 or -N // 2
                pass
            basis_set.append(basis[:])
            decimal_representation = utils.misc.bin_to_dec(basis)
            # i is the index within only this block
            to_diag[dim - decimal_representation - 1] = i
            to_ord[i] = dim - decimal_representation - 1
        G['complete_basis'][N][current_j] = (basis_set, to_diag, to_ord)
    return basis_set, to_diag, to_ord


def full_matrix(matrix, k, N):
    """
    Builds the S matrices in an N particle system. Assumes periodic boundary
    condition.
    "S" could be an operator/state we want to work on. If it is a state, it
    must be put in a column vector form. "S" must be sparse.
    "k" is the location index of the particle in a particle chain. The first
    particle has k=0, the second has k=1 and so on.
    Returns a sparse matrix.
    """
    dim = 2
    if not sp.sparse.issparse(matrix):
        S = sp.sparse.csc_matrix(matrix)
    else:
        S = matrix
    if k == 0:
        S_full = sp.sparse.kron(S, sp.sparse.eye(dim ** (N - 1)))
    elif k == 1:
        S_full = sp.sparse.eye(dim)
        S_full = sp.sparse.kron(S_full, S)
        S_full = sp.sparse.kron(S_full, sp.sparse.eye(dim ** (N - 2)))
    else:
        S_full = sp.sparse.eye(dim)
        S_full = sp.sparse.kron(S_full, sp.sparse.eye(dim ** (k - 1)))
        S_full = sp.sparse.kron(S_full, S)
        S_full = sp.sparse.kron(S_full, sp.sparse.eye(dim ** (N - k - 1)))

    return S_full


def expand_and_reorder(N, psi_diag, current_j=0):
    """
    Expands and reorders the basis of a vector from one arranged by its
    total <Sz> to the tensor product full Hilbert space.

    Args: "N" System size
          "psi_diag" State in a block diagonalized basis arrangement
          "current_j" Total <Sz>
    Returns: Numpy 1D vector
    """
    to_ord = generate_complete_basis(N, current_j)[2]
    veclen = max(psi_diag.shape)
    indices = [to_ord[i] for i in range(veclen)]
    # uses csc_matrix for efficient reordering of the vector. Reshape at
    #  the end ensures the vector comes out to be a normal 1D vector
    psi_ord = sp.sparse.csc_matrix((psi_diag, indices, [0, veclen]),
                                   shape=[2 ** N, 1]).toarray().reshape(2 ** N)
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
        blksize = int(round(sp.misc.comb(N, spin_ups)))
        to_diag = generate_complete_basis(N, current_j)[1]
        for ord_ind, diag_ind in to_diag.items():
            row_ind[current_pos] = diag_ind + offset
            col_ind[current_pos] = ord_ind
            current_pos += 1
        offset += blksize
    return sp.sparse.csc_matrix((data, (row_ind, col_ind)), shape=(dim, dim))
