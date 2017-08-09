"""
This file is part of spinsys.

Spinsys is free software: you can redistribute it and/or modify
it under the terms of the BSD 3-clause license. See LICENSE.txt
for exact terms and conditions.


This module provides functions working on half-spin systems.
Functions included:
    create_complete_basis
    full_matrix
    reorder_basis
    similarity_trans_matrix
"""

import numpy as np
from scipy import misc, sparse
from spinsys import utils
from spinsys.utils.cache import Globals as G
from spinsys.exceptions import SizeMismatchError


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
        blksize = int(round(misc.comb(N, spin_ups)))
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
    if not sparse.issparse(matrix):
        S = sparse.csc_matrix(matrix)
    else:
        S = matrix
    if k == 0:
        S_full = sparse.kron(S, sparse.eye(dim ** (N - 1)))
    elif k == 1:
        S_full = sparse.eye(dim)
        S_full = sparse.kron(S_full, S)
        S_full = sparse.kron(S_full, sparse.eye(dim ** (N - 2)))
    else:
        S_full = sparse.eye(dim)
        S_full = sparse.kron(S_full, sparse.eye(dim ** (k - 1)))
        S_full = sparse.kron(S_full, S)
        S_full = sparse.kron(S_full, sparse.eye(dim ** (N - k - 1)))

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
    # provides compatibility with both 1D and 2D 'vectors'
    psi_diag = psi_diag.flatten()
    # uses csc_matrix for efficient reordering of the vector. Reshape at
    #  the end ensures the vector comes out to be a normal 1D vector
    psi_ord = sparse.csc_matrix((psi_diag, indices, [0, veclen]),
                                   shape=[2 ** N, 1]).toarray().reshape(2 ** N)
    return psi_ord


def bipartite_reduced_density_op(N, state):
    """
    Creates the density matrix using a state. Useful for calculating
    bipartite entropy

    Args: "state" a column vector that has to be dense (numpy.array)
          "N" total system size
    Returns: Numpy array
    """
    dim = 2 ** (N // 2)            # Dimensions of the reduced density matrices
    if not max(state.shape) == dim ** 2:
        error_msg = 'Did you forget to expand the state into the full' + \
            'Hilbert space?'
        raise SizeMismatchError(error_msg)
    reshaped_state = np.reshape(state, [dim, dim])
    return np.dot(reshaped_state, reshaped_state.conjugate().transpose())


def reduced_density_op(N, sysA, state, curr_j=0):
    """Creates the density matrix using a state. Useful for calculating
    non-bipartite i.e. arbitrary cut entanglement entropy

    Args: "N" total system size
          "sysA" sites along the chain that belong to system A, the system
                 we are interested in (Python list)
          "state" a column vector that has to be dense (numpy.array)
          "curr_j" total spin
    Returns: Numpy array
    """
    # TODO: Partially working. Passed 2-leg bipartite configurations in both
    #  directions. Discrepencies found in configurations involving more than
    #  two legs.
    # @utils.cache.cache_ram
    def reorder_basis_dict(N, sysA, curr_j):
        A_len = len(sysA)
        B_len = N - A_len
        sysB = [i for i in range(N) if i not in sysA]
        # Possible spin configurations of sysA, in 1's and 0's
        sysA_configs = [list(map(int, bin(i)[2:])) for i in
                        range(2 ** A_len - 1, -1, -1)]
        sysA_configs = [[0] * (A_len - len(c)) + c for c in sysA_configs]
        # Possible spin configurations of sysB, in 1's and 0's
        sysB_configs = [list(map(int, bin(i)[2:])) for i in
                        range(2 ** B_len - 1, -1, -1)]
        sysB_configs = [[0] * (B_len - len(c)) + c for c in sysB_configs]
        # The full basis set when we merge the above configurations. Now in
        #  our desired order.
        full_basis = [list(zip(*sorted(zip(sysA + sysB, bA + bB))))[1]
                      for bA in sysA_configs for bB in sysB_configs]
        # Indices indicating the new locations of the vector elements.
        indices = [hilbert_dim - utils.misc.bin_to_dec(b) - 1
                   for i, b in enumerate(full_basis)]
        return indices

    dim = 2 ** (N // 2)            # Dimensions of the reduced density matrices
    if not max(state.shape) == dim ** 2:
        error_msg = 'Did you forget to expand the state into the full' + \
            'Hilbert space?'
        raise SizeMismatchError(error_msg)

    hilbert_dim = 2 ** N
    indices = reorder_basis_dict(N, sysA, curr_j)
    reordered_vec = sparse.csc_matrix((state, indices, [0, hilbert_dim]),
                                       shape=[2 ** N, 1]).toarray() \
                                       .reshape(2 ** N)

    A_len = len(sysA)
    B_len = N - A_len
    reshaped_state = np.reshape(reordered_vec, [2 ** A_len, 2 ** B_len])
    return reshaped_state.dot(reshaped_state.T.conjugate())


def block_diagonalization_transformation(N):
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
        blksize = int(round(misc.comb(N, spin_ups)))
        to_diag = generate_complete_basis(N, current_j)[1]
        for ord_ind, diag_ind in to_diag.items():
            row_ind[current_pos] = diag_ind + offset
            col_ind[current_pos] = ord_ind
            current_pos += 1
        offset += blksize
    return sparse.csc_matrix((data, (row_ind, col_ind)), shape=(dim, dim))
