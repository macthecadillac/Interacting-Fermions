"""
This file is part of spinsys.

Spinsys is free software: you can redistribute it and/or modify
it under the terms of the BSD 3-clause license. See LICENSE.txt
for exact terms and conditions.


This module provides functions working on half-spin systems.
Functions included:
    generate_complete_basis
    full_matrix
    expand_and_reorder
    reduced_density_op_arbitrary_sys
    reduced_density_op
    block_diagonalization_transformation
"""

import functools
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
    "matrix" could be an operator/state we want to work on. If it is a state, it
    must be put in a column vector form. "matrix" must be sparse.
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


def reduced_density_op(N, sys_size, vec):
    """
    Creates the density matrix using a state. Useful for calculating
    entanglement entropy that does not require an arbitrary cut.

    Parameters:
    --------------------
    N: int
        size of lattice
    sys_size: int
        size of system
    vec: numpy.array
        a column vector that has to be dense

    Returns
    --------------------
    ρ: numpy.array
    """
    hilbert_dim = 2 ** N
    if not max(vec.shape) == hilbert_dim:
        error_msg = 'Did you forget to expand the state into the full' + \
            'Hilbert space?'
        raise SizeMismatchError(error_msg)
    env_size = N - sys_size
    reshaped_state = np.reshape(vec, [2 ** sys_size, 2 ** env_size])
    return np.dot(reshaped_state, reshaped_state.conjugate().transpose())


def reduced_density_op_arbitrary_sys(N, sys, vec):
    """Creates the density matrix using a state. Useful for calculating
    non-bipartite i.e. arbitrary cut entanglement entropy

    Parameters:
    --------------------
    N: int
        size of lattice
    sys: list
        the indices of the sites that is considered part of the system
    vec: numpy.array
        a column vector that has to be dense
    j: int/float
        total spin

    Returns
    --------------------
    ρ: numpy.array
    """
    @functools.lru_cache(maxsize=None)
    def generate_binlists(partition_len):
        configs = [format(i, '0{}b'.format(partition_len)) for i in
                   range(2 ** partition_len - 1, -1, -1)]
        configs = map(list, configs)
        return [list(map(int, config)) for config in configs]

    @functools.lru_cache(maxsize=None)
    def reorder_basis_dict(N, sysstr):
        """Returns a dictionary that maps the old ordering of the sites
        to the new
        """
        env = sorted(set(range(N)) - set(sys))
        # Possible spin configurations of sys and env, in 1's and 0's
        sys_configs = generate_binlists(syslen)
        env_configs = generate_binlists(envlen)
        # The full basis set when we merge the above configurations. Now in
        #  our desired order.
        sites = sys + env
        full_basis = [(sites, sysi + envj) for sysi in sys_configs
                      for envj in env_configs]

        reordered_basis = []
        for basis_state in full_basis:
            site_and_spin = zip(*basis_state)
            sorted_by_site = sorted(site_and_spin)
            basis_state_config = list(zip(*sorted_by_site))[1]
            reordered_basis.append(basis_state_config)

        # Indices indicating the new locations of the vector elements.
        # orig_ind = range(hilbert_dim)
        new_ind = [hilbert_dim - utils.misc.bin_to_dec(b) - 1
                   for b in reordered_basis]
        return new_ind

    syslen = len(sys)
    envlen = N - syslen
    hilbert_dim = 2 ** N
    if not max(vec.shape) == hilbert_dim:
        error_msg = 'Did you forget to expand the vec into the full ' + \
            'Hilbert space?'
        raise SizeMismatchError(error_msg)

    sys = sorted(sys)
    # sysstr makes hashing possible (for LRU cache)
    sysstr = ''.join(map(str, sys))
    indices = reorder_basis_dict(N, sysstr)
    indptr = np.array([0, hilbert_dim])
    reordered_vec = sparse.csc_matrix((vec, indices, indptr),
                                      shape=[hilbert_dim, 1]).toarray() \
        .reshape(2 ** N)

    reshaped_state = np.reshape(reordered_vec, [2 ** syslen, 2 ** envlen])
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
