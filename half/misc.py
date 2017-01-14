"""This module provides other functions that pertain to half-spin
systems but do not belong to any other modules. Functions indlucded:
    full_matrix
    create_complete_basis

1-13-2017
"""

from scipy.misc import comb
import scipy.sparse as ss
from spinsys import utils


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
    S = ss.csc_matrix(matrix) if not ss.issparse(matrix) else matrix
    if k == 0:
        S_full = ss.kron(S, ss.eye(dim ** (N - 1)))
    elif k == 1:
        S_full = ss.eye(dim)
        S_full = ss.kron(S_full, S)
        S_full = ss.kron(S_full, ss.eye(dim ** (N - 2)))
    else:
        S_full = ss.eye(dim)
        S_full = ss.kron(S_full, ss.eye(dim ** (k - 1)))
        S_full = ss.kron(S_full, S)
        S_full = ss.kron(S_full, ss.eye(dim ** (N - k - 1)))

    return S_full


def create_complete_basis(N, current_j):
    """Creates a complete basis for the current total <Sz>"""
    dim = 2 ** N
    spin_ups = int(round(0.5 * N + current_j))
    spin_downs = N - spin_ups
    blksize = int(round(comb(N, spin_ups)))
    basis_seed = [0] * spin_downs + [1] * spin_ups
    basis = basis_seed
    # "to_diag" is a dict that maps ordinary indices to block diagonalized
    #  indices. "to_ord" is the opposite.
    basis_set, to_diag, to_ord = [], {}, {}
    for i in range(blksize):
        try:
            basis = utils.misc.binary_permutation(basis)
        except IndexError:                # When current_j is N // 2 or -N // 2
            pass
        basis_set.append(basis[:])
        decimal_basis = utils.misc.bin_to_dec(basis)
        # i is the index within only this block
        to_diag[dim - decimal_basis - 1] = i
        to_ord[i] = dim - decimal_basis - 1
    return basis_set, to_diag, to_ord
