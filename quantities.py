"""This module provides functions that calculate certain quantities of
the spin system.

1-15-2017
"""

import numpy as np
import scipy.sparse as ss
import spinsys
from spinsys.utils.globalvar import Globals as G
from spinsys.exceptions import SizeMismatchError


def adj_gap_ratio(sorted_eigvals):
    """
    Takes a list of eigenvalues that have been sorted low to high, finds the
    adjacent gap ratio for each set of 3 adjacent eigenvalues
    :param sorted_eigvals:
    :return adj_gap_ratio:
    """
    deltas = np.diff(sorted_eigvals)
    agrs = [min(deltas[i], deltas[i + 1]) / max(deltas[i], deltas[i + 1])
            for i in range(len(deltas) - 1)]
    return np.array(agrs)


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


def von_neumann_entropy(N, state, base='e'):
    """
    Find the bipartite von Neumann entropy of a given state.

    Args: "N" total system size
          "state" a vector that has to be dense
          "base" the base of the log function. It could be 'e', '2' or '10'
          "return_eigvals" a boolean that if set to True will return all
                the eigenvalues of the density matrix
    Returns: "entropy" a float
             Numpy array of all eigenvalues of the density matrix if
                "return_eigvals" is set to True. The eigenvalues are
                sorted from largest to smallest.
    """
    log = {'e': np.log, '2': np.log2, '10': np.log10}
    reduced_rho = bipartite_reduced_density_op(N, state)
    eigs = np.real(np.linalg.eigvalsh(reduced_rho))     # eigs are real anyways
    # Eigenvalues are always greater or equal to zero but rounding errors
    #  might sometimes introduce very small negative numbers
    eigs_filtered = np.array([eig if abs(eig) > 1e-8 else 1 for eig in eigs])
    entropy = sum(-eigs_filtered * log[base](eigs_filtered))
    G['reduced_density_op_eigs'] = np.sort(eigs)[::-1]
    return entropy


def half_chain_spin_dispersion(N, psi):
    """Find the half chain Sz dispersion.

    Args: "N" number of sites
          "psi" a vector. Must be 1D numpy array.
    Return: float
    """
    @spinsys.utils.io.cache_ram
    def first_half_chain_Sz_op_diagonal(N):
        """Returns the diagonal of the half chain Sz operator.
        (First half of the chain)
        """
        basis_set = spinsys.half.generate_complete_basis(N, 0)[0]
        conv = {1: 1, 0: -1}         # dict for conversion of 0's to -1's
        mat_dim = len(basis_set)     # size of the block for the given total <Sz>
        diagonal = np.empty(mat_dim)
        for i, basis in enumerate(basis_set):
            # convert 0's to -1's so the basis configuration would look like
            #  [-1, -1, 1, 1] instead of [0, 0, 1, 1]
            basis = [conv[s] for s in basis[: N // 2]]  # half chain total <Sz>
            diagonal[i] = 0.5 * sum(basis)
        return diagonal

    tot_Sz_diagonal = first_half_chain_Sz_op_diagonal(N)
    # product of a diagonal matrix and a vector is the same as element wise
    #  multiplcation of the diagonal of the said matrix with the vector
    Sz_expected = psi.conjugate().dot(tot_Sz_diagonal * psi)
    Sz2_expected = psi.conjugate().dot(tot_Sz_diagonal ** 2 * psi)
    return Sz2_expected - Sz_expected ** 2
