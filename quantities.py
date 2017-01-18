"""This module provides functions that calculate certain quantities of
the spin system.

1-15-2017
"""

import numpy as np
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
          "state" a column vector that has to be dense
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
