"""
Module that stores functions that calculate entropy

1-12-2017
"""
from spinsys.utils.globalvar import Globals as G
import numpy as np


def bipartite_reduced_density_op(N, state):
    """
    Creates the density matrix using a state. Useful for calculating
    bipartite entropy

    Args: "state" a column vector that has to be dense (numpy.array)
          "N" total system size
    Returns: Numpy array
    """
    dim = 2 ** (N // 2)            # Dimensions of the reduced density matrices
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


# def entropy_time_evo_exactdiag(
#         N,
#         H,
#         psi_diag,
#         to_ord,
#         delta_ts,
#         start_time=0,
#         return_eigvals=False,
#         eigvals_range=None
# ):
#     """
#     This function plots the time evolution of bipartite von Neuman entropy
#     using exact diagonalization.
#
#     Args: "N" System size
#           "H" Hamiltonian
#           "psi_diag" initial state. Column vector in the spin block
#               basis arrangement
#           "to_ord" Dictionary that maps indices from the spin block
#               arrangement to that of the tensor product arrangement
#           "delta_ts" List/array of delta t
#           "save_eigvals" boolean that if set to True will save all the
#               eigenvalues of the density matrices to disk.
#           "eigvals_range" the number (from the greatest) of eigenvalues
#               to save
#     Returns: "entropy_evo" is a list of values to be plotted.
#              "eigvals" a numpy array of density matrix eigenvalues extracted
#                 during entropy calculations. Each row represents each point
#                 in time
#              "error" is the status of the state choosing function that
#                 is called from this function. If "error" is True, then no
#                 state of a zero total <Sz> with an energy density could be
#                 found for the current configuration.
#     """
#     points = len(delta_ts) + 1
#     entropy_evo = np.zeros(points)
#     eigvals = []
# 
#     # Setting up environment for exact diagonalization time evolution.
#     H = H.toarray()
#     E, V = np.linalg.eigh(H)
#     tm = common.TimeMachine(E, V, psi_diag)
# 
#     for plot_point in range(points):
#         # Evolve to start_time if start_time
#         if plot_point == 0:
#             psi_diag_evolved = tm.evolve(start_time)
#         else:
#             psi_diag_evolved = tm.evolve(delta_ts[plot_point - 1])
#         # Reorder basis before using the passing the state to
#         #  von_neumann_entropy since the algorithm only works with
#         #  bases in the tensor product order.
#         psi_ord_evolved = block.reorder_basis(N, psi_diag_evolved, to_ord)
# 
#         if return_eigvals:
#             entropy, eiglist = von_neumann_entropy(N, psi_ord_evolved,
#                                                    return_eigvals=True)
#             entropy_evo[plot_point] += entropy
#             eigvals.append(eiglist[:eigvals_range].copy())
#         else:
#             entropy = von_neumann_entropy(N, psi_ord_evolved)
#             entropy_evo[plot_point] += entropy
# 
#     if return_eigvals:
#         return entropy_evo, np.array(eigvals)
#     else:
#         return entropy_evo
