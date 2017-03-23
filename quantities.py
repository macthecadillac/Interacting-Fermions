"""
This file is part of spinsys.

Spinsys is free software: you can redistribute it and/or modify
it under the terms of the BSD 3-clause license. See LICENSE.txt
for exact terms and conditions.


This module provides functions that calculate certain quantities of
the spin system.

1-15-2017
"""

import numpy as np
import scipy as sp
import spinsys
from spinsys.utils.globalvar import Globals as G
from spinsys.half import bipartite_reduced_density_op


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


def half_chain_spin_dispersion(N, psi, curr_j=0):
    """Find the half chain Sz dispersion.

    Args: "N" number of sites
          "psi" a vector. Must be 1D numpy array.
    Return: float
    """
    @spinsys.utils.io.cache_ram
    def first_half_chain_Sz_op_diagonal(N, curr_j):
        """Returns the diagonal of the half chain Sz operator.
        (First half of the chain)
        """
        basis_set = spinsys.half.generate_complete_basis(N, curr_j)[0]
        conv = {1: 1, 0: -1}         # dict for conversion of 0's to -1's
        mat_dim = len(basis_set)     # size of the block for the given total <Sz>
        diagonal = np.empty(mat_dim)
        for i, basis in enumerate(basis_set):
            # convert 0's to -1's so the basis configuration would look like
            #  [-1, -1, 1, 1] instead of [0, 0, 1, 1]
            basis = [conv[s] for s in basis[: N // 2]]  # half chain total <Sz>
            diagonal[i] = 0.5 * sum(basis)
        return diagonal

    tot_Sz_diagonal = first_half_chain_Sz_op_diagonal(N, curr_j)
    # product of a diagonal matrix and a vector is the same as element wise
    #  multiplcation of the diagonal of the said matrix with the vector
    Sz_expected = psi.conjugate().dot(tot_Sz_diagonal * psi)
    Sz2_expected = psi.conjugate().dot(tot_Sz_diagonal ** 2 * psi)
    return Sz2_expected - Sz_expected ** 2


def spin_glass_order(N, psi):
    """Calculates the spin glass order 1/N * \sum_{i, j} |<S_i \cdot S_j>^2|
    NEEDS TO BE TESTED FOR BUGS

    Args: "N" number of sites
          "psi" a given state.
    Returns: float
    """
    @spinsys.utils.io.cache_ram
    def full_spin_operators(N):
        """Generate full spin operators for every site in the block
        diagonal basis for the <Sz>=0 subspace
        """
        Sx = spinsys.constructors.sigmax()
        Sy = spinsys.constructors.sigmay()
        Sz = spinsys.constructors.sigmaz()
        U = spinsys.half.similarity_trans_matrix(N)
        blk_size = int(round(sp.misc.comb(N, N / 2)))
        start = (2 ** N - blk_size) // 2
        end = start + blk_size

        # Generate full spin operators for every site
        full_S = [[spinsys.half.full_matrix(S, k, N) for k in range(N)]
                  for S in [Sx, Sy, Sz]]
        # Transform the spin operator into block diagonalized basis and
        #  truncate
        full_S = [map(lambda x: (U * x * U.T)[start: end, start: end], full_S[s])
                  for s in range(3)]
        return list(map(list, full_S))

    def Sj_dot_ket(j):
        """Generate Sj|psi> for every site for the current state."""
        # The outputs are numpy 1D arrays because of toarray and flatten
        return [full_S[s][j].dot(psi).toarray().flatten() for s in range(3)]

    def bra_dot_Si(i):
        """Generate <psi|Si for every site for the current state."""
        # The outputs are numpy 1D arrays because of toarray and flatten
        psi_conjtransp = psi.T.conjugate()
        return [psi_conjtransp.dot(full_S[s][i]).toarray().flatten()
                for s in range(3)]

    full_S = full_spin_operators(N)
    # Convert psi to sparse for better performance in the next stage
    psi = sp.sparse.csc_matrix(psi.reshape(psi.shape[0], 1))
    bra_dot_Sis = [bra_dot_Si(i) for i in range(N)]
    Sj_dot_kets = [Sj_dot_ket(j) for j in range(N)]
    sg_order_off_diag = sum(np.abs(bra_dot_Sis[i][s].dot(Sj_dot_kets[j][s])) ** 2
                            for s in range(3)
                            for i in range(N)
                            for j in range(i + 1, N)) * 2
    sg_order_diag = sum(np.abs(bra_dot_Sis[i][s].dot(Sj_dot_kets[i][s])) ** 2
                        for s in range(3) for i in range(N))
    return (sg_order_off_diag + sg_order_diag) / N
