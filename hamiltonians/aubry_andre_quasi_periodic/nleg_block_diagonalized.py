"""
This function generates a block diagonalized Hamiltonian for a
multi-legged Hamiltonian with a pseudo-random, quasi-periodic field.

1-11-2017
"""

import numpy as np
import scipy.sparse as sp
from spinsys.half import misc
from spinsys import utils


def _abs_diff(x, y):
    return abs(x - y)


def diagonal_single_block(N, h, c, phi, J1, J2, I, current_j):
    """
    Creates the diagonal of a block of the Hamiltonian.

    Args: 'N' System length
          'h' Field strength
          'c' Irrational/transcendental number happens to be in the field
          'phi' Phase shift
          'J1' Interaction constant between sites along a leg
          'J2' Interaction constant between legs
          'I' Number of legs
          'current_j' Total <Sz>
    Returns: Sparse matrix (dia matrix)
    """
    basis_set = misc.create_complete_basis(N, current_j)[0]
    blksize = len(basis_set)
    diagonal = np.zeros(blksize)
    sites = np.array(range(1, N // I + 1)).repeat(I, axis=0)
    field = h * np.cos(2 * np.pi * c * sites + phi)
    d = {1: 1, 0: -1}         # For conversion of 0s to -1s in the basis later
    for i, b in enumerate(basis_set):
        # ***Interaction Terms***
        # Number of repeated 1s and 0s separated by I
        #  Compute absolute values of differences of pairs separated by I
        #  and sum. Periodic BC (horizontal interaction).
        diff_pairs = sum(map(_abs_diff, b, b[I:] + b[:I]))
        same_pairs = N - diff_pairs
        diagonal[i] += 0.25 * J1 * (same_pairs - diff_pairs)

        # Number of different adjacent 1s and 0s.
        #  Closed BC (interaction between legs)
        if I > 1:
            comp = [m for m in range(N) if not (m + 1) % I == 0]
            diff_pairs = sum(map(_abs_diff, [b[m] for m in comp],
                                 [b[m + 1] for m in comp]))
            same_pairs = len(comp) - diff_pairs
            diagonal[i] += 0.25 * J2 * (same_pairs - diff_pairs)

        # ***Field Terms***
        diagonal[i] += 0.5 * sum([d[m] for m in b] * field)
    return sp.diags(diagonal, 0, dtype=complex)


@utils.io.matcache
def off_diagonal_single_block(N, J1, J2, I, current_j):
    """
    Creates the off diagonals of a block of the Hamiltonian.

    Args: 'N' System size
          'J1' Coupling constant between neighboring sites
          'J2' Coupling constant between legs
          'I' Number of legs
          'current_j' Total <Sz>
    Returns: Sparse matrix (lil matrix)
    """
    def non_zero_element(i, bi, pair, J):
        """Sets non-zero elements in the matrix"""
        bj = bi[:]
        bj[pair[0]], bj[pair[1]] = bj[pair[1]], bj[pair[0]]
        if not sum(map(_abs_diff, bi, bj)) == 0:
            j = to_diag[dim - utils.misc.bin_to_dec(bj) - 1]
            off_diagonal[i, j] += 0.5 * J

    dim = 2 ** N
    basis_set, to_diag, to_ord = misc.create_complete_basis(N, current_j)
    blksize = len(basis_set)
    off_diagonal = sp.lil_matrix((blksize, blksize), dtype=complex)

    # Pairs of elements to inspect. I_pairs = pairs I sites apart
    #  adjacent_pairs = pairs adjacent to each other
    I_pairs = [(N + i if i < 0 else i, i + I) for i in range(-I, N - I)]
    adjacent_pairs = [(i, i + 1) for i in range(N - 1) if not (i + 1) % I == 0]

    for i, bi in enumerate(basis_set):
        # Flipping of elements I sites apart
        for pair in I_pairs:
            non_zero_element(i, bi, pair, J1)

        # Flipping of elements adjacent to each other
        if I > 1:
            for pair in adjacent_pairs:
                non_zero_element(i, bi, pair, J2)
    return off_diagonal


def single_block(N, h, c, phi, J1=1, J2=1, I=2, current_j=0):
    """
    Creates a block of the Hamiltonian

    Args: 'N' System length
          'h' Field strength
          'c' Irrational/transcendental number happens to be in the field
          'phi' Phase shift
          'J1' Interaction constant between sites along a leg
          'J2' Interaction constant between legs
          'I' Number of legs
          'current_j' Total <Sz>
    Returns: Sparse matrix (CSC matrix)
    """
    diagonals = diagonal_single_block(N, h, c, phi, J1, J2, I, current_j)
    off_diagonals = off_diagonal_single_block(N, J1, J2, I, current_j)
    return diagonals + off_diagonals
