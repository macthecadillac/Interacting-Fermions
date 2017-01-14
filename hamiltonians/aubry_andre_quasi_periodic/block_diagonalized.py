"""
This function generates a block diagonalized Hamiltonian for a
fermionic Aubry-Andre Hamiltonian with a pseudo-random, quasi-periodic field.
Periodic BC

8-13-2016
"""

from spinsys import utils
import numpy as np
from scipy.sparse import lil_matrix
from scipy.misc import comb
import os
import pickle


def bin2dec(l):
    """
    Convert a list of 1's and 0's, a binary representation of a number,
    into a decimal number.

    Args: "l" is a list/array of 1's and 0's.
    Return: An integer in decimal.
    """
    dec = int(''.join(map(str, l)), 2)
    return dec


def basis_set(N, blk_sz, j_max, current_j):
    """
    This function creates a list of the full set of basis for the
    current total <Sz>. It also returns a dictionary that matches the
    coordinates of a configuration in the product state basis against
    its index in the current individual spin block.

    Args: "N" is the size of the particle system
          "blk_sz" is the side length of the block corresponding to this
          specific total spin
          "j_max" is the highest total spin
          "current_j" is the current total spin
    Return: 1. A full set of spin configurations in a list of lists of 1's and
               0's. 1 represents spin up and 0 represents spin down.
            2. A dictionary that maps the locations of spin configurations in
               the product state ordering with the **local** position of
               the configurations. (Local means within the current spin block)
    """
    # TODO: Move to aubry_andre_common because it is not specific to this
    #       Hamiltonian.
    # TODO: Remove blk_sz and j_max as arguments since they could be
    #       easily calculated from N and current_j.

    # Check and see if there is disk cache for this basis set. If so,
    #  load from disk.
    bs_fname = 'cache/basisset_' + str(N) + 'spins_' + str(current_j) + 'Sz'
    bidict_fname = 'cache/basisinddict_' + str(N)
    bidict_fname += 'spins_' + str(current_j) + 'Sz'
    if os.path.isfile(bs_fname) and os.path.isfile(bidict_fname):
        data0 = open(bs_fname, 'rb')
        data1 = open(bidict_fname, 'rb')
        current_j_basis_set = pickle.load(data0)
        basis_ind_dict = pickle.load(data1)
        data0.close()
        data1.close()

    else:
        # Find all the binary representations of the current j.
        D = 2**N
        basis_set_seed = [0] * N
        basis_ind_dict = {}
        basis_index = 0
        for n in range(j_max + current_j):
            basis_set_seed[N - 1 - n] = 1
        current_j_basis_set = []
        if blk_sz != 1:
            for i in range(blk_sz):
                current_j_basis_set.append(utils.misc.binary_permutation(
                                           basis_set_seed)[:])
                # Create a dictionary that links the decimal
                #  representation of a basis and its position in this
                #  particular way of basis ordering.
                basis_ind_dict[D - 1 - bin2dec(basis_set_seed)] = basis_index
                basis_index += 1
        else:
            # The permute function cannot permute lists for which there is
            #  only one permutation.
            current_j_basis_set.append(basis_set_seed)

        # Save the basis sets and their respective dictionaries
        #  for future uses.
        with open(bs_fname, 'wb') as data0:
            pickle.dump(current_j_basis_set, data0, pickle.HIGHEST_PROTOCOL)
        with open(bidict_fname, 'wb') as data1:
            pickle.dump(basis_ind_dict, data1, pickle.HIGHEST_PROTOCOL)
    return current_j_basis_set, basis_ind_dict


def blk_off_diag_ut_nocache(options):
    """
    Provides the upper half of the off-diagonal elements of one
    block of the Hamiltonian corresponding to the given total Sz.
    Returns an upper triangle sparse matrix.

    This function does NOT save the results on disk.

    Args: "options" is a set of arguments enclosed in a Python list. "options"
          must contain three objects: "N", "total_Sz" and "J." "N" is the
          size of the particle system. "total_Sz" is the total spin of this
          particular block of off-diagonal elements for the Hamiltonian. "J"
          is the coupling constant. If not specified, set it to 1.
    Return: A sparse matrix (lil_matrix) containing the upper half of the
            off-diagonal elements of the specified total <Sz>.
    """
    [N, total_Sz, J] = options
    D = 2**N
    # Side length of the current block.
    j_max = int(round(0.5 * N))
    blk_sz = int(round(comb(N, j_max - total_Sz)))
    current_j_basis_set, basis_ind_dict = basis_set(N, blk_sz,
                                                    j_max, total_Sz)
    H_curr_blk_off_diag = lil_matrix((blk_sz, blk_sz), dtype=complex)

    # Fill in the off-diagonal elements for the current block.
    for i in range(blk_sz):
        k = N - 1
        nz_list = []
        while k >= 0:
            curr_bs = current_j_basis_set[i][:]
            while True:
                if abs(curr_bs[k] - curr_bs[k - 1]) == 1:
                    break
                else:
                    k -= 1
            curr_bs[k], curr_bs[k - 1] = curr_bs[k - 1], curr_bs[k]
            curr_ind = basis_ind_dict[D - 1 - bin2dec(curr_bs)]
            if curr_ind > i:
                nz_list.append(curr_ind)
            k -= 1
        for k in nz_list:
            H_curr_blk_off_diag[i, k] = 0.5 * J
    return H_curr_blk_off_diag


def blk_off_diag_ut(N, total_Sz, J=1):
    """
    Provides the upper half of the off-diagonal elements of one
    block of the Hamiltonian corresponding to the given total Sz.
    Returns an upper triangle sparse matrix.

    This function automatically saves/loads cache to/from disk.

    Args: "N" is the size of the entire particle system.
          "total_Sz" is the total z direction spin of the current block.
          "J" is the coupling constant. It is defaulted to 1.
    Return: A sparse matrix (lil_matrix) containing the upper half of the
            off-diagonal elements of the specified total <Sz>.
    """
    options = [N, total_Sz, J]
    fname = 'cache/H_block_off_diag_ut' + str(N) + 'spins_J' + str(J) + '_'
    fname += str(total_Sz) + 'Sz.mat'
    H_curr_blk_off_diag_ut = qm.iowrapper(blk_off_diag_ut_nocache,
                                          fname, options)
    return H_curr_blk_off_diag_ut


def blk_off_diag(N, total_Sz, J=1):
    """
    Provides the off-diagonal elements of one block of the Hamiltonian
    corresponding to the given total Sz. Returns a sparse matrix.
    (A full matrix, not just an upper triangular matrix)

    This function automatically saves/loads cache to/from disk.

    Args: "N" is the size of the entire particle system.
          "total_Sz" is the total z direction spin of the current block.
          "J" is the coupling constant. It is defaulted to 1.
    Return: A sparse matrix (lil_matrix) containing the off-diagonal
            elements of the specified total <Sz>.
    """
    H_curr_blk_off_diag = blk_off_diag_ut(N, total_Sz, J)
    H_curr_blk_off_diag += H_curr_blk_off_diag.transpose()
    return H_curr_blk_off_diag


def blk_diag(N, h, c, total_Sz, phi=0):
    """
    Provides the diagonal elements of one block of the Hamiltonian
    corresponding to the given total Sz. Returns a sparse matrix.

    This function does not load/save cache to/from disk.

    Args: "N" is the size of the entire particle system.
          "h" is the strength of the pseudo-random field.
          "c" is the angular frequency of the field.
          "total_Sz" is the total z direction spin of the current block.
          "phi" is the phase shift of the field. Defaults to 0.
    Return: A sparse matrix (lil_matrix) containing the diagonal elements
            of the specified total <Sz>.
    """
    # Side length of the current block.
    j_max = int(round(0.5 * N))
    blk_sz = int(round(comb(N, int(j_max - total_Sz))))
    current_j_basis_set, basis_ind_dict = basis_set(N, blk_sz,
                                                    j_max, total_Sz)

    # Create the field
    field = np.empty(N)
    for i in range(N):
        field[i] = h * np.cos(c * 2 * np.pi * (i + 1) + phi)

    H_curr_blk_diag = lil_matrix((blk_sz, blk_sz), dtype=complex)

    # Fill in the diagonal elements for the current block.
    for i in range(blk_sz):
        h_sum = 0
        tot = 0
        for k in range(N):
            # h(i)Sz(i)
            if current_j_basis_set[i][k] == 0:
                h_sum -= field[k]
            elif current_j_basis_set[i][k] == 1:
                h_sum += field[k]

            # Sz(i)Sz(i+1)
            diff = current_j_basis_set[i][N - 1 - k]
            diff -= current_j_basis_set[i][N - 2 - k]
            tot += abs(diff)
        imb = N - 2 * tot
        H_curr_blk_diag[i, i] = imb * 0.25 + 0.5 * h_sum
    return H_curr_blk_diag


def blk_full(N, h, c, total_Sz, phi=0, J=1):
    """
    Provides one full block of the Hamiltonian corresponding to the given
    total Sz, complete with diagonal and off-diagonal elements.
    Returns a sparse matrix.

    This function indirectly makes use of the caching function of
    blk_off_diag.

    Args: "N" is the size of the entire particle system.
          "h" is the strength of the pseudo-random field.
          "c" is the angular frequency of the field.
          "total_Sz" is the total z direction spin of the current block.
          "phi" is the phase shift of the field. Defaults to 0.
          "J" is the coupling constant. Defaults to 1.
    Return: A sparse matrix (lil_matrix) containing all the elements of
            the entire block of the Hamiltonian corresponding to the
            specified total <Sz>.
    """
    H_curr_blk_diag = blk_diag(N, h, c, total_Sz, phi)
    H_curr_blk_off_diag = blk_off_diag(N, total_Sz, J)
    return H_curr_blk_diag + H_curr_blk_off_diag


def aubry_andre_H_off_diag_nocache(options):
    """
    Provides the off-diagonal elements of the full Hamiltonian.
    Returns a sparse matrix.

    This function does NOT save the results on disk.

    Args: "options" is a set of arguments enclosed in a Python list. "options"
          must contain two objects: "N" and "J." "N" is the
          size of the particle system. "J" is the coupling constant.
          If not specified, set it to 1.
    Return: A sparse matrix (lil_matrix) containing all the off-diagonal
            elements of the full Hamiltonian.
    """
    [N, J] = options
    D = 2**N
    H_off_diag = lil_matrix((D, D), dtype=complex)
    j_max = int(round(0.5 * N))
    current_j = j_max
    pos = 0
    while current_j >= -1 * j_max:
        blk_sz = int(round(comb(N, j_max - current_j)))
        if blk_sz != 1:
            H_j_off_diag = blk_off_diag(N, current_j, J)
            H_off_diag[pos:pos + blk_sz,
                       pos:pos + blk_sz] += H_j_off_diag
        pos += blk_sz
        current_j -= 1
    return H_off_diag


def aubry_andre_H_off_diag(N, J=1):
    """
    Provides the off-diagonal elements of the full Hamiltonian.
    Returns a sparse matrix.

    This function automatically saves/loads the results to/from disk.

    Args: "N" is the size of the partical system.
          "J" is the coupling constant. Defaults to 1.
    Return: A sparse matrix (lil_matrix) containing all the off-diagonal
            elements of the full Hamiltonian.
    """
    options = [N, J]
    fname = 'cache/aubry_H_off_diag_' + str(N) + 'spins_J' + str(J) + '.mat'
    H_off_diag = qm.iowrapper(aubry_andre_H_off_diag_nocache, fname, options)
    return H_off_diag


def aubry_andre_H_diag_nocache(options):
    """
    Provides the diagonal elements of the full Hamiltonian.
    Returns a sparse matrix.

    This function does NOT save/load results to/from disk.

    Args: "options" is a set of arguments enclosed in a Python list. "options"
          must contain four objects: "N", "h", "c" and "phi." "N" is the
          size of the particle system. "h" is the strength of the pseudo-
          random field. "c" is the angular frequency of the field. "phi"
          is the phase shift of the field.
    Return: A sparse matrix (lil_matrix) containing all the diagonal
            elements of the full Hamiltonian.
    """
    [N, h, c, phi] = options
    D = 2**N
    H_diag = lil_matrix((D, D), dtype=complex)
    j_max = int(round(0.5 * N))
    current_j = j_max
    pos = 0
    while current_j >= -1 * j_max:
        blk_sz = int(round(comb(N, int(j_max - current_j))))
        H_j_diag = blk_diag(N, h, c, current_j, phi)
        H_diag[pos:pos + blk_sz,
               pos:pos + blk_sz] += H_j_diag
        pos += blk_sz
        current_j -= 1
    return H_diag


def aubry_andre_H_diag(N, h, c, phi=0):
    """
    Provides the diagonal elements of the full Hamiltonian.
    Returns a sparse matrix.

    This function automatically saves/loads the results to/from disk.

    Args: "N" is the size of the particle system.
          "h" is the strength of the pseudo-random field.
          "c" is the angular frequency of the field.
          "phi" is the phase shift. Defaults to 0.
    Return: A sparse matrix (lil_matrix) containing all the diagonal
            elements of the full Hamiltonian.
    """
    options = [N, h, c, phi]
    fname = 'cache/aubry_H_diag_' + str(N) + 'spins' + '_h' + str(h)
    fname += '_c' + str(c) + '_phi' + str(phi) + '.mat'
    H_off_diag = qm.iowrapper(aubry_andre_H_diag_nocache, fname, options)
    return H_off_diag


def aubry_andre_H(N, h, c, phi, J=1):
    """
    Provides the full Hamiltonian.
    Returns a sparse matrix.

    This function indirectly makes use of the caching functions of
    aubry_andre_H_diag and aubry_andre_H_off_diag.

    Args: "N" is the size of the particle system.
          "h" is the strength of the pseudo-random field.
          "c" is the angular frequency of the field.
          "phi" is the phase shift.
          "J" is the coupling constant. Defaults to 1.
    Return: The full Hamiltonian in a sparse matrix format. (lil_matrix)
    """
    H_off_diag = aubry_andre_H_off_diag(N, J)
    H_diag = aubry_andre_H_diag(N, h, c, phi)
    H = H_diag + H_off_diag
    return H
