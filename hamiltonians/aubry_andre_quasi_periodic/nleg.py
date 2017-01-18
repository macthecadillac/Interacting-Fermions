"""This module provides functions to generate a N-legged hamiltonian
using tensor products

1-17-2017
"""

import numpy as np
from spinsys import half
from spinsys import constructors as c
from spinsys.utils.globalvar import Globals as G


def full_smatrices(N):
    """Generates operators in the full N-particle Hilbert space
    for the x, y, z directions
    """
    # try to load Sx, Sy, Sz from the Globals dictionary if they have been
    #  generated
    try:
        Sx, Sy, Sz = G['sigmas']
    except KeyError:
        Sx, Sy, Sz = c.sigmax(), c.sigmay(), c.sigmaz()
        G['sigmas'] = [Sx, Sy, Sz]

    # try to load the full Hilbert space operators from the Globals
    #  dictionary if they have been generated
    if not G.__contains__('full_S'):
        G['full_S'] = {}
    try:
        full_S = G['full_S'][N]
    except KeyError:
        full_S = [[half.full_matrix(S, k, N) for k in range(N)]
                  for S in [Sx, Sy, Sz]]
        G['full_S'][N] = full_S
    return full_S


def nearest_neighbor_interation(N, I, mode):
    """Generates the contribution from nearest neighber interations

    Args: "N" the number of sites
          "I" the number of legs
          "mode" "periodic" or "open"
    Returns: csc_matrix
    """
    total_interaction = []
    lower_bound = 0 if (mode == 'periodic' and not I == N) else I
    Sxs, Sys, Szs = full_smatrices(N)
    for k in range(lower_bound, N):
        total_interaction.append(Sxs[k - I] * Sxs[k])
        total_interaction.append(Sys[k - I] * Sys[k])
        total_interaction.append(Szs[k - I] * Szs[k])
    return sum(total_interaction)


def inter_chain_interation(N, I, mode):
    """Generates the contribution from interation between legs

    Args: "N" the number of sites
          "I" the number of legs
          "mode" "periodic" or "open"
    Returns: csc_matrix
    """
    total_interaction = []
    lower_bound = 0 if (mode == 'periodic' and not I == 1) else 1
    Sxs, Sys, Szs = full_smatrices(N)
    for i in range(lower_bound, I):
        for j in range(N // I):
            total_interaction.append(Sxs[i + j * I - 1] * Sxs[i + j * I])
            total_interaction.append(Sys[i + j * I - 1] * Sys[i + j * I])
            total_interaction.append(Szs[i + j * I - 1] * Szs[i + j * I])
    return sum(total_interaction)


def field_terms(N, h, c, phi, I):
    """Generates the contribution from the quasi-periodic field.

    Args: "N" the number of sites
          "h" disorder strength
          "c" transcendental number that finds itself popping up in the field
          "phi" phase
          "I" number of legs
    Returns: csc_matrix
    """
    Szs = full_smatrices(N)[2]
    sites = np.array(range(1, N // I + 1)).repeat(I, axis=0)
    field = h * np.cos(2 * np.pi * c * sites + phi)
    return sum(Szs * field)


def full_hamiltonian(N, h, c, phi, J1=1, J2=1, I=1, mode='open'):
    """Generates the entire hamiltonian

    Args: "N" the number of sites
          "h" disorder strength
          "c" transcendental number that finds itself popping up in the field
          "phi" phase
          "J1" interaction constant between nearest neighbors
          "J2" interation constant between adjacent sites across legs
          "I" number of legs
    Returns: csc_matrix
    """
    neighbor_terms = nearest_neighbor_interation(N, I, mode)
    inter_leg_terms = inter_chain_interation(N, I, mode)
    field_contribution = field_terms(N, h, c, phi, I)
    return J1 * neighbor_terms + J2 * inter_leg_terms + field_contribution
