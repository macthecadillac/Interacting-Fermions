import numpy as np
from spinsys import constructors as c
from spinsys.utils.globalvar import Globals as G


def full_smatrices(N):
    spin = 0.5
    Sx, Sy, Sz = c.sigmax(), c.sigmay(), c.sigmaz()
    Sxs, Sys, Szs = [], [], []
    for k in range(N):
        Sxs.append(qm.get_full_matrix(Sx, k, N))
        Sys.append(qm.get_full_matrix(Sy, k, N))
        Szs.append(qm.get_full_matrix(Sz, k, N))
    return Sxs, Sys, Szs


def nearest_neighbor_interation(N, I):
    total_interaction = []
    for k in range(N):
        total_interaction.append(g.Sxs[k - I] * g.Sxs[k])
        total_interaction.append(g.Sys[k - I] * g.Sys[k])
        total_interaction.append(g.Szs[k - I] * g.Szs[k])
    return sum(total_interaction)


def inter_chain_interation(N, I):
    total_interaction = []
    for k in range(N):
        if not (k + 1) % I == 0:
            total_interaction.append(g.Sxs[k] * g.Sxs[k + 1])
            total_interaction.append(g.Sys[k] * g.Sys[k + 1])
            total_interaction.append(g.Szs[k] * g.Szs[k + 1])
    return sum(total_interaction)


def field_terms(N, h, c, phi, I):
    Szs = g.Szs[:]
    sites = np.array(range(1, N // I + 1)).repeat(I, axis=0)
    field = h * np.cos(2 * np.pi * c * sites + phi)
    for i, Sz in enumerate(g.Szs):
        Sz *= field[i]
    return sum(Szs)


def full_hamiltonian(N, h, c, phi, J1, J2, I):
    neighor_terms = nearest_neighbor_interation(N, I)
    inter_leg_terms = inter_chain_interation(N, I)
    field_contribution = field_terms(N, h, c, phi, I)
    return J1 * neighor_terms + J2 * inter_leg_terms + field_contribution
