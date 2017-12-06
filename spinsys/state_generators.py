"""
This file is part of spinsys.

Spinsys is free software: you can redistribute it and/or modify
it under the terms of the BSD 3-clause license. See LICENSE.txt
for exact terms and conditions.
"""

import numpy as np
from scipy.sparse.linalg import eigsh, ArpackNoConvergence
from spinsys.exceptions import NoConvergence, NotFoundError
from spinsys import half


def generate_eigenpairs(N, H, num_psi, expand=True, enden=0.5):
    """Generates eigenvalue-eigenvector pairs around a specified
    energy density

    Args: "N" number of sites
          "H" Hamiltonian of the system
          "num_psi" number of eigenvectors to find
          "expand" option to expand the state vector into full
            Hilbert space and reorder into the tensor product
            basis ordering. Boolean
    Returns: 1. eigen-energies. 1D numpy array
             2. eigen-vectors. Python list of numpy 1D arrays
    """
    try:
        E = np.sort(eigsh(H, k=2, which='BE', return_eigenvectors=False))
        target_E = enden * E[1] + (1 - enden) * E[0]
        eigs, eigvecs = eigsh(H, k=int(num_psi), which='LM', sigma=target_E)
    except ArpackNoConvergence:
        raise NoConvergence

    if expand:
        psis = []
        for i in range(num_psi):
            psi = half.expand_and_reorder(N, eigvecs[:, i])
            psis.append(psi)
    else:
        psis = [eigvecs[:, i] for i in range(num_psi)]
    return eigs, psis


def generate_product_state(H, enden=0.5, tol=1e-6):
    """Generates a product state with an energy density close to the
    specified value.  The state will be returned in whatever basis H
    is in.

    Args: "H" Hamiltonian of the system in block diagonal form.
          "enden" energy density
          "tol" tolerance. It is highly unlikely to find a product state
            with energy density exactly equal to 0.5. This sets the
            tolerable range from 0.5
    Returns: 1D numpy array
    """
    def energy_density(E, Emin, Emax):
        return (E - Emin) / (Emax - Emin)

    veclen = H.shape[0]
    try:
        Emin, Emax = np.sort(eigsh(H, k=2, which='BE',
                             return_eigenvectors=False))
    except ArpackNoConvergence:
        raise NoConvergence
    # This works because we are using basis states of H. <H> with basis states
    #  always return diagonal elements
    energy_expvals = H.diagonal()
    for i, E_exp in enumerate(energy_expvals):
        if abs(energy_density(E_exp, Emin, Emax) - enden) < tol:
            break
    else:
        raise NotFoundError
    product_state = np.zeros(veclen)
    product_state[i] = 1
    return product_state
