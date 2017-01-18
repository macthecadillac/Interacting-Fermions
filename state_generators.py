import numpy as np
from scipy.sparse.linalg import eigsh, ArpackNoConvergence
from spinsys.exceptions import NoConvergence
from spinsys import half


def generate_eigenpairs(N, H, num_psi):
    """
    Generate Eigenpairs using Shift Inversion Method
    :param N:
    :param H:
    :param num_psi:
    :return:
    """
    try:
        E = np.sort(eigsh(H, k=2, which='BE', return_eigenvectors=False))
        target_E = np.mean(E)
        psis = []
        eigs, eigvecs = eigsh(H, k=int(num_psi), which='LM', sigma=target_E)
    except ArpackNoConvergence:
        raise NoConvergence

    for i in range(num_psi):
        psi = half.expand_and_reorder(N, eigvecs[:, i])
        psis.append(psi)
    return eigs, psis
