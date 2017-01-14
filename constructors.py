"""The constructors module provides commonly used constructors for
quantum computation. Constructors provided in this module:
    raising
    lowering
    sigmax
    sigmay
    sigmaz

1-14-2017
"""

import numpy as np
import scipy.sparse as sp


def raising(spin=0.5):
    """Returns the raising operator.

    Args: 'spin' a float
    Returns: csc_matrix
    """
    dim = round(2 * spin + 1)
    data = []
    for i in np.arange(-spin, spin, 1):
        m = -i - 1
        data.append(np.sqrt(spin * (spin + 1) - m * (m + 1)))
    row_ind = [int(round(spin + i)) for i in np.arange(-spin, spin, 1)]
    col_ind = [int(round(spin + i + 1)) for i in np.arange(-spin, spin, 1)]
    return sp.csc_matrix((data, (row_ind, col_ind)), shape=(dim, dim))


def lowering(spin=0.5):
    """Returns the lowering operator.

    Args: 'spin' a float
    Returns: csc_matrix
    """
    return raising(spin).T


def sigmax(spin=0.5):
    """Returns the Sx operator.

    Args: 'spin' a float
    Returns: csc_matrix
    """
    return 0.5 * (raising(spin) + lowering(spin))


def sigmay(spin=0.5):
    """Returns the Sy operator.

    Args: 'spin' a float
    Returns: csc_matrix
    """
    return -0.5j * (raising(spin) - lowering(spin))


def sigmaz(spin=0.5):
    """Returns the Sz operator.

    Args: 'spin' a float
    Returns: csc_matrix
    """
    dim = round(2 * spin + 1)
    data = [i for i in np.arange(spin, -spin - 1, -1)]
    ind = list(range(dim))
    return sp.csc_matrix((data, (ind, ind)), shape=(dim, dim))
