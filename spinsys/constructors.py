"""
This file is part of spinsys.

Spinsys is free software: you can redistribute it and/or modify
it under the terms of the BSD 3-clause license. See LICENSE.txt
for exact terms and conditions.


The constructors module provides several functions that generate
quantum mechanical operators and also one class that makes traversing
a lattice simpler. Function included are:
    raising
    lowering
    sigmax
    sigmay
    sigmaz

Class:
    SiteVector
"""

import copy
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


class SiteVector:

    def __init__(self, ordered_pair, Nx, Ny):
        self.x = ordered_pair[0]
        self.y = ordered_pair[1]
        self.Nx = Nx
        self.Ny = Ny

    def next_site(self):
        new_index = self.lattice_index + 1
        new_vec = copy.copy(self)
        new_vec.x = new_index % self.Nx
        new_vec.y = new_index // self.Nx
        return new_vec    # Returns a modified instance of self

    def xhop(self, stride):
        new_vec = copy.copy(self)
        new_vec.x = (self.x + stride) % self.Nx
        return new_vec

    def yhop(self, stride):
        new_vec = copy.copy(self)
        new_vec.y = (self.y + self.Nx * stride) % self.Ny
        return new_vec

    @property
    def lattice_index(self):
        return self.x + self.Nx * self.y
