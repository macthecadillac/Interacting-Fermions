"""
This file is part of spinsys.

Spinsys is free software: you can redistribute it and/or modify
it under the terms of the BSD 3-clause license. See LICENSE.txt
for exact terms and conditions.


The constructors module provides several functions that generate
quantum mechanical operators and also three classes that makes
traversing a lattice simpler. Functions included are:
    raising
    lowering
    sigmax
    sigmay
    sigmaz
Classes:
    SiteVector      # Not meant to be used directly
    PeriodicBCSiteVector
    OpenBCSiteVector
"""

import copy
import numpy as np
import scipy.sparse as sp
from spinsys.exceptions import OutOfBoundsError


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

    def __repr__(self):
        """Pretty print for debug purposes"""
        s = "{}(pair=({}, {}), Nx={}, Ny={})"
        # an ugly hack but it seems to be the only way to get the type
        #  of an instantiated child class directly from the parent class
        type_of_self = str(self.__class__).split('.')[-1][:-2]
        return s.format(type_of_self, self.x, self.y, self.Nx, self.Ny)

    def __eq__(self, other):
        dictcomp = self.__dict__ == other.__dict__
        typecomp = self.__class__ == other.__class__
        return dictcomp and typecomp

    def __lt__(self, other):
        """needed for the class to be sortable"""
        N = self.Nx * self.Ny
        otherN = other.Nx * other.Ny
        if not N == otherN:
            return N < otherN
        elif not self.y == other.y:
            return self.y < other.y
        else:
            return self.x < other.x

    def __hash__(self):
        return hash((self.__class__, self.x, self.y, self.Nx, self.Ny))

    def diff(self):
        raise NotImplementedError

    def __sub__(self, other):
        """Only as syntactic sugar. The function obviously does not
        return another SiteVecter
        """
        return self.diff(other)

    def next_site(self):
        # This method does not check for bounds so it is possible
        #  that you'll end up outside of the lattice. Use with care.
        new_index = self.lattice_index + 1
        new_vec = copy.copy(self)
        new_vec.x = new_index % self.Nx
        new_vec.y = new_index // self.Nx
        return new_vec    # Returns a modified instance of self

    @property
    def lattice_index(self):
        return self.x + self.Nx * self.y

    @property
    def coord(self):
        return (self.x, self.y)

    @classmethod
    def from_index(cls, index, Nx, Ny):
        x = index % Nx
        y = index // Nx
        return cls((x, y), Nx, Ny)


class PeriodicBCSiteVector(SiteVector):

    def __init__(self, ordered_pair, Nx, Ny):
        super().__init__(ordered_pair, Nx, Ny)

    def diff(self, other):
        def Δ(diffvar):
            # Assuming that only short range interaction exists
            #  (less than half the length of the lattice)
            d, N = diffvar
            if round(abs(d) / N) == 0:
                Δvar = d
            else:
                sgn = d // abs(d)
                Δvar = sgn * (N % abs(d))
            return Δvar

        diffx = (self.x - other.x, self.Nx)
        diffy = (self.y - other.y, self.Ny)
        Δx, Δy = Δ(diffx), Δ(diffy)
        return (Δx, Δy)

    def xhop(self, stride):
        new_vec = copy.copy(self)
        new_vec.x = (self.x + stride) % self.Nx
        return new_vec

    def yhop(self, stride):
        new_vec = copy.copy(self)
        new_vec.y = (self.y + stride) % self.Ny
        return new_vec


class OpenBCSiteVector(SiteVector):

    def __init__(self, ordered_pair, Nx, Ny):
        super().__init__(ordered_pair, Nx, Ny)

    def diff(self, other):
        """Finds the shortest distance from this site to the other"""
        Δx = self.x - other.x
        Δy = self.y - other.y
        return (Δx, Δy)

    def xhop(self, stride):
        new_vec = copy.copy(self)
        new_x = self.x + stride
        if new_x // self.Nx == self.x // self.Nx:
            new_vec.x = new_x
        else:
            raise OutOfBoundsError("Hopping off the lattice")
        return new_vec

    def yhop(self, stride):
        new_vec = copy.copy(self)
        new_y = self.y + stride
        if new_y // self.Ny == self.x // self.Ny:
            new_vec.y = new_y
        else:
            raise OutOfBoundsError("Hopping off the lattice")
        return new_vec
