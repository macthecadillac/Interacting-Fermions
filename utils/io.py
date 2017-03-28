"""
This file is part of spinsys.

Spinsys is free software: you can redistribute it and/or modify
it under the terms of the BSD 3-clause license. See LICENSE.txt
for exact terms and conditions.


This module provides facilities with regard to caching results to
disk. These functions are meant to speed up functions when no better
ways are available. Functions included:
    cache
    matcache

1-14-2017
"""

import os
from . import globalvar
import numpy as np
import scipy.sparse as ss
import functools
import tempfile


def matcache(function):
    """Caching wrapper for sparse matrix generating functions.
    Only works if the matrix in question is a
    1. numpy ndarray
    2. scipy CSC matrix or CSR matrix or BSR matrix
    When reading from cache, only numpy ndarrays and scipy CSC matrices
    are returned.
    """
    @functools.wraps(function)
    def wrapper(*args, **kargs):
        tmpdir = tempfile.gettempdir()
        cachedir = os.path.join(tmpdir, 'spinsys')
        cachefile = os.path.join(tmpdir, 'spinsys', '{}{}'
                                 .format(function.__name__, (args, kargs)))
        if not os.path.isdir(cachedir):
            os.mkdir(cachedir)
        try:
            return np.load(cachefile + '.npy')
        except (FileNotFoundError, EOFError):
            try:    # try loading a sparse matrix
                data = np.load(cachefile + '.data.npy')
                indices = np.load(cachefile + '.indices.npy')
                indptr = np.load(cachefile + '.indptr.npy')
                return ss.csc_matrix((data, indices, indptr))
            except FileNotFoundError:
                result = function(*args, **kargs)
                try:
                    # This will fail if the matrix is a scipy sparse matrix
                    np.save(cachefile, result, allow_pickle=False)
                except ValueError:
                    # For CSC and CSR matrices specifically
                    np.save(cachefile + '.data.npy', result.data,
                            allow_pickle=False)
                    np.save(cachefile + '.indices.npy', result.indices,
                            allow_pickle=False)
                    np.save(cachefile + '.indptr.npy', result.indptr,
                            allow_pickle=False)
                return result
    return wrapper


def cache_ram(function):
    """Wrapper for caching results into the Globals dictionary"""
    @functools.wraps(function)
    def wrapper(*args, **kargs):
        dict_key = '{}{}'.format(function.__name__, (args, kargs))
        try:
            result = globalvar.Globals[dict_key]
        except KeyError:
            result = function(*args, **kargs)
            globalvar.Globals[dict_key] = result
        return result
    return wrapper
