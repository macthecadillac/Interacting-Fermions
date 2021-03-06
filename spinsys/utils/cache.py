"""
This file is part of spinsys.

Spinsys is free software: you can redistribute it and/or modify
it under the terms of the BSD 3-clause license. See LICENSE.txt
for exact terms and conditions.


This module provides a dictionary to store data that is cumbersome
to be passed around through function calls but at the same time should
only be computed once. The dictionary could hold the data for as long
as it is required.

3-29-2017
"""

import functools
import pickle
import os
import numpy as np
import scipy.sparse as ss
import tempfile


Globals = {}
tmpdir = tempfile.gettempdir()


def matcache(function):
    """Caching wrapper for sparse matrix generating functions.
    Only works if the matrix in question is a
    1. numpy ndarray
    2. scipy CSC matrix
    When reading from cache, only numpy ndarrays and scipy CSC matrices
    are returned.

    DOESN'T WORK WITH FUNCTIONS WITH KEYWORD ARGUMENTS
    """
    @functools.wraps(function)
    def wrapper(*args):
        hashed = str(hash((function.__name__, args)))
        cachedir = os.path.join(tmpdir, 'spinsys')
        cachefile = os.path.join(tmpdir, 'spinsys', hashed)
        if not os.path.isdir(cachedir):
            os.mkdir(cachedir)
        try:
            # try to load a dense matrix
            return np.load(cachefile + '.npy', allow_pickle=False)
        except FileNotFoundError:
            try:    # try to load a sparse matrix
                data = np.load(cachefile + '.data.npy', allow_pickle=False)
                indices = np.load(cachefile + '.indices.npy', allow_pickle=False)
                indptr = np.load(cachefile + '.indptr.npy', allow_pickle=False)
                with open(cachefile + '.shape', 'rb') as fh:
                    shape = pickle.load(fh)
                return ss.csc_matrix((data, indices, indptr), shape=shape)
            except FileNotFoundError:
                result = function(*args)
                if isinstance(result, np.ndarray):
                    np.save(cachefile + '.npy', result, allow_pickle=False)
                elif isinstance(result, ss.csc.csc_matrix):
                    np.save(cachefile + '.data.npy', result.data,
                            allow_pickle=False)
                    np.save(cachefile + '.indices.npy', result.indices,
                            allow_pickle=False)
                    np.save(cachefile + '.indptr.npy', result.indptr,
                            allow_pickle=False)
                    with open(cachefile + '.shape', 'wb') as fh:
                        pickle.dump(result.shape, fh)
                else:
                    msg = "Supported types of matrix/array are numpy.ndarray " + \
                        "and scipy.sparse.csc.csc_matrix"
                    raise ValueError(msg)
                return result
    return wrapper


def cache_to_ram(function):
    """Wrapper for caching results into the Globals dictionary.
    Works like functools.lru_cache but it also works on private
    (nested) functions which lru_cache does not. Currently does
    not have LRU limits.

    DOESN'T WORK WITH FUNCTIONS WITH KEYWORD ARGUMENTS
    """
    @functools.wraps(function)
    def wrapper(*args):
        dict_key = hash((function.__name__, args))
        try:
            result = Globals[dict_key]
        except KeyError:
            result = function(*args)
            Globals[dict_key] = result
        return result
    return wrapper
