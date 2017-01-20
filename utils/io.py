"""This module provides facilities with regard to caching results to
disk. These functions are meant to speed up functions when no better
ways are available. Functions included:
    cache
    matcache

1-14-2017
"""

import os
from . import globalvar
from scipy import io
import functools
import tempfile
import msgpack


def cache(function):
    """Generic caching wrapper. Should work on any kind of I/O"""
    @functools.wraps(function)
    def wrapper(*args, **kargs):
        # for cross platform compatibility
        tmpdir = tempfile.gettempdir()
        cachedir = os.path.join(tmpdir, 'spinsys')
        cachefile = os.path.join(tmpdir, 'spinsys', '{}{}'
                                 .format(function.__name__, (args, kargs)))
        if not os.path.isdir(cachedir):
            os.mkdir(cachedir)
        try:
            with open(cachefile, 'rb') as c:
                return msgpack.load(c)
        except FileNotFoundError:
            result = function(*args, **kargs)
            with open(cachefile, 'wb') as c:
                msgpack.dump(result, c)
            return result
    return wrapper


def matcache(function):
    """Caching wrapper for sparse matrix generating functions."""
    @functools.wraps(function)
    def wrapper(*args, **kargs):
        tmpdir = tempfile.gettempdir()
        cachedir = os.path.join(tmpdir, 'spinsys')
        cachefile = os.path.join(tmpdir, 'spinsys', '{}{}'
                                 .format(function.__name__, (args, kargs)))
        if not os.path.isdir(cachedir):
            os.mkdir(cachedir)
        try:
            return io.loadmat(cachefile)['i']
        except FileNotFoundError:
            result = function(*args, **kargs)
            io.savemat(cachefile, {'i': result}, appendmat=False)
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
